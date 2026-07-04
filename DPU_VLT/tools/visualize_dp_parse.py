#!/usr/bin/env python3
import argparse
import base64
import html
import json
import mimetypes
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def read_text(path):
    return Path(path).read_text(encoding="utf-8").strip()


def resolve_tnllt(root, seq_name, frame_id):
    seq_dir = Path(root) / seq_name
    text_path = seq_dir / "language.txt"
    image_path = seq_dir / "imgs" / f"{frame_id:05d}.png"
    return read_text(text_path), image_path


def resolve_tnl2k(root, split, seq_name, frame_id):
    seq_dir = Path(root) / split / seq_name
    text_path = seq_dir / "language.txt"
    img_dir = seq_dir / "imgs"
    frames = sorted([p for p in img_dir.iterdir() if p.is_file()])
    if not frames:
        raise FileNotFoundError(f"No image frames found under {img_dir}")
    idx = max(0, min(len(frames) - 1, frame_id - 1))
    return read_text(text_path), frames[idx]


def resolve_inputs(args):
    if args.qwen_key and args.qwen_cache:
        qwen_entry = load_qwen_entry(args.qwen_cache, args.qwen_key, args.seq, args.frame_id)
        if qwen_entry is not None:
            args.seq = args.seq or str(qwen_entry.get("seq_name", ""))
            args.frame_id = int(qwen_entry.get("frame_id", args.frame_id))
            if not args.image:
                args.image = str(qwen_entry.get("image_path", ""))
    if args.text:
        raw_text = args.text
        image_path = Path(args.image) if args.image else None
        return raw_text, image_path
    if args.dataset == "tnllt":
        return resolve_tnllt(args.root, args.seq, args.frame_id)
    if args.dataset == "tnl2k":
        return resolve_tnl2k(args.root, args.split, args.seq, args.frame_id)
    raise ValueError("--text is required when --dataset=generic")


def load_qwen_cache(cache_path):
    if not cache_path:
        return {}
    data = json.loads(Path(cache_path).read_text(encoding="utf-8"))
    entries = data.get("entries", data)
    return entries if isinstance(entries, dict) else {}


def load_qwen_entry(cache_path, qwen_key="", seq_name="", frame_id=None):
    entries = load_qwen_cache(cache_path)
    if not entries:
        return None
    if qwen_key and qwen_key in entries:
        return entries[qwen_key]
    if seq_name and frame_id is not None:
        exact_key = f"{seq_name}:{int(frame_id)}"
        if exact_key in entries:
            return entries[exact_key]
        candidates = []
        for key, value in entries.items():
            if not isinstance(value, dict):
                continue
            if str(value.get("seq_name", "")) != str(seq_name):
                continue
            try:
                cur_frame = int(value.get("frame_id", str(key).rsplit(":", 1)[-1]))
            except Exception:
                continue
            if cur_frame <= int(frame_id):
                candidates.append((cur_frame, value))
        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[-1][1]
    return None


def qwen_triplet_from_entry(entry):
    if not isinstance(entry, dict):
        return None
    return {
        "target": str(entry.get("qwen_target", entry.get("target", ""))).strip(),
        "concepts": str(entry.get("qwen_concept", entry.get("concepts", ""))).strip(),
        "background": str(entry.get("qwen_background", entry.get("background", ""))).strip(),
        "anchor_frame_id": entry.get("frame_id", ""),
        "raw_response": entry.get("raw_response", ""),
    }


def triplet_complete(triplet):
    if not isinstance(triplet, dict):
        return False
    return all(str(triplet.get(k, "")).strip() for k in ("target", "concepts", "background"))


def entry_complete(entry):
    if not isinstance(entry, dict):
        return False
    keys = ("target", "concepts", "background", "qwen_target", "qwen_concept", "qwen_background")
    return all(str(entry.get(k, "")).strip() for k in keys)


def image_to_data_uri(image_path):
    if not image_path:
        return ""
    path = Path(image_path)
    if not path.is_file():
        return ""
    mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def infer_raw_text_from_image_path(image_path):
    if not image_path:
        return ""
    img_path = Path(image_path)
    candidates = []
    if img_path.parent.name in {"imgs", "img"}:
        candidates.append(img_path.parent.parent / "language.txt")
    candidates.append(img_path.parent / "language.txt")
    for path in candidates:
        if path.is_file():
            return read_text(path)
    return ""


def normalize_phrase_tokens(value):
    if isinstance(value, list):
        value = " ".join(str(x) for x in value)
    return {x.strip(".,:;!?()[]{}\"'").lower() for x in str(value).split() if x.strip()}


def token_roles(tokens, triplet):
    role_sets = {
        "target": normalize_phrase_tokens(triplet.get("target", "")),
        "concepts": normalize_phrase_tokens(triplet.get("concepts", [])),
        "background": normalize_phrase_tokens(triplet.get("background", [])),
    }
    roles = []
    for tok in tokens:
        low = tok.strip(".,:;!?()[]{}\"'").lower()
        cur = [name for name, vals in role_sets.items() if low and low in vals]
        roles.append(cur)
    return roles


def build_arcs(tokens, heads, deprels):
    arcs = []
    for dep_idx in range(1, len(tokens)):
        head_idx = heads[dep_idx] if dep_idx < len(heads) else 0
        if head_idx < 0 or head_idx >= len(tokens):
            head_idx = 0
        rel = deprels[dep_idx] if dep_idx < len(deprels) else "dep"
        arcs.append({
            "head_index": head_idx,
            "head": tokens[head_idx],
            "dependent_index": dep_idx,
            "dependent": tokens[dep_idx],
            "relation": rel,
        })
    return arcs


def escape_attr(value):
    return html.escape(str(value), quote=True)


def make_svg(tokens, arcs, roles):
    step = 104
    left = 70
    baseline = 225
    top_pad = 28
    width = max(900, left * 2 + step * max(1, len(tokens) - 1))
    height = 300
    role_color = {
        "target": "#f2b705",
        "concepts": "#4f8cff",
        "background": "#3fb06f",
    }
    x_pos = [left + i * step for i in range(len(tokens))]
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="300" xmlns="http://www.w3.org/2000/svg">',
        '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#445"/></marker></defs>',
        '<rect x="0" y="0" width="100%" height="100%" fill="#fbfbfc"/>',
    ]
    for arc in arcs:
        h = arc["head_index"]
        d = arc["dependent_index"]
        x1 = x_pos[h]
        x2 = x_pos[d]
        dist = abs(d - h)
        arc_height = 34 + min(125, dist * 18)
        y_ctrl = baseline - arc_height
        label_x = (x1 + x2) / 2
        label_y = y_ctrl - 6
        parts.append(
            f'<path d="M{x1},{baseline - 28} Q{label_x},{y_ctrl} {x2},{baseline - 28}" '
            'fill="none" stroke="#48505f" stroke-width="1.6" marker-end="url(#arrow)"/>'
        )
        parts.append(
            f'<text x="{label_x}" y="{label_y}" text-anchor="middle" font-size="12" fill="#29313d">'
            f'{escape_attr(arc["relation"])}</text>'
        )
    for i, tok in enumerate(tokens):
        x = x_pos[i]
        y = baseline
        token_roles = roles[i] if i < len(roles) else []
        fill = "#ffffff"
        stroke = "#bdc3cf"
        if token_roles:
            fill = role_color.get(token_roles[0], fill)
            stroke = "#252a33"
        parts.append(f'<line x1="{x}" y1="{baseline - 20}" x2="{x}" y2="{baseline - 8}" stroke="#8992a3"/>')
        parts.append(f'<rect x="{x - 43}" y="{y - 2}" width="86" height="34" rx="6" fill="{fill}" stroke="{stroke}"/>')
        text_fill = "#151922" if fill != "#4f8cff" else "#ffffff"
        parts.append(
            f'<text x="{x}" y="{y + 20}" text-anchor="middle" font-size="13" fill="{text_fill}">'
            f'{escape_attr(tok)}</text>'
        )
        parts.append(f'<text x="{x}" y="{y + 52}" text-anchor="middle" font-size="11" fill="#687182">{i}</text>')
    parts.append(f'<text x="{left}" y="{top_pad}" font-size="12" fill="#687182">Head to dependent arcs; token 0 is ROOT.</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def make_html(payload, svg):
    image_path = payload.get("image_path") or ""
    image_block = "<div class=\"muted\">No image path provided.</div>"
    image_src = payload.get("image_data_uri") or image_path
    if image_src:
        image_block = f'<img class="frame" src="{escape_attr(image_src)}" alt="sequence frame">'
    triplet = payload["triplet"]
    qwen_triplet = payload.get("qwen_triplet") or {}
    qwen_block = ""
    if qwen_triplet:
        qwen_block = f"""
      <div class="label refine-label">Qwen Refined Triplet</div>
      <div class="triplet">
        <div class="field qwen"><div class="label">Target</div>{escape_attr(qwen_triplet.get("target", ""))}</div>
        <div class="field qwen"><div class="label">Concepts</div>{escape_attr(qwen_triplet.get("concepts", ""))}</div>
        <div class="field qwen"><div class="label">Background</div>{escape_attr(qwen_triplet.get("background", ""))}</div>
      </div>
      <p class="muted">Qwen anchor frame: {escape_attr(qwen_triplet.get("anchor_frame_id", ""))}</p>
"""
    arcs_rows = "\n".join(
        "<tr>"
        f"<td>{a['dependent_index']}</td><td>{escape_attr(a['dependent'])}</td>"
        f"<td>{a['head_index']}</td><td>{escape_attr(a['head'])}</td>"
        f"<td>{escape_attr(a['relation'])}</td>"
        "</tr>"
        for a in payload["arcs"]
    )
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>DP Parse Visualization - {escape_attr(payload.get("seq_name", "sample"))}</title>
<style>
body {{ margin: 0; font-family: Arial, Helvetica, sans-serif; color: #151922; background: #f3f5f8; }}
.wrap {{ max-width: 1220px; margin: 0 auto; padding: 24px; }}
.panel {{ background: #fff; border: 1px solid #d9dee8; border-radius: 8px; padding: 18px; margin-bottom: 16px; }}
.grid {{ display: grid; grid-template-columns: minmax(320px, 0.95fr) minmax(420px, 1.4fr); gap: 16px; align-items: start; }}
.frame {{ width: 100%; max-height: 520px; object-fit: contain; background: #111; border-radius: 6px; border: 1px solid #d9dee8; }}
.sentence {{ font-size: 18px; line-height: 1.45; }}
.triplet {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 14px; }}
.field {{ border: 1px solid #d9dee8; border-radius: 6px; padding: 10px; background: #fbfbfc; }}
.field.qwen {{ background: #f7fbf8; border-color: #b9dec5; }}
.label {{ font-size: 12px; color: #687182; text-transform: uppercase; margin-bottom: 5px; }}
.refine-label {{ margin-top: 16px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th, td {{ border-bottom: 1px solid #e6e9ef; padding: 7px 8px; text-align: left; }}
th {{ color: #4b5565; background: #f7f8fa; }}
.muted {{ color: #687182; }}
</style>
</head>
<body>
<div class="wrap">
  <div class="grid">
    <div class="panel">
      <div class="label">Sequence Frame</div>
      {image_block}
      <p class="muted">{escape_attr(image_path)}</p>
    </div>
    <div class="panel">
      <div class="label">Raw Query</div>
      <div class="sentence">{escape_attr(payload["raw_text"])}</div>
      <div class="triplet">
        <div class="field"><div class="label">Target</div>{escape_attr(triplet.get("target", ""))}</div>
        <div class="field"><div class="label">Concepts</div>{escape_attr(" ".join(triplet.get("concepts", [])) if isinstance(triplet.get("concepts", []), list) else triplet.get("concepts", ""))}</div>
        <div class="field"><div class="label">Background</div>{escape_attr(" ".join(triplet.get("background", [])) if isinstance(triplet.get("background", []), list) else triplet.get("background", ""))}</div>
      </div>
      {qwen_block}
    </div>
  </div>
  <div class="panel">
    <div class="label">Dependency Parse</div>
    {svg}
  </div>
  <div class="panel">
    <div class="label">Arc Table</div>
    <table>
      <thead><tr><th>Dep Id</th><th>Dependent</th><th>Head Id</th><th>Head</th><th>Relation</th></tr></thead>
      <tbody>{arcs_rows}</tbody>
    </table>
  </div>
</div>
</body>
</html>
"""


def build_parser_model(args, parser_cls):
    return parser_cls(
        checkpoint_path=args.checkpoint,
        cache_dir=args.cache_dir,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        lstm_layers=args.lstm_layers,
        ffnn_dim=args.ffnn_dim,
        device=args.device,
        use_spacy=bool(args.use_spacy),
        spacy_model=args.spacy_model,
    )


def write_visualization(args, parser_model, raw_text, image_path, qwen_entry, seq_name, frame_id, name):
    tokens, heads, deprels, upos_tags = parser_model.parse(raw_text)
    triplet = parser_model.extract_triplet(raw_text)
    arcs = build_arcs(tokens, heads, deprels)
    roles = token_roles(tokens, triplet)

    payload = {
        "dataset": args.dataset,
        "seq_name": seq_name,
        "frame_id": frame_id,
        "image_path": str(Path(image_path).resolve()) if image_path else "",
        "image_data_uri": "" if args.no_embed_image else image_to_data_uri(image_path),
        "raw_text": raw_text,
        "tokens": tokens,
        "heads": heads,
        "deprels": deprels,
        "upos": upos_tags,
        "arcs": arcs,
        "triplet": triplet,
        "qwen_triplet": qwen_triplet_from_entry(qwen_entry),
    }
    svg = make_svg(tokens, arcs, roles)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{name}.json"
    html_path = out_dir / f"{name}.html"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path.write_text(make_html(payload, svg), encoding="utf-8")
    return html_path, json_path, triplet


def iter_batch_entries(args):
    entries = load_qwen_cache(args.qwen_cache)
    if not entries:
        raise ValueError("--batch-from-qwen-cache requires a non-empty --qwen-cache")
    items = []
    for key, entry in entries.items():
        if not isinstance(entry, dict):
            continue
        if args.require_complete_fields and not entry_complete(entry):
            continue
        image_path = entry.get("image_path", "")
        if image_path and not Path(image_path).is_file():
            continue
        seq_name = str(entry.get("seq_name", str(key).rsplit(":", 1)[0]))
        try:
            frame_id = int(entry.get("frame_id", str(key).rsplit(":", 1)[-1]))
        except Exception:
            continue
        items.append((seq_name, frame_id, key, entry))
    items.sort(key=lambda x: (x[0], x[1]))
    if args.one_per_seq:
        selected = {}
        for seq_name, frame_id, key, entry in items:
            selected.setdefault(seq_name, (seq_name, frame_id, key, entry))
        items = list(selected.values())
    if args.max_samples > 0:
        items = items[:args.max_samples]
    return items


def run_batch(args, parser_model):
    items = iter_batch_entries(args)
    manifest = []
    print(f"[INFO] batch samples={len(items)}")
    for idx, (seq_name, frame_id, key, entry) in enumerate(items, start=1):
        image_path = entry.get("image_path", "")
        raw_text = args.text or infer_raw_text_from_image_path(image_path)
        if not raw_text:
            print(f"[WARN] skip {key}: cannot find raw language text")
            continue
        safe_seq = seq_name.replace("/", "_")
        name = f"{args.name_prefix or args.dataset}_{safe_seq}_f{frame_id:05d}"
        try:
            html_path, json_path, dp_triplet = write_visualization(
                args=args,
                parser_model=parser_model,
                raw_text=raw_text,
                image_path=image_path,
                qwen_entry=entry,
                seq_name=seq_name,
                frame_id=frame_id,
                name=name,
            )
        except Exception as exc:
            print(f"[WARN] failed {key}: {exc}")
            continue
        manifest.append({
            "key": key,
            "seq_name": seq_name,
            "frame_id": frame_id,
            "html": str(html_path),
            "json": str(json_path),
            "raw_text": raw_text,
            "dp_triplet": dp_triplet,
            "qwen_triplet": qwen_triplet_from_entry(entry),
        })
        if idx % max(1, args.log_interval) == 0:
            print(f"[INFO] processed {idx}/{len(items)}")
    manifest_path = Path(args.output_dir) / f"{args.name_prefix or args.dataset}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] wrote manifest {manifest_path} ({len(manifest)} items)")


def main():
    parser = argparse.ArgumentParser(description="Visualize DP parsing, extracted triplet, and the corresponding video frame.")
    parser.add_argument("--dataset", default="tnllt", choices=["tnllt", "tnl2k", "generic"])
    parser.add_argument("--root", default="", help="Dataset root. TNLLT: root/seq; TNL2K: root/split/seq.")
    parser.add_argument("--split", default="test", help="TNL2K split, usually train or test.")
    parser.add_argument("--seq", default="", help="Sequence name, e.g. test_001_xxx or class/seq for two-level TNL2K.")
    parser.add_argument("--frame-id", type=int, default=1, help="1-based frame id to display.")
    parser.add_argument("--text", default="", help="Raw query for generic mode or ad-hoc visualization.")
    parser.add_argument("--image", default="", help="Image path for generic mode or ad-hoc visualization.")
    parser.add_argument("--qwen-cache", default="", help="Optional Qwen refine cache JSON with entries keyed by seq:frame.")
    parser.add_argument("--qwen-key", default="", help="Optional exact Qwen cache key, e.g. Bird1:49. Also supplies image path if --image is empty.")
    parser.add_argument("--no-embed-image", action="store_true", help="Reference image path instead of embedding base64 image data in HTML.")
    parser.add_argument("--batch-from-qwen-cache", action="store_true", help="Generate visualizations from all entries in --qwen-cache.")
    parser.add_argument("--one-per-seq", action="store_true", help="In batch mode, keep only the first usable entry of each sequence.")
    parser.add_argument("--require-complete-fields", action="store_true", help="In batch mode, require DP and Qwen target/concepts/background fields in cache to be non-empty.")
    parser.add_argument("--max-samples", type=int, default=0, help="In batch mode, limit number of generated samples. 0 means no limit.")
    parser.add_argument("--name-prefix", default="", help="In batch mode, output filename prefix.")
    parser.add_argument("--log-interval", type=int, default=25, help="In batch mode, progress print interval.")
    parser.add_argument("--checkpoint", required=True, help="DPModel2 checkpoint path.")
    parser.add_argument("--cache-dir", required=True, help="HF datasets cache dir used by DPModel2Parser.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--embed-dim", type=int, default=300)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lstm-layers", type=int, default=3)
    parser.add_argument("--ffnn-dim", type=int, default=256)
    parser.add_argument("--use-spacy", type=int, default=1)
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument("--hf-endpoint", default="", help="Optional Hugging Face endpoint mirror, e.g. https://hf-mirror.com.")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "output" / "dp_vis"))
    parser.add_argument("--name", default="", help="Output basename. Defaults to dataset_seq_frame.")
    args = parser.parse_args()

    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    from DPModel2.parse_text import DPModel2Parser

    parser_model = build_parser_model(args, DPModel2Parser)
    if args.batch_from_qwen_cache:
        run_batch(args, parser_model)
        return

    raw_text, image_path = resolve_inputs(args)
    qwen_entry = load_qwen_entry(args.qwen_cache, args.qwen_key, args.seq, args.frame_id) if args.qwen_cache else None
    if qwen_entry is not None and not image_path and qwen_entry.get("image_path"):
        image_path = Path(qwen_entry["image_path"])
    safe_seq = args.seq.replace("/", "_") if args.seq else "sample"
    name = args.name or f"{args.dataset}_{safe_seq}_f{args.frame_id:05d}"
    html_path, json_path, _ = write_visualization(
        args=args,
        parser_model=parser_model,
        raw_text=raw_text,
        image_path=image_path,
        qwen_entry=qwen_entry,
        seq_name=args.seq,
        frame_id=args.frame_id,
        name=name,
    )
    print(f"[DONE] wrote {html_path}")
    print(f"[DONE] wrote {json_path}")


if __name__ == "__main__":
    main()

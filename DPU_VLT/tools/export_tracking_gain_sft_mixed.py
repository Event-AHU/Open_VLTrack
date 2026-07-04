#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


SYSTEM_PROMPTS = {
    "v1": "You are a structured language refiner for visual tracking.",
    "v2": "You are refining structured tracking text for the current search region.",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export mixed SFT dataset from tracking gain parts: positives from gain threshold, negatives from worst drops."
    )
    parser.add_argument(
        "--score-paths",
        nargs="+",
        required=True,
        help="One or more tracking gain json files, e.g. part0..part3",
    )
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-stats", default="")
    parser.add_argument("--pos-gain-th", type=float, default=0.003)
    parser.add_argument("--neg-gain-th", type=float, default=0.0, help="Negative pool condition: gain < neg_gain_th")
    parser.add_argument("--neg-topk", type=int, default=1000)
    parser.add_argument("--neg-ratio", type=float, default=0.0, help="If > 0, select negatives by ratio: ceil(num_pos * neg_ratio).")
    parser.add_argument("--min-neg-topk", type=int, default=0, help="Minimum negatives when --neg-ratio is used.")
    parser.add_argument("--system-prompt-version", default="v2", choices=sorted(SYSTEM_PROMPTS.keys()))
    parser.add_argument("--drop-concept-eq-target", action="store_true")
    parser.add_argument("--drop-pos-concept-eq-baseline", action="store_true", help="Drop positive sample if candidate concept equals baseline concept.")
    parser.add_argument("--sort-by-gain", action="store_true", help="Sort final records by gain descending")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize_text(x):
    return str(x or "").strip().lower()


def build_user_prompt(text_struct):
    target = str(text_struct.get("target", "")).strip()
    concept = str(text_struct.get("concepts", "")).strip()
    background = str(text_struct.get("background", "")).strip()
    return "\n".join([
        f"Target: {target}",
        f"Concept: {concept}",
        f"Background: {background}",
        "",
        "Task:",
        "Update concept and background for the current search image.",
        "Return JSON only as:",
        '{"concept":"...", "background":"..."}',
    ])


def make_record(sample, out_text, gain, sample_type, system_prompt_version):
    base = sample.get("baseline_text", {}) if isinstance(sample, dict) else {}
    return {
        "seq_name": sample.get("seq_name", ""),
        "frame_id": int(sample.get("frame_id", -1)),
        "image_path": sample.get("image_path", ""),
        "gain": float(gain),
        "sample_type": sample_type,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPTS[system_prompt_version],
            },
            {
                "role": "user",
                "content": build_user_prompt({
                    "target": base.get("target", ""),
                    "concepts": base.get("concepts", ""),
                    "background": base.get("background", ""),
                }),
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "concept": str(out_text.get("concepts", "")).strip(),
                        "background": str(out_text.get("background", "")).strip(),
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    }


def extract_best_gain(sample):
    g = sample.get("best_gain", None)
    if g is not None:
        return float(g)
    best = sample.get("best_candidate", None)
    if isinstance(best, dict) and best.get("gain", None) is not None:
        return float(best["gain"])
    return None


def extract_best_text(sample):
    best = sample.get("best_candidate", None)
    if isinstance(best, dict):
        t = best.get("text", None)
        if isinstance(t, dict):
            return {
                "concepts": str(t.get("concepts", "")).strip(),
                "background": str(t.get("background", "")).strip(),
            }
    return None


def extract_baseline_text(sample):
    base = sample.get("baseline_text", {}) if isinstance(sample, dict) else {}
    return {
        "concepts": str(base.get("concepts", "")).strip(),
        "background": str(base.get("background", "")).strip(),
    }


def sample_key(sample):
    return f"{sample.get('seq_name','')}:{int(sample.get('frame_id',-1))}"


def main():
    args = parse_args()

    out_path = Path(args.output_jsonl)
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"Output exists: {out_path}. Use --overwrite")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_samples = []
    loaded = []
    for p in args.score_paths:
        path = Path(p)
        if not path.exists():
            print(f"[WARN] missing score file: {path}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        samples = data.get("samples", []) if isinstance(data, dict) else []
        all_samples.extend(samples)
        loaded.append((str(path), len(samples)))

    positives = []
    negative_pool = []
    dropped_pos_concept_eq_baseline = 0
    dropped_pos_concept_eq_target = 0
    for s in all_samples:
        g = extract_best_gain(s)
        if g is None:
            continue

        if g > args.pos_gain_th:
            best_text = extract_best_text(s)
            if best_text is None:
                continue
            if args.drop_concept_eq_target:
                tgt = normalize_text((s.get("baseline_text", {}) or {}).get("target", ""))
                cpt = normalize_text(best_text.get("concepts", ""))
                if tgt and cpt and tgt == cpt:
                    dropped_pos_concept_eq_target += 1
                    continue
            if args.drop_pos_concept_eq_baseline:
                base_cpt = normalize_text((s.get("baseline_text", {}) or {}).get("concepts", ""))
                cpt = normalize_text(best_text.get("concepts", ""))
                if cpt and base_cpt and cpt == base_cpt:
                    dropped_pos_concept_eq_baseline += 1
                    continue
            positives.append((g, s, best_text))

        if g < args.neg_gain_th:
            negative_pool.append((g, s))

    # most negative first
    negative_pool.sort(key=lambda x: x[0])
    if args.neg_ratio > 0:
        target_neg = int(max(args.min_neg_topk, round(len(positives) * args.neg_ratio)))
        negatives = negative_pool[: max(0, target_neg)]
    else:
        negatives = negative_pool[: max(0, args.neg_topk)]

    records = []
    seen = set()

    for g, s, out_text in positives:
        k = sample_key(s)
        if k in seen:
            continue
        seen.add(k)
        records.append(make_record(s, out_text, g, "positive", args.system_prompt_version))

    for g, s in negatives:
        k = sample_key(s)
        if k in seen:
            continue
        seen.add(k)
        base_text = extract_baseline_text(s)
        records.append(make_record(s, base_text, g, "negative", args.system_prompt_version))

    if args.sort_by_gain:
        records.sort(key=lambda r: float(r.get("gain", 0.0)), reverse=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = {
        "loaded_files": loaded,
        "total_samples": len(all_samples),
        "pos_gain_th": args.pos_gain_th,
        "neg_gain_th": args.neg_gain_th,
        "neg_topk": args.neg_topk,
        "neg_ratio": args.neg_ratio,
        "min_neg_topk": args.min_neg_topk,
        "positive_candidates": len(positives),
        "negative_pool": len(negative_pool),
        "negative_selected": len(negatives),
        "dropped_pos_concept_eq_target": dropped_pos_concept_eq_target,
        "dropped_pos_concept_eq_baseline": dropped_pos_concept_eq_baseline,
        "final_records": len(records),
        "final_positive": sum(1 for r in records if r.get("sample_type") == "positive"),
        "final_negative": sum(1 for r in records if r.get("sample_type") == "negative"),
    }

    if args.output_stats:
        stats_path = Path(args.output_stats)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"[DONE] wrote stats to {stats_path}")

    print(f"[DONE] wrote {len(records)} records to {out_path}")
    print(f"[STAT] {json.dumps(stats, ensure_ascii=False)}")


if __name__ == "__main__":
    main()

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from DPModel2.parse_text import DPModel2Parser


def load_sequence_list(split):
    base = REPO_ROOT / "lib" / "train" / "data_specs"
    mapping = {
        "train": base / "got10k_train_split.txt",
        "val": base / "got10k_val_split.txt",
        "train_full": base / "got10k_train_full_split.txt",
        "vottrain": base / "got10k_vot_train_split.txt",
        "votval": base / "got10k_vot_val_split.txt",
    }
    if split == "all":
        names = []
        for key in ("train", "val", "train_full", "vottrain", "votval"):
            names.extend(load_sequence_list(key))
        return sorted(set(names))
    file_path = mapping[split]
    with open(file_path, "r", encoding="utf-8") as f:
        # split file stores sequence indices
        return [int(line.strip()) for line in f if line.strip()]


def load_seq_names_from_list(got10k_root):
    list_path = Path(got10k_root) / "list.txt"
    with open(list_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def parse_concise_file(txt_path, parser_model):
    out = {}
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                frame_key, raw = line.split(" ", 1)
                frame_id = int(frame_key)
            except Exception:
                continue
            triplet = parser_model.extract_triplet(raw)
            out[str(frame_id)] = {
                "raw": raw,
                "target": str(triplet.get("target", "")),
                "concepts": [str(x) for x in triplet.get("concepts", [])],
                "background": [str(x) for x in triplet.get("background", [])],
            }
    return out


def main():
    parser = argparse.ArgumentParser(description="Precompute GOT-10K parsed text cache from got-10k_train_concise.")
    parser.add_argument("--got10k_root", required=True, help="Path to GOT-10K split root (e.g. .../full_data/train)")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument(
        "--split",
        default="train_full",
        choices=["train", "val", "train_full", "vottrain", "votval", "all"],
    )
    parser.add_argument("--checkpoint", required=True, help="DPModel2 checkpoint path")
    parser.add_argument("--cache_dir", required=True, help="HF datasets cache dir used by DPModel2Parser")
    parser.add_argument("--device", default="cpu", help="Parser device, recommend cpu for offline preprocessing")
    parser.add_argument("--embed_dim", type=int, default=300)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--lstm_layers", type=int, default=3)
    parser.add_argument("--ffnn_dim", type=int, default=256)
    parser.add_argument("--use_spacy", type=int, default=1)
    parser.add_argument("--spacy_model", default="en_core_web_sm")
    args = parser.parse_args()

    seq_names = load_seq_names_from_list(args.got10k_root)
    seq_ids = load_sequence_list(args.split)

    parser_model = DPModel2Parser(
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

    root = Path(args.got10k_root)
    concise_root = root / "got-10k_train_concise"
    result = {}
    num_ok = 0
    num_fail = 0

    for n, seq_id in enumerate(seq_ids, start=1):
        if seq_id < 0 or seq_id >= len(seq_names):
            print(f"[WARN] invalid seq_id={seq_id}")
            num_fail += 1
            continue
        seq_name = seq_names[seq_id]
        txt_path = concise_root / f"{seq_name}.txt"
        if not txt_path.is_file():
            print(f"[WARN] missing concise text file: {txt_path}")
            num_fail += 1
            continue
        try:
            frames_map = parse_concise_file(txt_path, parser_model)
            if not frames_map:
                print(f"[WARN] empty parsed frames: {seq_name}")
                num_fail += 1
                continue
            result[seq_name] = frames_map
            num_ok += 1
        except Exception as e:
            print(f"[ERROR] {seq_name}: {e}")
            num_fail += 1
        if n % 100 == 0:
            print(f"[INFO] processed {n}/{len(seq_ids)} sequences")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[DONE] saved {num_ok} sequence entries to {out_path}; failures={num_fail}")


if __name__ == "__main__":
    main()

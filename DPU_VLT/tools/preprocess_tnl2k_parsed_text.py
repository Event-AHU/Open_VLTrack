#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from DPModel2.parse_text import DPModel2Parser


def load_sequence_list(split_root):
    sequence_list = []
    subset_list = [f for f in os.listdir(split_root)
                   if os.path.isdir(os.path.join(split_root, f)) and f != 'revised_annotations']
    if len(subset_list) > 14:
        return sorted(subset_list)
    for x in subset_list:
        sub_sequence_list_path = os.path.join(split_root, x)
        for seq in os.listdir(sub_sequence_list_path):
            sequence_list.append(os.path.join(x, seq))
    return sorted(sequence_list)


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


def parse_test_language_file(txt_path, parser_model):
    with open(txt_path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    triplet = parser_model.extract_triplet(raw)
    return {
        "raw": raw,
        "target": str(triplet.get("target", "")),
        "concepts": [str(x) for x in triplet.get("concepts", [])],
        "background": [str(x) for x in triplet.get("background", [])],
    }


def main():
    parser = argparse.ArgumentParser(description="Precompute TNL2K parsed text cache from tnl2k_train_concise.")
    parser.add_argument("--tnl2k-root", required=True, help="Path to TNL2K root, e.g. /rydata/dataset/SOT/TNL2k")
    parser.add_argument("--split", default="train", choices=["train", "test"], help="train: frame-wise concise parsing; test: sequence-level language.txt parsing.")
    parser.add_argument("--output", required=True, help="Output JSON path")
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

    split_root = Path(args.tnl2k_root) / args.split
    concise_root = Path(args.tnl2k_root) / "tnl2k_train_concise"
    seq_names = load_sequence_list(str(split_root))

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

    result = {}
    num_ok = 0
    num_fail = 0

    for n, seq_name in enumerate(seq_names, start=1):
        try:
            if args.split == "train":
                txt_path = concise_root / f"{seq_name}.txt"
                if not txt_path.is_file():
                    print(f"[WARN] missing concise text file: {txt_path}")
                    num_fail += 1
                    continue
                parsed_item = parse_concise_file(txt_path, parser_model)
                if not parsed_item:
                    print(f"[WARN] empty parsed frames: {seq_name}")
                    num_fail += 1
                    continue
            else:
                txt_path = split_root / seq_name / "language.txt"
                if not txt_path.is_file():
                    print(f"[WARN] missing test language file: {txt_path}")
                    num_fail += 1
                    continue
                parsed_item = parse_test_language_file(txt_path, parser_model)
                if not parsed_item.get("raw", ""):
                    print(f"[WARN] empty test language: {seq_name}")
                    num_fail += 1
                    continue
            result[seq_name] = parsed_item
            num_ok += 1
        except Exception as e:
            print(f"[ERROR] {seq_name}: {e}")
            num_fail += 1
        if n % 100 == 0:
            print(f"[INFO] processed {n}/{len(seq_names)} sequences")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[DONE] saved {num_ok} sequence entries to {out_path}; failures={num_fail}")


if __name__ == "__main__":
    main()

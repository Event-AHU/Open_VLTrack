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


def load_sequence_list(split):
    if split == "train":
        split_file = REPO_ROOT / "lib" / "train" / "data_specs" / "lasot_train_split.txt"
    elif split == "test":
        # The evaluation class hard-codes the protocol-II test split. Reuse it to avoid duplicating the list.
        from lib.test.evaluation.lasotdataset import LaSOTDataset

        return list(LaSOTDataset().sequence_list)
    else:
        raise ValueError(f"Unknown split: {split}")

    with open(split_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def parse_framewise_concise(txt_path, parser_model):
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


def parse_sequence_nlp(txt_path, parser_model):
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
    parser = argparse.ArgumentParser(description="Precompute LaSOT parsed text cache.")
    parser.add_argument("--lasot-root", required=True, help="Path to LaSOTBenchmark root")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--embed_dim", type=int, default=300)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--lstm_layers", type=int, default=3)
    parser.add_argument("--ffnn_dim", type=int, default=256)
    parser.add_argument("--use_spacy", type=int, default=1)
    parser.add_argument("--spacy_model", default="en_core_web_sm")
    args = parser.parse_args()

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

    root = Path(args.lasot_root)
    seq_names = load_sequence_list(args.split)
    result = {}
    num_ok = 0
    num_fail = 0

    for n, seq_name in enumerate(seq_names, start=1):
        try:
            class_name = seq_name.split("-")[0]
            if args.split == "train":
                txt_path = root / "lasot_train_concise" / f"{seq_name}.txt"
                if not txt_path.is_file():
                    print(f"[WARN] missing concise text file: {txt_path}")
                    num_fail += 1
                    continue
                parsed_item = parse_framewise_concise(txt_path, parser_model)
            else:
                txt_path = root / class_name / seq_name / "nlp.txt"
                if not txt_path.is_file():
                    print(f"[WARN] missing nlp file: {txt_path}")
                    num_fail += 1
                    continue
                parsed_item = parse_sequence_nlp(txt_path, parser_model)

            if not parsed_item:
                print(f"[WARN] empty parsed item: {seq_name}")
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

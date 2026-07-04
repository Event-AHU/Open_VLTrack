#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from DPModel2.parse_text import DPModel2Parser
from lib.utils.string_utils import clean_string


def parse_language_file(txt_path, parser_model):
    with open(txt_path, "r", encoding="utf-8") as f:
        raw = clean_string(f.read().strip())
    triplet = parser_model.extract_triplet(raw)
    return {
        "raw": raw,
        "target": str(triplet.get("target", "")),
        "concepts": [str(x) for x in triplet.get("concepts", [])],
        "background": [str(x) for x in triplet.get("background", [])],
    }


def list_query_files(root, split):
    if split == "train":
        query_root = root / "OTB_query_train"
    elif split == "test":
        query_root = root / "OTB_query_test"
    elif split == "all":
        query_root = None
    else:
        raise ValueError(f"Unsupported split: {split}")

    if split == "all":
        txt_paths = []
        for subdir in ["OTB_query_train", "OTB_query_test"]:
            cur_root = root / subdir
            txt_paths.extend(sorted(cur_root.glob("*.txt")))
        return txt_paths

    return sorted(query_root.glob("*.txt"))


def main():
    parser = argparse.ArgumentParser(description="Precompute OTB99-Lang parsed text cache from OTB query files.")
    parser.add_argument("--otb-lang-root", required=True, help="Path to OTB_sentences root.")
    parser.add_argument("--split", default="test", choices=["train", "test", "all"],
                        help="Which OTB query split to parse.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument("--checkpoint", required=True, help="DPModel2 checkpoint path.")
    parser.add_argument("--cache_dir", required=True, help="HF datasets cache dir used by DPModel2Parser.")
    parser.add_argument("--device", default="cpu", help="Parser device, recommend cpu for offline preprocessing.")
    parser.add_argument("--embed_dim", type=int, default=300)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--lstm_layers", type=int, default=3)
    parser.add_argument("--ffnn_dim", type=int, default=256)
    parser.add_argument("--use_spacy", type=int, default=1)
    parser.add_argument("--spacy_model", default="en_core_web_sm")
    args = parser.parse_args()

    root = Path(args.otb_lang_root)
    txt_paths = list_query_files(root, args.split)
    if not txt_paths:
        raise FileNotFoundError(f"No query txt files found under {root} for split={args.split}")

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

    for idx, txt_path in enumerate(txt_paths, start=1):
        seq_name = txt_path.stem
        try:
            parsed = parse_language_file(txt_path, parser_model)
            if not parsed.get("raw", ""):
                print(f"[WARN] empty language file: {txt_path}")
                num_fail += 1
                continue
            result[seq_name] = parsed
            num_ok += 1
        except Exception as e:
            print(f"[ERROR] {seq_name}: {e}")
            num_fail += 1
        if idx % 20 == 0:
            print(f"[INFO] processed {idx}/{len(txt_paths)} sequences")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[DONE] split={args.split} saved {num_ok} sequence entries to {out_path}; failures={num_fail}")


if __name__ == "__main__":
    main()

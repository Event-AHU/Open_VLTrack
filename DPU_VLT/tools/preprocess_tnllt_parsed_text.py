
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
    base = Path(__file__).resolve().parents[1] / 'lib' / 'train' / 'data_specs'
    mapping = {
        'train': base / 'tnl_lt_train_split.txt',
        'val': base / 'tnl_lt_val_split.txt',
        'test': base / 'tnl_lt_test_split.txt',
    }
    if split == 'all':
        names = []
        for key in ('train', 'val', 'test'):
            names.extend(load_sequence_list(key))
        return sorted(set(names))
    file_path = mapping[split]
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(description='Precompute TNLLT parsed text cache.')
    parser.add_argument('--tnllt_root', required=True, help='Path to TNLLT dataset root')
    parser.add_argument('--output', required=True, help='Output JSON path')
    parser.add_argument('--split', default='train', choices=['train', 'val', 'test', 'all'])
    parser.add_argument('--checkpoint', required=True, help='DPModel2 checkpoint path')
    parser.add_argument('--cache_dir', required=True, help='HF datasets cache dir used by DPModel2Parser')
    parser.add_argument('--device', default='cpu', help='Parser device, recommend cpu for offline preprocessing')
    parser.add_argument('--embed_dim', type=int, default=300)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--lstm_layers', type=int, default=3)
    parser.add_argument('--ffnn_dim', type=int, default=256)
    parser.add_argument('--use_spacy', type=int, default=1)
    parser.add_argument('--spacy_model', default='en_core_web_sm')
    args = parser.parse_args()

    seq_names = load_sequence_list(args.split)
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

    root = Path(args.tnllt_root)
    result = {}
    num_ok = 0
    num_fail = 0
    for idx, seq_name in enumerate(seq_names, start=1):
        txt_path = root / seq_name / 'language.txt'
        if not txt_path.is_file():
            print(f'[WARN] missing language file: {txt_path}')
            num_fail += 1
            continue
        raw = txt_path.read_text(encoding='utf-8').strip()
        try:
            triplet = parser_model.extract_triplet(raw)
            result[seq_name] = {
                'raw': raw,
                'target': str(triplet.get('target', '')),
                'concepts': [str(x) for x in triplet.get('concepts', [])],
                'background': [str(x) for x in triplet.get('background', [])],
            }
            num_ok += 1
        except Exception as e:
            print(f'[ERROR] {seq_name}: {e}')
            num_fail += 1
        if idx % 100 == 0:
            print(f'[INFO] processed {idx}/{len(seq_names)} sequences')

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f'[DONE] saved {num_ok} entries to {out_path}; failures={num_fail}')


if __name__ == '__main__':
    main()

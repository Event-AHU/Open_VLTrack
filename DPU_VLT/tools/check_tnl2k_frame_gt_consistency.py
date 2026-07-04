#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check per-sequence consistency between image count and groundtruth rows in TNL2K."
    )
    parser.add_argument("--tnl2k-root", required=True, help="Path to TNL2K root, e.g. /rydata/dataset/SOT/TNL2k")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--output-json", default="", help="Optional output json path")
    parser.add_argument("--output-csv", default="", help="Optional output csv path for mismatches")
    return parser.parse_args()


def load_sequence_list(split_root: Path):
    sequence_list = []
    subset_list = [p for p in split_root.iterdir() if p.is_dir() and p.name != "revised_annotations"]
    if len(subset_list) > 14:
        return sorted([p.name for p in subset_list])
    for subset in sorted(subset_list):
        for seq in sorted([x for x in subset.iterdir() if x.is_dir()]):
            sequence_list.append(f"{subset.name}/{seq.name}")
    return sequence_list


def count_gt_rows(gt_path: Path):
    if not gt_path.is_file():
        return -1
    n = 0
    with open(gt_path, "r", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row:
                continue
            # guard malformed blank rows like ",,,"
            if all(str(x).strip() == "" for x in row):
                continue
            n += 1
    return n


def count_imgs(imgs_dir: Path):
    if not imgs_dir.is_dir():
        return -1
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    n = 0
    for p in imgs_dir.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            n += 1
    return n


def main():
    args = parse_args()
    split_root = Path(args.tnl2k_root) / args.split
    if not split_root.is_dir():
        raise SystemExit(f"Split path not found: {split_root}")

    seq_names = load_sequence_list(split_root)
    total = 0
    mismatch = 0
    missing = 0
    rows = []

    for seq_name in seq_names:
        seq_path = split_root / seq_name
        gt_path = seq_path / "groundtruth.txt"
        imgs_dir = seq_path / "imgs"
        gt_n = count_gt_rows(gt_path)
        img_n = count_imgs(imgs_dir)
        total += 1
        status = "ok"
        if gt_n < 0 or img_n < 0:
            status = "missing"
            missing += 1
        elif gt_n != img_n:
            status = "mismatch"
            mismatch += 1
        rows.append(
            {
                "seq_name": seq_name,
                "gt_rows": gt_n,
                "img_count": img_n,
                "delta_gt_minus_img": (gt_n - img_n) if (gt_n >= 0 and img_n >= 0) else None,
                "status": status,
            }
        )

    print(f"[INFO] split={args.split} total={total} mismatch={mismatch} missing={missing}")
    if mismatch > 0:
        print("[INFO] mismatches:")
        for r in rows:
            if r["status"] == "mismatch":
                print(
                    f"  {r['seq_name']}: gt_rows={r['gt_rows']} img_count={r['img_count']} "
                    f"delta={r['delta_gt_minus_img']}"
                )

    if args.output_json:
        out_json = Path(args.output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "meta": {
                        "tnl2k_root": args.tnl2k_root,
                        "split": args.split,
                        "total": total,
                        "mismatch": mismatch,
                        "missing": missing,
                    },
                    "rows": rows,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"[DONE] wrote json: {out_json}")

    if args.output_csv:
        out_csv = Path(args.output_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["seq_name", "gt_rows", "img_count", "delta_gt_minus_img", "status"]
            )
            writer.writeheader()
            for r in rows:
                if r["status"] != "ok":
                    writer.writerow(r)
        print(f"[DONE] wrote csv: {out_csv}")


if __name__ == "__main__":
    main()

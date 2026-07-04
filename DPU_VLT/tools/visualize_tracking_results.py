#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np


OURS_COLOR = (0, 0, 255)
GT_COLOR = (0, 180, 0)
FRAME_TEXT_COLOR = (0, 180, 0)
OTHER_METHOD_COLORS = [
    (255, 0, 0),
    (0, 165, 255),
    (255, 0, 255),
    (255, 255, 0),
    (128, 0, 255),
    (0, 255, 255),
    (180, 120, 0),
    (80, 80, 255),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Draw tracking-result boxes from multiple methods on sequence frames."
    )
    parser.add_argument(
        "--result-root",
        required=True,
        help="Root like tracking_vis/tnllt, containing method subdirectories.",
    )
    parser.add_argument(
        "--dataset-root",
        "--tnllt-root",
        dest="dataset_root",
        default="/rydata/dataset/SOT/TNLLT",
        help="Dataset root containing sequence/imgs folders.",
    )
    parser.add_argument("--output-dir", required=True, help="Output visualization directory.")
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Optional method names. Defaults to all subdirectories under result-root.",
    )
    parser.add_argument(
        "--seq",
        default="",
        help="Optional single sequence name. If omitted, visualize all sequences found in results.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Resize output image by this factor after drawing. Use 0.5 to save smaller images.",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=0,
        help="Resize output so the longer side is at most this value. 0 disables.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Visualize every N-th frame.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=1,
        help="1-based first frame to visualize.",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=0,
        help="1-based last frame to visualize. 0 means until sequence end.",
    )
    parser.add_argument(
        "--no-draw-gt",
        action="store_true",
        help="Disable drawing ground-truth boxes.",
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=2,
        help="Rectangle line width.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip frames whose output image already exists.",
    )
    parser.add_argument(
        "--ext",
        default=".jpg",
        choices=[".jpg", ".png"],
        help="Output image extension.",
    )
    parser.add_argument(
        "--jpg-quality",
        type=int,
        default=90,
        help="JPEG quality when --ext .jpg.",
    )
    parser.add_argument(
        "--ours-name",
        default="ours",
        help="Method name that should always use the red box color.",
    )
    return parser.parse_args()


def load_boxes(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p for p in re.split(r"[\s,;\t]+", line) if p]
            if len(parts) < 4:
                continue
            try:
                rows.append([float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])])
            except ValueError:
                continue
    return np.asarray(rows, dtype=np.float32)


def discover_methods(result_root, method_names):
    root = Path(result_root)
    if method_names:
        methods = [(name, root / name) for name in method_names]
    else:
        methods = [(p.name, p) for p in sorted(root.iterdir()) if p.is_dir()]
    methods = [(name, path) for name, path in methods if path.is_dir()]
    if not methods:
        raise FileNotFoundError(f"No method directories found under {root}")
    return methods


def pick_method_color(method_name, ours_name, used_colors):
    if method_name == ours_name:
        return OURS_COLOR
    for color in OTHER_METHOD_COLORS:
        if color not in used_colors and color != OURS_COLOR and color != GT_COLOR:
            return color
    return OTHER_METHOD_COLORS[len(used_colors) % len(OTHER_METHOD_COLORS)]


def result_file_for(method_dir, seq_name):
    candidates = [
        method_dir / seq_name / f"{seq_name}.txt",
        method_dir / seq_name / "pred.txt",
        method_dir / seq_name / "bbox.txt",
        method_dir / seq_name / "result.txt",
        method_dir / f"{seq_name}.txt",
        method_dir / "tnllt" / f"{seq_name}.txt",
    ]
    for path in candidates:
        if path.is_file() and not path.name.endswith("_time.txt"):
            return path
    seq_dir = method_dir / seq_name
    if seq_dir.is_dir():
        txts = sorted(
            p for p in seq_dir.rglob("*.txt")
            if p.is_file() and not p.name.endswith("_time.txt")
        )
        if txts:
            return txts[0]
    return None


def discover_sequences(methods, explicit_seq):
    if explicit_seq:
        return [explicit_seq]
    seqs = set()
    for _, method_dir in methods:
        search_dirs = [method_dir, method_dir / "tnllt"]
        for cur in search_dirs:
            if not cur.is_dir():
                continue
            for path in cur.glob("*.txt"):
                if path.name.endswith("_time.txt"):
                    continue
                seqs.add(path.stem)
        for seq_dir in method_dir.iterdir():
            if not seq_dir.is_dir():
                continue
            txts = [p for p in seq_dir.rglob("*.txt") if not p.name.endswith("_time.txt")]
            if txts:
                seqs.add(seq_dir.name)
    return sorted(seqs)


def list_frames(seq_dir):
    img_dir = seq_dir / "imgs"
    if not img_dir.is_dir():
        return []
    frames = [p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    return sorted(frames)


def draw_frame_index(image, frame_idx, color):
    text = f"#{frame_idx + 1:04d}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(1.2, min(image.shape[0], image.shape[1]) / 900.0)
    thickness = max(3, int(round(scale * 3)))
    x = max(10, int(round(16 * scale)))
    y = max(26, int(round(32 * scale)))
    cv2.putText(image, text, (x, y), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_box(image, box, color, line_width):
    x, y, w, h = [float(v) for v in box[:4]]
    if not np.isfinite([x, y, w, h]).all() or w <= 0 or h <= 0:
        return
    x1, y1 = int(round(x)), int(round(y))
    x2, y2 = int(round(x + w)), int(round(y + h))
    h_img, w_img = image.shape[:2]
    x1 = max(0, min(w_img - 1, x1))
    y1 = max(0, min(h_img - 1, y1))
    x2 = max(0, min(w_img - 1, x2))
    y2 = max(0, min(h_img - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return
    cv2.rectangle(image, (x1, y1), (x2, y2), color, line_width)


def resize_output(image, scale, max_side):
    if scale > 0 and scale != 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if max_side and max(image.shape[:2]) > max_side:
        ratio = float(max_side) / float(max(image.shape[:2]))
        image = cv2.resize(image, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
    return image


def frame_indices(num_frames, start_frame, end_frame, stride):
    start = max(1, int(start_frame))
    end = int(end_frame) if end_frame and end_frame > 0 else num_frames
    end = min(end, num_frames)
    stride = max(1, int(stride))
    return range(start - 1, end, stride)


def write_image(path, image, jpg_quality):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        return cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)])
    return cv2.imwrite(str(path), image)


def bgr_to_rgb_hex(color):
    b, g, r = [int(v) for v in color]
    return "#{:02X}{:02X}{:02X}".format(r, g, b)


def build_color_legend(methods, ours_name, draw_gt=False):
    legend = {}
    used_colors = set()
    for method_name, _ in methods:
        color = pick_method_color(method_name, ours_name, used_colors)
        used_colors.add(color)
        legend[method_name] = {
            "bgr": [int(v) for v in color],
            "rgb_hex": bgr_to_rgb_hex(color),
        }
    if draw_gt:
        legend["GT"] = {
            "bgr": [int(v) for v in GT_COLOR],
            "rgb_hex": bgr_to_rgb_hex(GT_COLOR),
        }
    legend["frame_index"] = {
        "bgr": [int(v) for v in FRAME_TEXT_COLOR],
        "rgb_hex": bgr_to_rgb_hex(FRAME_TEXT_COLOR),
    }
    return legend


def visualize_sequence(args, seq_name, methods, summary):
    seq_dir = Path(args.dataset_root) / seq_name
    if not seq_dir.is_dir():
        print(f"[WARN] missing sequence: {seq_dir}")
        return
    frames = list_frames(seq_dir)
    if not frames:
        print(f"[WARN] no frames found: {seq_dir / 'imgs'}")
        return

    boxes_by_method = {}
    used_colors = set()
    missing_methods = []
    for method_name, method_dir in methods:
        result_path = result_file_for(method_dir, seq_name)
        if result_path is None:
            missing_methods.append(method_name)
            continue
        color = pick_method_color(method_name, args.ours_name, used_colors)
        used_colors.add(color)
        boxes_by_method[method_name] = {
            "boxes": load_boxes(result_path),
            "color": color,
            "path": str(result_path),
        }

    draw_gt = not args.no_draw_gt
    if missing_methods:
        print(f"[WARN] {seq_name}: missing result files for methods: {', '.join(missing_methods)}")
    if not boxes_by_method and not draw_gt:
        print(f"[WARN] no result boxes found for {seq_name}")
        return
    if not boxes_by_method and draw_gt:
        print(f"[WARN] {seq_name}: no method boxes found; output will contain GT/frame index only.")

    gt_boxes = None
    if draw_gt:
        gt_path = seq_dir / "groundtruth.txt"
        if gt_path.is_file():
            gt_boxes = load_boxes(gt_path)

    out_seq_dir = Path(args.output_dir) / seq_name
    count = 0
    for frame_idx in frame_indices(len(frames), args.start_frame, args.end_frame, args.frame_stride):
        out_path = out_seq_dir / f"{frame_idx + 1:05d}{args.ext}"
        if args.skip_existing and out_path.is_file():
            continue
        image = cv2.imread(str(frames[frame_idx]), cv2.IMREAD_COLOR)
        if image is None:
            print(f"[WARN] failed to read frame: {frames[frame_idx]}")
            continue

        if gt_boxes is not None and frame_idx < len(gt_boxes):
            draw_box(image, gt_boxes[frame_idx], GT_COLOR, max(1, args.line_width))

        ordered_method_names = [name for name in boxes_by_method.keys() if name != args.ours_name]
        if args.ours_name in boxes_by_method:
            ordered_method_names.append(args.ours_name)

        for method_name in ordered_method_names:
            item = boxes_by_method[method_name]
            boxes = item["boxes"]
            if frame_idx < len(boxes):
                draw_box(image, boxes[frame_idx], item["color"], args.line_width)

        draw_frame_index(image, frame_idx, FRAME_TEXT_COLOR)

        image = resize_output(image, args.scale, args.max_side)
        if not write_image(out_path, image, args.jpg_quality):
            print(f"[WARN] failed to write: {out_path}")
            continue
        count += 1

    summary.append({
        "seq_name": seq_name,
        "num_frames": len(frames),
        "written_frames": count,
        "methods": {k: v["path"] for k, v in boxes_by_method.items()},
    })
    print(f"[DONE] {seq_name}: wrote {count} frames to {out_seq_dir}")


def main():
    args = parse_args()
    methods = discover_methods(args.result_root, args.methods)
    seqs = discover_sequences(methods, args.seq)
    if not seqs:
        raise RuntimeError("No sequences found to visualize.")

    print("[INFO] methods:", ", ".join(name for name, _ in methods))
    print(f"[INFO] sequences: {len(seqs)}")
    color_legend = build_color_legend(methods, ours_name=args.ours_name, draw_gt=not args.no_draw_gt)
    print("[INFO] colors:", ", ".join(f"{name}={item['rgb_hex']}" for name, item in color_legend.items()))
    summary = []
    for seq_name in seqs:
        visualize_sequence(args, seq_name, methods, summary)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    summary_payload = {
        "result_root": str(Path(args.result_root)),
        "dataset_root": str(Path(args.dataset_root)),
        "output_dir": str(Path(args.output_dir)),
        "ours_name": args.ours_name,
        "color_legend": color_legend,
        "sequences": summary,
    }
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] wrote summary: {summary_path}")


if __name__ == "__main__":
    main()

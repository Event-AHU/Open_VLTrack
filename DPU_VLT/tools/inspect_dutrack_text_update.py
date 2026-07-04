#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import BertTokenizer, BlipForConditionalGeneration, BlipProcessor


DEFAULT_BLIP_DIR = "/rydata/jinliye/RL/vltracking/LongTimeTracking/Trackingbaseline/DUTrack/pretrained/blip-image-captioning-base"
DEFAULT_BERT_DIR = "/rydata/jinliye/RL/vltracking/LongTimeTracking/Trackingbaseline/DUTrack/pretrained/bert"
DEFAULT_TNLLT_ROOT = "/rydata/dataset/SOT/TNLLT"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect DUTrack-main's BLIP based dynamic text update for images/frames."
    )
    parser.add_argument("--image", default="", help="Single image path.")
    parser.add_argument("--text", default="", help="Text prompt passed as BLIP conditional text.")
    parser.add_argument(
        "--prompts",
        nargs="*",
        default=None,
        help="Run multiple BLIP conditional prompts for every image.",
    )
    parser.add_argument("--seq", default="", help="TNLLT sequence name, e.g. JE_Horse_03.")
    parser.add_argument(
        "--frames",
        nargs="*",
        type=int,
        default=[],
        help="1-based frame ids under --tnllt-root/--seq/imgs, e.g. 50 150 250.",
    )
    parser.add_argument("--tnllt-root", default=DEFAULT_TNLLT_ROOT)
    parser.add_argument("--blip-dir", default=DEFAULT_BLIP_DIR)
    parser.add_argument("--bert-dir", default=DEFAULT_BERT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--num-beams", type=int, default=3)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--crop-gt",
        action="store_true",
        help="For --seq frames, caption the GT target crop instead of the full frame.",
    )
    parser.add_argument(
        "--crop-factor",
        type=float,
        default=2.0,
        help="GT crop expansion factor when --crop-gt is enabled.",
    )
    parser.add_argument(
        "--use-seq-name-as-class",
        action="store_true",
        help="Use --seq as the BLIP conditional text, matching DUTrack-main TNLLT object_class behavior.",
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Run BLIP without conditional text.",
    )
    parser.add_argument("--output-json", default="", help="Optional path to save results.")
    return parser.parse_args()


def load_image_rgb(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_gt_box(tnllt_root, seq, frame_id):
    gt_path = Path(tnllt_root) / seq / "groundtruth.txt"
    rows = []
    for line in gt_path.read_text(encoding="utf-8").splitlines():
        parts = [p for p in line.replace(",", " ").split() if p]
        if len(parts) >= 4:
            rows.append([float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])])
    if frame_id < 1 or frame_id > len(rows):
        raise IndexError(f"Frame {frame_id} outside GT length {len(rows)} for {seq}")
    return rows[frame_id - 1]


def crop_xywh(image_rgb, box, factor):
    x, y, w, h = [float(v) for v in box[:4]]
    cx, cy = x + w / 2.0, y + h / 2.0
    side_w, side_h = w * factor, h * factor
    x1 = int(round(cx - side_w / 2.0))
    y1 = int(round(cy - side_h / 2.0))
    x2 = int(round(cx + side_w / 2.0))
    y2 = int(round(cy + side_h / 2.0))
    ih, iw = image_rgb.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(iw, x2), min(ih, y2)
    if x2 <= x1 or y2 <= y1:
        return image_rgb
    return image_rgb[y1:y2, x1:x2]


def resolve_items(args):
    items = []
    if args.image:
        items.append(("image", Path(args.image), None))
    if args.seq and args.frames:
        img_dir = Path(args.tnllt_root) / args.seq / "imgs"
        for frame_id in args.frames:
            path = img_dir / f"{frame_id:05d}.png"
            if not path.is_file():
                jpg_path = img_dir / f"{frame_id:05d}.jpg"
                path = jpg_path if jpg_path.is_file() else path
            items.append((args.seq, path, frame_id))
    if not items:
        raise SystemExit("Provide either --image or --seq with --frames.")
    return items


def load_refiner(blip_dir, bert_dir, device):
    if device.startswith("cuda"):
        torch.cuda.set_device(torch.device(device).index or 0)
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    processor = BlipProcessor.from_pretrained(blip_dir)
    model = BlipForConditionalGeneration.from_pretrained(blip_dir, torch_dtype=dtype).to(device).eval()
    tokenizer = BertTokenizer.from_pretrained(bert_dir)
    return processor, model, tokenizer


def generate_caption(processor, model, image_rgb, prompt, args):
    if prompt:
        inputs = processor(image_rgb, prompt, return_tensors="pt")
    else:
        inputs = processor(image_rgb, return_tensors="pt")
    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    inputs = inputs.to(args.device, dtype)
    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "max_length": None,
        "num_beams": args.num_beams,
    }
    if args.do_sample:
        gen_kwargs.update({
            "do_sample": True,
            "top_p": args.top_p,
            "temperature": args.temperature,
            "num_beams": 1,
        })
    out = model.generate(**inputs, **gen_kwargs)
    return processor.decode(out[0], skip_special_tokens=True)


def resolve_prompts(args):
    if args.no_prompt:
        return [None]
    if args.use_seq_name_as_class and args.seq:
        return [args.seq]
    if args.prompts is not None and len(args.prompts) > 0:
        return args.prompts
    return [args.text or None]


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false.")

    processor, model, _ = load_refiner(args.blip_dir, args.bert_dir, args.device)
    items = resolve_items(args)
    prompts = resolve_prompts(args)

    results = []
    for seq_name, image_path, frame_id in items:
        image_rgb = load_image_rgb(image_path)
        input_kind = "full_frame"
        if args.crop_gt and args.seq and frame_id is not None:
            gt_box = load_gt_box(args.tnllt_root, args.seq, frame_id)
            image_rgb = crop_xywh(image_rgb, gt_box, args.crop_factor)
            input_kind = f"gt_crop_x{args.crop_factor:g}"
        frame_label = f"#{frame_id:05d}" if frame_id is not None else image_path.name
        print(f"{seq_name} {frame_label}")
        print(f"  image: {image_path}")
        print(f"  input: {input_kind}")
        for prompt in prompts:
            with torch.no_grad():
                description = generate_caption(processor, model, image_rgb, prompt, args)
            item = {
                "seq": seq_name,
                "frame": frame_id,
                "image": str(image_path),
                "input": input_kind,
                "prompt": prompt or "",
                "dutrack_description": str(description),
            }
            results.append(item)
            print(f"  prompt: {prompt or '<none>'}")
            print(f"  DUTrack updated text: {description}")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[DONE] wrote {out_path}")


if __name__ == "__main__":
    main()

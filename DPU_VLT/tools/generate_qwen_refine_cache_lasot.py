#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.train.data.processing_utils import sample_target


SYSTEM_PROMPT = """You are refining structured tracking text for the current search region in LaSOT.

Target is fixed and should not be changed.

You may only update:
- concept
- background

Preferred behavior:
- keep target unchanged
- produce a short concept when a reliable visual attribute, part, pose, state, color, or material cue is visible
- produce a short background only when there is a clear local disambiguating context
- keep phrases short, concrete, and visual

Forbidden:
- changing target
- rewriting the whole sentence
- inventing hidden or uncertain details
- returning explanations

Return JSON only with keys: concept, background."""


def parse_args():
    parser = argparse.ArgumentParser(description="Generate offline Qwen refine cache for LaSOT.")
    parser.add_argument("--lasot-root", required=True, help="Path to LaSOTBenchmark root")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--lora-path", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--cache-root", default=str(REPO_ROOT / "output/test/qwencache"))
    parser.add_argument("--dataset-name", default="LaSOT")
    parser.add_argument("--exp-name", default="")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--parsed-text-path", default="")
    parser.add_argument("--search-factor", type=float, default=4.0)
    parser.add_argument("--search-size", type=int, default=256)
    parser.add_argument("--cache-interval", type=int, default=50)
    parser.add_argument("--anchor-offset", type=int, default=0)
    parser.add_argument("--max-seqs", type=int, default=0)
    parser.add_argument("--max-anchors-per-seq", type=int, default=0)
    parser.add_argument("--seq-start", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--visible-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit-total", type=int, default=0)
    parser.add_argument("--max-field-words", type=int, default=6)
    parser.add_argument("--max-growth", type=float, default=2.0)
    args = parser.parse_args()
    if not args.output and not args.exp_name:
        parser.error("Either --output or --exp-name must be provided.")
    return args


def resolve_output_path(args):
    if args.output:
        return Path(args.output)
    filename = f"lasot_qwen_refine_{args.split}_full.json"
    return Path(args.cache_root) / args.split / args.dataset_name / args.exp_name / filename


def load_qwen(model_path, device, lora_path=""):
    from transformers import AutoProcessor

    model_cls = None
    try:
        from transformers import AutoModelForImageTextToText  # type: ignore
        model_cls = AutoModelForImageTextToText
    except Exception:
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration  # type: ignore
            model_cls = Qwen2_5_VLForConditionalGeneration
        except Exception:
            from transformers import AutoModelForVision2Seq  # type: ignore
            model_cls = AutoModelForVision2Seq

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = model_cls.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
    if lora_path:
        from peft import PeftModel
        print(f"[INFO] Loading LoRA adapter from {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
    model = model.to(device)
    model.eval()
    return processor, model


def load_sequence_list(split):
    if split == "train":
        split_file = REPO_ROOT / "lib" / "train" / "data_specs" / "lasot_train_split.txt"
        with open(split_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    from lib.test.evaluation.lasotdataset import LaSOTDataset

    return list(LaSOTDataset().sequence_list)


def load_parsed_cache(path):
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def nearest_parsed(parsed_cache, seq_name, frame_id):
    item = parsed_cache.get(seq_name, None)
    if not isinstance(item, dict):
        return None
    if "target" in item:
        return normalize_parsed(item)

    frame_keys = []
    for key in item.keys():
        try:
            frame_keys.append(int(key))
        except Exception:
            continue
    if not frame_keys:
        return None
    frame_keys = sorted(frame_keys)
    import bisect

    idx = bisect.bisect_left(frame_keys, int(frame_id))
    if idx == 0:
        target_frame = frame_keys[0]
    elif idx == len(frame_keys):
        target_frame = frame_keys[-1]
    else:
        target_frame = frame_keys[idx - 1]
    parsed = item.get(str(target_frame), None)
    return normalize_parsed(parsed) if isinstance(parsed, dict) else None


def normalize_parsed(parsed):
    def _join(value):
        if isinstance(value, list):
            return " ".join(str(x) for x in value)
        return str(value)

    return {
        "raw": str(parsed.get("raw", "")),
        "target": str(parsed.get("target", "")),
        "concepts": _join(parsed.get("concepts", [])),
        "background": _join(parsed.get("background", [])),
    }


def read_gt(seq_path):
    boxes = []
    with open(Path(seq_path) / "groundtruth.txt", "r", encoding="utf-8") as f:
        for row in csv.reader(f):
            if row:
                boxes.append([float(x) for x in row[:4]])
    return np.asarray(boxes, dtype=np.float32)


def read_visible(seq_path):
    def _read_first_row(name):
        with open(Path(seq_path) / name, "r", encoding="utf-8") as f:
            return np.asarray([int(x) for x in next(csv.reader(f))], dtype=np.uint8)

    full_occ = _read_first_row("full_occlusion.txt")
    out_of_view = _read_first_row("out_of_view.txt")
    return (full_occ == 0) & (out_of_view == 0)


def read_raw_nlp(seq_path):
    path = Path(seq_path) / "nlp.txt"
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8").strip()


def load_image_rgb(path):
    import cv2

    bgr = cv2.imread(str(path))
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def extract_json_object(text):
    if not text:
        raise ValueError("Empty generation output.")
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            candidate = text[start:end + 1]
    if candidate is None:
        raise ValueError(f"Failed to find JSON object in: {text[:200]}")
    return json.loads(candidate)


def build_prompt(parsed):
    return "\n".join([
        f"Target: {parsed.get('target', '')}",
        f"Concept: {parsed.get('concepts', '')}",
        f"Background: {parsed.get('background', '')}",
        "",
        "Task:",
        "Update concept and background for the current search image while keeping target fixed.",
        "Return JSON only as:",
        '{"concept":"...", "background":"..."}',
    ])


def validate_output(raw_output, concept, background, max_field_words, max_growth):
    out_concept = str(raw_output.get("concept", raw_output.get("concepts", ""))).strip()
    out_background = str(raw_output.get("background", "")).strip()

    def _check_field(name, original, updated):
        if not updated:
            return ""
        words = updated.split()
        if len(words) > max_field_words:
            raise ValueError(f"{name} too long: {updated!r}")
        orig_words = max(1, len(original.split())) if original else 1
        if len(words) > math.ceil(orig_words * max_growth):
            raise ValueError(f"{name} grows too much: {original!r} -> {updated!r}")
        return updated

    return {
        "concept": _check_field("concept", concept, out_concept),
        "background": _check_field("background", background, out_background),
    }


def run_qwen(processor, model, image_rgb, user_prompt, device, max_new_tokens):
    image_token = getattr(processor, "image_token", "<|image_pad|>")
    prompt_text = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n<|vision_start|>{image_token}<|vision_end|>\n{user_prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    inputs = processor(text=[prompt_text], images=[image_rgb], padding=True, return_tensors="pt")
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    prompt_len = inputs["input_ids"].shape[1]
    trimmed = generated[:, prompt_len:]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def get_anchor_ids(num_frames, interval, offset):
    if interval <= 0:
        raise ValueError("cache interval must be > 0")
    if offset < 0:
        offset = 0
    return list(range(offset, num_frames, interval))


def main():
    args = parse_args()
    output_path = resolve_output_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and args.overwrite:
        raise SystemExit("--resume and --overwrite cannot be used together.")

    cache = None
    if output_path.exists():
        if args.resume:
            with open(output_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
            if "entries" not in cache or not isinstance(cache["entries"], dict):
                raise SystemExit(f"Invalid cache format in {output_path}")
            cache.setdefault("meta", {})
        elif not args.overwrite:
            raise SystemExit(f"Output already exists: {output_path}. Use --overwrite or --resume.")

    parsed_cache = load_parsed_cache(args.parsed_text_path)
    processor, model = load_qwen(args.model_path, args.device, args.lora_path)

    seq_names = load_sequence_list(args.split)[args.seq_start:]
    if args.max_seqs > 0:
        seq_names = seq_names[:args.max_seqs]

    if cache is None:
        cache = {
            "meta": {
                "model_path": args.model_path,
                "lora_path": args.lora_path,
                "lasot_root": args.lasot_root,
                "dataset_name": args.dataset_name,
                "split": args.split,
                "parsed_text_path": args.parsed_text_path,
                "search_factor": args.search_factor,
                "search_size": args.search_size,
                "cache_interval": args.cache_interval,
                "anchor_offset": args.anchor_offset,
            },
            "entries": {},
        }

    def flush_cache(reason):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        print(f"[INFO] flush ({reason}) -> total entries: {len(cache['entries'])}")

    written = 0
    skipped_existing = 0
    root = Path(args.lasot_root)
    for seq_name in seq_names:
        class_name = seq_name.split("-")[0]
        seq_path = root / class_name / seq_name
        boxes = read_gt(seq_path)
        visible = read_visible(seq_path)
        frame_dir = seq_path / "img"
        frame_files = sorted(frame_dir.glob("*.jpg"))
        raw_nlp = read_raw_nlp(seq_path)
        anchor_ids = get_anchor_ids(len(frame_files), args.cache_interval, args.anchor_offset)
        if args.max_anchors_per_seq > 0:
            anchor_ids = anchor_ids[:args.max_anchors_per_seq]

        seq_new = 0
        for frame_id in anchor_ids:
            if frame_id >= len(boxes):
                continue
            is_visible = bool(visible[frame_id]) if frame_id < len(visible) else True
            if args.visible_only and not is_visible:
                continue
            key = f"{seq_name}:{frame_id}"
            if args.resume and key in cache["entries"]:
                skipped_existing += 1
                continue

            parsed = nearest_parsed(parsed_cache, seq_name, frame_id)
            if parsed is None:
                parsed = {"raw": raw_nlp, "target": class_name, "concepts": "", "background": ""}
            if not parsed.get("target"):
                parsed["target"] = class_name

            try:
                image_rgb_full = load_image_rgb(frame_files[frame_id])
                image_rgb, _, _ = sample_target(image_rgb_full, boxes[frame_id], args.search_factor, output_sz=args.search_size)
                raw_text = run_qwen(processor, model, image_rgb, build_prompt(parsed), args.device, args.max_new_tokens)
                parsed_output = extract_json_object(raw_text)
                validated = validate_output(parsed_output, parsed.get("concepts", ""), parsed.get("background", ""), args.max_field_words, args.max_growth)
            except Exception as exc:
                print(f"[WARN] {key} failed: {exc}")
                continue

            cache["entries"][key] = {
                "seq_name": seq_name,
                "frame_id": frame_id,
                "image_path": str(frame_files[frame_id]),
                "raw": parsed.get("raw", ""),
                "target": parsed.get("target", ""),
                "concepts": parsed.get("concepts", ""),
                "background": parsed.get("background", ""),
                "qwen_target": parsed.get("target", ""),
                "qwen_concept": validated["concept"],
                "qwen_background": validated["background"],
                "visible": is_visible,
                "raw_response": raw_text,
            }
            written += 1
            seq_new += 1
            print(f"[INFO] cached {key} -> target={parsed.get('target')!r} concept={validated['concept']!r} background={validated['background']!r}")
            if args.limit_total > 0 and written >= args.limit_total:
                break

        flush_cache(f"seq={seq_name}, new={seq_new}")
        if args.limit_total > 0 and written >= args.limit_total:
            break

    print(f"[INFO] run finished: new={written}, skipped_existing={skipped_existing}, total_entries={len(cache['entries'])}, output={output_path}")


if __name__ == "__main__":
    main()

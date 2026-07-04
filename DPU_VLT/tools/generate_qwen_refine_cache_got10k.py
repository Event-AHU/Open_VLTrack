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


SYSTEM_PROMPTS = {
    "v2": """You are refining structured tracking text for the current search region.

Target is fixed and should not be changed.

You may only update:
- concept
- background

Allowed operations on concept/background:
- keep
- shorten
- replace with a visually supported short phrase
- drop by returning an empty string

Forbidden:
- changing target
- rewriting the whole sentence
- inventing unseen entities
- returning explanations

Return JSON only with keys: concept, background.""",
    "got10k_v1": """You are refining structured tracking text for the current search region in GOT-10K.

Target is fixed and should not be changed.

The current input may have an empty concept and empty background.
If the search image shows a visually clear object attribute, part, pose, state, color, material, or local context,
you should fill concept and/or background with a short visually grounded phrase.

You may only update:
- concept
- background

Preferred behavior:
- keep target unchanged
- produce a short concept when a reliable visual attribute or object-part cue is visible
- produce a short background only when there is a clear local disambiguating context
- keep phrases short, concrete, and visual
- return an empty string only when the field is truly unsupported

Allowed concept examples:
- red car
- white bird
- person head
- running person
- blue bottle

Allowed background examples:
- left side
- near fence
- on grass
- beside tree

Forbidden:
- changing target
- rewriting the whole sentence
- copying generic filler words
- inventing hidden or uncertain details
- returning explanations

Return JSON only with keys: concept, background.""",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate offline Qwen refine cache for GOT-10K val/test-time analysis.")
    parser.add_argument("--got10k-root", required=True, help="Path to GOT-10K split root, e.g. .../GOT-10k/full_data/val")
    parser.add_argument("--model-path", required=True, help="Path to a loadable Qwen2.5-VL snapshot directory.")
    parser.add_argument("--lora-path", default="", help="Optional PEFT/LoRA adapter path.")
    parser.add_argument("--output", default="", help="Output JSON cache path. If empty, build it from cache-root/split/dataset-name/exp-name.")
    parser.add_argument("--cache-root", default=str(REPO_ROOT / "output/test/qwencache"), help="Root for structured Qwen cache outputs.")
    parser.add_argument("--dataset-name", default="GOT10K", help="Dataset folder name under cache-root.")
    parser.add_argument("--exp-name", default="", help="Experiment folder name under cache-root/split/dataset-name.")
    parser.add_argument("--split", default="val", choices=["val"], help="Current offline cache generation requires GT boxes, so only val is supported now.")
    parser.add_argument("--search-factor", type=float, default=4.0)
    parser.add_argument("--search-size", type=int, default=256)
    parser.add_argument("--cache-interval", type=int, default=50, help="Generate Qwen cache every N frames.")
    parser.add_argument("--anchor-offset", type=int, default=0, help="Zero-based first anchor offset. Default 0 to ensure early frames also have cache support.")
    parser.add_argument("--max-seqs", type=int, default=0, help="Only process the first N sequences if > 0.")
    parser.add_argument("--max-anchors-per-seq", type=int, default=0, help="Only process the first N anchor frames per sequence if > 0.")
    parser.add_argument("--seq-start", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--system-prompt-version", default="v2", choices=sorted(SYSTEM_PROMPTS.keys()))
    parser.add_argument("--visible-only", action="store_true", help="Skip anchor frames where the target is absent.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing output cache and skip existing keys.")
    parser.add_argument("--limit-total", type=int, default=0, help="Stop after writing this many cache entries if > 0.")
    parser.add_argument("--max-field-words", type=int, default=6, help="Maximum allowed words for a refined concept/background field.")
    parser.add_argument("--max-growth", type=float, default=2.0, help="Maximum allowed word-count growth relative to the original field.")
    args = parser.parse_args()
    if not args.output and not args.exp_name:
        parser.error("Either --output or --exp-name must be provided.")
    return args


def resolve_output_path(args):
    if args.output:
        return Path(args.output)
    filename = f"got10k_qwen_refine_{args.split}_full.json"
    return Path(args.cache_root) / args.split / args.dataset_name / args.exp_name / filename


def load_qwen(model_path, device, lora_path=""):
    try:
        from transformers import AutoProcessor
    except Exception as exc:
        raise RuntimeError("transformers is required to run this script.") from exc

    model_cls = None
    try:
        from transformers import AutoModelForImageTextToText  # type: ignore
        model_cls = AutoModelForImageTextToText
    except Exception:
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration  # type: ignore
            model_cls = Qwen2_5_VLForConditionalGeneration
        except Exception:
            try:
                from transformers import AutoModelForVision2Seq  # type: ignore
                model_cls = AutoModelForVision2Seq
            except Exception as exc:
                raise RuntimeError("No suitable Qwen2.5-VL model class found in transformers.") from exc

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = model_cls.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
    if lora_path:
        try:
            from peft import PeftModel
        except Exception as exc:
            raise RuntimeError("peft is required when --lora-path is provided.") from exc
        print(f"[INFO] Loading LoRA adapter from {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
    model = model.to(device)
    model.eval()
    return processor, model


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


def load_image_rgb(path):
    try:
        import cv2

        bgr = cv2.imread(path)
        if bgr is not None:
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        pass

    try:
        from PIL import Image

        return np.asarray(Image.open(path).convert("RGB"))
    except Exception:
        pass

    try:
        import imageio.v3 as iio

        arr = iio.imread(path)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        return arr
    except Exception as exc:
        raise RuntimeError(f"Failed to read image: {path}. Install cv2, pillow, or imageio. Last error: {exc}")


def read_sequence_list(root):
    list_path = Path(root) / "list.txt"
    with open(list_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def read_meta_info(seq_path):
    meta_path = Path(seq_path) / "meta_info.ini"
    object_class = ""
    if not meta_path.is_file():
        return object_class
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if len(lines) > 5 and ": " in lines[5]:
            object_class = lines[5].split(": ", 1)[1].strip()
    except Exception:
        object_class = ""
    return object_class


def read_gt(seq_path):
    gt_path = Path(seq_path) / "groundtruth.txt"
    boxes = []
    with open(gt_path, "r", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row:
                continue
            boxes.append([float(x) for x in row[:4]])
    return np.asarray(boxes, dtype=np.float32)


def read_visible(seq_path):
    abs_path = Path(seq_path) / "absence.label"
    cover_path = Path(seq_path) / "cover.label"
    occlusion = []
    cover = []
    with open(abs_path, "r", encoding="utf-8") as f:
        for row in csv.reader(f):
            if row:
                occlusion.append(int(row[0]))
    with open(cover_path, "r", encoding="utf-8") as f:
        for row in csv.reader(f):
            if row:
                cover.append(int(row[0]))
    occlusion = np.asarray(occlusion, dtype=np.uint8)
    cover = np.asarray(cover, dtype=np.uint8)
    return (1 - occlusion) * (cover > 0)


def build_prompt(target, concept="", background=""):
    lines = [
        f"Target: {target}",
        f"Concept: {concept}",
        f"Background: {background}",
        "",
        "Task:",
        "Update concept and background for the current search image while keeping target fixed.",
        "Return JSON only as:",
        '{"concept":"...", "background":"..."}',
    ]
    return "\n".join(lines)


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


def run_qwen(processor, model, image_rgb, system_prompt, user_prompt, device, max_new_tokens):
    image_token = getattr(processor, "image_token", "<|image_pad|>")
    prompt_text = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n<|vision_start|>{image_token}<|vision_end|>\n{user_prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    inputs = processor(text=[prompt_text], images=[image_rgb], padding=True, return_tensors="pt")
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    prompt_len = inputs["input_ids"].shape[1]
    trimmed = generated[:, prompt_len:]
    decoded = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return decoded


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
            if not isinstance(cache, dict):
                raise SystemExit(f"Invalid cache format in {output_path}: root is not a dict.")
            if "entries" not in cache or not isinstance(cache["entries"], dict):
                raise SystemExit(f"Invalid cache format in {output_path}: missing dict field 'entries'.")
            if "meta" not in cache or not isinstance(cache["meta"], dict):
                cache["meta"] = {}
            print(f"[INFO] resume enabled: loaded {len(cache['entries'])} existing entries from {output_path}")
        elif not args.overwrite:
            raise SystemExit(f"Output already exists: {output_path}. Use --overwrite to replace it or --resume to continue.")

    root = Path(args.got10k_root)
    seq_names = read_sequence_list(root)
    seq_names = seq_names[args.seq_start:]
    if args.max_seqs > 0:
        seq_names = seq_names[:args.max_seqs]

    processor, model = load_qwen(args.model_path, args.device, args.lora_path)
    system_prompt = SYSTEM_PROMPTS[args.system_prompt_version]

    if cache is None:
        cache = {
            "meta": {
                "model_path": args.model_path,
                "lora_path": args.lora_path,
                "got10k_root": str(root),
                "dataset_name": args.dataset_name,
                "exp_name": args.exp_name,
                "split": args.split,
                "search_factor": args.search_factor,
                "search_size": args.search_size,
                "cache_interval": args.cache_interval,
                "anchor_offset": args.anchor_offset,
                "system_prompt_version": args.system_prompt_version,
                "visible_only": bool(args.visible_only),
            },
            "entries": {},
        }
    else:
        cache["meta"].update({
            "model_path": args.model_path,
            "lora_path": args.lora_path,
            "got10k_root": str(root),
            "dataset_name": args.dataset_name,
            "exp_name": args.exp_name,
            "split": args.split,
            "search_factor": args.search_factor,
            "search_size": args.search_size,
            "cache_interval": args.cache_interval,
            "anchor_offset": args.anchor_offset,
            "system_prompt_version": args.system_prompt_version,
            "visible_only": bool(args.visible_only),
        })

    def flush_cache(reason):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        print(f"[INFO] flush ({reason}) -> total entries: {len(cache['entries'])}")

    written = 0
    skipped_existing = 0
    for seq_name in seq_names:
        seq_path = root / seq_name
        boxes = read_gt(seq_path)
        visible = read_visible(seq_path)
        target = read_meta_info(seq_path)
        frame_files = sorted([x for x in os.listdir(seq_path) if x.endswith(".jpg")], key=lambda x: int(x[:-4]))
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

            image_path = seq_path / frame_files[frame_id]
            try:
                image_rgb_full = load_image_rgb(str(image_path))
                image_rgb, _, _ = sample_target(image_rgb_full, boxes[frame_id], args.search_factor, output_sz=args.search_size)
            except Exception as exc:
                print(f"[WARN] {key} sample_target failed: {exc}")
                continue

            user_prompt = build_prompt(target=target, concept="", background="")
            try:
                raw_text = run_qwen(processor, model, image_rgb, system_prompt, user_prompt, args.device, args.max_new_tokens)
                parsed_output = extract_json_object(raw_text)
                validated = validate_output(parsed_output, "", "", args.max_field_words, args.max_growth)
            except Exception as exc:
                print(f"[WARN] {key} failed: {exc}")
                continue

            cache["entries"][key] = {
                "seq_name": seq_name,
                "frame_id": frame_id,
                "image_path": str(image_path),
                "target": target,
                "concepts": "",
                "background": "",
                "qwen_target": target,
                "qwen_concept": validated["concept"],
                "qwen_background": validated["background"],
                "visible": is_visible,
                "raw_response": raw_text,
            }
            written += 1
            seq_new += 1
            print(f"[INFO] cached {key} -> target={target!r} concept={validated['concept']!r} background={validated['background']!r}")
            if args.limit_total > 0 and written >= args.limit_total:
                break

        flush_cache(f"seq={seq_name}, new={seq_new}")
        if args.limit_total > 0 and written >= args.limit_total:
            break

    print(
        f"[INFO] run finished: new={written}, skipped_existing={skipped_existing}, "
        f"total_entries={len(cache['entries'])}, output={output_path}"
    )


if __name__ == "__main__":
    main()

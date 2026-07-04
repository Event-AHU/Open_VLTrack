#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

import torch
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.train.admin.local import EnvironmentSettings
from lib.train.data.processing_utils import sample_target
from lib.train.dataset.tnl_lt import TNLLT


SYSTEM_PROMPTS = {
    "v1": """You are a structured language refiner for visual tracking.

Your job is to update only the concept and background fields for the current search image.

Rules:
- Do not rewrite the full sentence.
- Do not add new objects that are not visually supported.
- Prefer shorter and more concrete phrases.
- If a field is not supported, return an empty string for that field.
- Return JSON only with keys: concept, background.""",
    "v2": """You are refining structured tracking text for the current search region.

You may only update:
- concept
- background

Allowed operations on concept/background:
- keep
- shorten
- replace with a visually supported short phrase
- drop by returning an empty string

Forbidden:
- rewriting the whole sentence
- inventing unseen entities
- returning explanations

Return JSON only with keys: concept, background.""",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate offline Qwen refine cache for TNLLT.")
    parser.add_argument("--model-path", required=True, help="Path to a loadable Qwen2.5-VL snapshot directory.")
    parser.add_argument("--lora-path", default="", help="Optional PEFT/LoRA adapter path.")
    parser.add_argument("--output", default="", help="Output JSON cache path. If empty, build it from cache-root/split/dataset-name/exp-name.")
    parser.add_argument("--cache-root", default=str(REPO_ROOT / "output/test/qwencache"), help="Root for structured Qwen cache outputs.")
    parser.add_argument("--dataset-name", default="TNLLT", help="Dataset folder name under cache-root.")
    parser.add_argument("--exp-name", default="", help="Experiment folder name under cache-root/split/dataset-name.")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--tnllt-root", default="", help="Optional TNLLT root override.")
    parser.add_argument("--parsed-text-path", default=str(REPO_ROOT / "tools/tnllt_train_parsed_text_v2.json"))
    parser.add_argument("--search-factor", type=float, default=4.0)
    parser.add_argument("--search-size", type=int, default=256)
    parser.add_argument("--cache-interval", type=int, default=50, help="Generate Qwen cache every N frames.")
    parser.add_argument("--anchor-offset", type=int, default=49, help="Zero-based first anchor offset. 49 means 50th frame.")
    parser.add_argument("--max-seqs", type=int, default=0, help="Only process the first N sequences if > 0.")
    parser.add_argument("--max-anchors-per-seq", type=int, default=0, help="Only process the first N anchor frames per sequence if > 0.")
    parser.add_argument("--seq-start", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--system-prompt-version", default="v2", choices=sorted(SYSTEM_PROMPTS.keys()))
    parser.add_argument("--use-hints", action="store_true", help="Include coarse support hints in the prompt.")
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
    filename = f"tnllt_qwen_refine_{args.split}_full.json"
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


def to_discrete_hint(text, visible=True):
    if not text:
        return "low"
    if visible:
        return "mid"
    return "low"


def build_user_prompt(parsed, use_hints, visible):
    concept = str(parsed.get("concepts", "")).strip()
    background = str(parsed.get("background", "")).strip()
    lines = [
        f"Concept: {concept}",
        f"Background: {background}",
    ]
    if use_hints:
        lines.extend([
            f"Target evidence: {'high' if visible else 'low'}",
            f"Concept evidence: {to_discrete_hint(concept, visible=visible)}",
            f"Background reliability: {to_discrete_hint(background, visible=visible)}",
            f"Tracker confidence: {'high' if visible else 'low'}",
        ])
    lines.extend([
        "",
        "Task:",
        "Update concept and background for the current search image.",
        "Return JSON only as:",
        '{"concept":"...", "background":"..."}',
    ])
    return "\n".join(lines)


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


def validate_output(raw_output, parsed, max_field_words, max_growth):
    concept = str(parsed.get("concepts", "")).strip()
    background = str(parsed.get("background", "")).strip()
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
        offset = interval - 1
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

    env = EnvironmentSettings()
    tnllt_root = args.tnllt_root or env.tnllt_dir
    dataset = TNLLT(root=tnllt_root, split=args.split, parsed_text_path=args.parsed_text_path)
    seq_ids = list(range(args.seq_start, dataset.get_num_sequences()))
    if args.max_seqs > 0:
        seq_ids = seq_ids[:args.max_seqs]

    processor, model = load_qwen(args.model_path, args.device, args.lora_path)
    system_prompt = SYSTEM_PROMPTS[args.system_prompt_version]

    if cache is None:
        cache = {
            "meta": {
                "model_path": args.model_path,
                "lora_path": args.lora_path,
                "cache_root": args.cache_root,
                "dataset_name": args.dataset_name,
                "exp_name": args.exp_name,
                "split": args.split,
                "search_factor": args.search_factor,
                "search_size": args.search_size,
                "cache_interval": args.cache_interval,
                "anchor_offset": args.anchor_offset,
                "system_prompt_version": args.system_prompt_version,
                "use_hints": bool(args.use_hints),
                "visible_only": bool(args.visible_only),
            },
            "entries": {},
        }
    else:
        cache["meta"].update({
            "model_path": args.model_path,
            "lora_path": args.lora_path,
            "cache_root": args.cache_root,
            "dataset_name": args.dataset_name,
            "exp_name": args.exp_name,
            "split": args.split,
            "search_factor": args.search_factor,
            "search_size": args.search_size,
            "cache_interval": args.cache_interval,
            "anchor_offset": args.anchor_offset,
            "system_prompt_version": args.system_prompt_version,
            "use_hints": bool(args.use_hints),
            "visible_only": bool(args.visible_only),
        })

    def flush_cache(reason):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        print(f"[INFO] flush ({reason}) -> total entries: {len(cache['entries'])}")

    written = 0
    skipped_existing = 0
    for seq_id in seq_ids:
        seq_name = dataset.sequence_list[seq_id]
        seq_path = dataset._get_sequence_path(seq_id)
        info = dataset.get_sequence_info(seq_id)
        parsed = info["language"]
        if not isinstance(parsed, dict):
            print(f"[WARN] {seq_name}: parsed fields unavailable, skip.")
            continue
        num_frames = int(info["bbox"].shape[0])
        anchor_ids = get_anchor_ids(num_frames, args.cache_interval, args.anchor_offset)
        if args.max_anchors_per_seq > 0:
            anchor_ids = anchor_ids[:args.max_anchors_per_seq]

        seq_new = 0
        for frame_id in anchor_ids:
            visible = bool(info["visible"][frame_id].item()) if hasattr(info["visible"][frame_id], "item") else bool(info["visible"][frame_id])
            if args.visible_only and not visible:
                continue
            image_path = dataset._get_frame_path(seq_path, frame_id)
            try:
                image_rgb_full = load_image_rgb(image_path)
            except Exception as exc:
                print(f"[WARN] {exc}")
                continue
            key = f"{seq_name}:{frame_id}"
            if args.resume and key in cache["entries"]:
                skipped_existing += 1
                continue
            bbox = info["bbox"][frame_id]
            try:
                image_rgb, _, _ = sample_target(image_rgb_full, bbox, args.search_factor, output_sz=args.search_size)
            except Exception as exc:
                print(f"[WARN] {key} sample_target failed: {exc}")
                continue
            user_prompt = build_user_prompt(parsed, args.use_hints, visible)
            try:
                raw_text = run_qwen(processor, model, image_rgb, system_prompt, user_prompt, args.device, args.max_new_tokens)
                parsed_output = extract_json_object(raw_text)
                validated = validate_output(parsed_output, parsed, args.max_field_words, args.max_growth)
            except Exception as exc:
                print(f"[WARN] {key} failed: {exc}")
                continue

            cache["entries"][key] = {
                "seq_name": seq_name,
                "frame_id": frame_id,
                "image_path": image_path,
                "target": str(parsed.get("target", "")).strip(),
                "concepts": str(parsed.get("concepts", "")).strip(),
                "background": str(parsed.get("background", "")).strip(),
                "qwen_target": str(parsed.get("target", "")).strip(),
                "qwen_concept": validated["concept"],
                "qwen_background": validated["background"],
                "visible": visible,
                "raw_response": raw_text,
            }
            written += 1
            seq_new += 1
            print(f"[INFO] cached {key} -> concept={validated['concept']!r} background={validated['background']!r}")
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




#!/usr/bin/env python3
import argparse
import bisect
import csv
import json
import math
import random
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.train.data.processing_utils import sample_target


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

INVALID_LITERAL_SET = {"high", "mid", "low"}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate multi-candidate Qwen refine outputs for LaSOT anchors.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lasot-root", required=True, help="Path to LaSOTBenchmark root")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--parsed-text-path", required=True)
    parser.add_argument("--search-factor", type=float, default=4.0)
    parser.add_argument("--search-size", type=int, default=256)
    parser.add_argument("--cache-interval", type=int, default=50)
    parser.add_argument("--anchor-offset", type=int, default=49)
    parser.add_argument("--max-seqs", type=int, default=0)
    parser.add_argument("--max-anchors-per-seq", type=int, default=0)
    parser.add_argument("--seq-start", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--system-prompt-version", default="v2", choices=sorted(SYSTEM_PROMPTS.keys()))
    parser.add_argument("--use-hints", action="store_true")
    parser.add_argument("--visible-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit-total", type=int, default=0)
    parser.add_argument("--max-field-words", type=int, default=8)
    parser.add_argument("--max-growth", type=float, default=2.0)
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--num-sampling-rounds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_qwen(model_path, device):
    try:
        from transformers import AutoProcessor
    except Exception as exc:
        raise RuntimeError("transformers is required to run this script.") from exc

    model_cls = None
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
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid parsed text cache: {path}")
    return data


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
    idx = bisect.bisect_left(frame_keys, int(frame_id))
    if idx == 0:
        target_frame = frame_keys[0]
    elif idx == len(frame_keys):
        target_frame = frame_keys[-1]
    else:
        target_frame = frame_keys[idx - 1]
    parsed = item.get(str(target_frame), None)
    return normalize_parsed(parsed) if isinstance(parsed, dict) else None


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


def to_discrete_hint(text, visible=True):
    if not text:
        return "low"
    if visible:
        return "mid"
    return "low"


def build_user_prompt(parsed, use_hints, visible):
    target = str(parsed.get("target", "")).strip()
    concept = str(parsed.get("concepts", "")).strip()
    background = str(parsed.get("background", "")).strip()
    lines = [
        f"Target: {target}",
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
        "Generate one concise candidate update for concept and background for the current search image.",
        "Keep target identity consistent with the input.",
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


def validate_output(raw_output, parsed, max_field_words, max_growth):
    concept = str(parsed.get("concepts", "")).strip()
    background = str(parsed.get("background", "")).strip()
    out_concept = str(raw_output.get("concept", raw_output.get("concepts", ""))).strip()
    out_background = str(raw_output.get("background", "")).strip()

    def _check_field(name, original, updated):
        if not updated:
            return ""
        lowered = updated.strip().lower()
        if lowered in INVALID_LITERAL_SET:
            raise ValueError(f"{name} leaks hint literal: {updated!r}")
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


def candidate_signature(candidate):
    concept = str(candidate.get("concept", "")).strip().lower()
    background = str(candidate.get("background", "")).strip().lower()
    return (concept, background)


def run_qwen_sample(processor, model, image_rgb, system_prompt, user_prompt, device, max_new_tokens,
                    temperature, top_p, top_k, num_return_sequences):
    image_token = getattr(processor, "image_token", "<|image_pad|>")
    prompt_text = (
        f"System:\n{system_prompt}\n\n"
        f"User:\n<|vision_start|>{image_token}<|vision_end|>\n{user_prompt}\n\n"
        "Assistant:\n"
    )
    inputs = processor(text=[prompt_text], images=[image_rgb], padding=True, return_tensors="pt")
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_return_sequences=num_return_sequences,
            pad_token_id=processor.tokenizer.pad_token_id if hasattr(processor, "tokenizer") and processor.tokenizer.pad_token_id is not None else None,
        )
    prompt_len = inputs["input_ids"].shape[1]
    trimmed = generated[:, prompt_len:]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def get_anchor_ids(num_frames, interval, offset):
    if interval <= 0:
        raise ValueError("cache interval must be > 0")
    if offset < 0:
        offset = interval - 1
    return list(range(offset, num_frames, interval))


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and args.overwrite:
        raise SystemExit("--resume and --overwrite cannot be used together.")

    data = None
    if output_path.exists():
        if args.resume:
            with open(output_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict) or not isinstance(data.get("entries", None), dict):
                raise SystemExit(f"Invalid output format in {output_path}.")
            data.setdefault("meta", {})
            print(f"[INFO] resume enabled: loaded {len(data['entries'])} existing entries from {output_path}")
        elif not args.overwrite:
            raise SystemExit(f"Output already exists: {output_path}. Use --overwrite or --resume.")

    parsed_cache = load_parsed_cache(args.parsed_text_path)
    processor, model = load_qwen(args.model_path, args.device)
    system_prompt = SYSTEM_PROMPTS[args.system_prompt_version]

    root = Path(args.lasot_root)
    seq_names = load_sequence_list(args.split)[args.seq_start:]
    if args.max_seqs > 0:
        seq_names = seq_names[:args.max_seqs]

    meta = {
        "model_path": args.model_path,
        "dataset": "LaSOT",
        "split": args.split,
        "parsed_text_path": args.parsed_text_path,
        "search_factor": args.search_factor,
        "search_size": args.search_size,
        "cache_interval": args.cache_interval,
        "anchor_offset": args.anchor_offset,
        "system_prompt_version": args.system_prompt_version,
        "use_hints": bool(args.use_hints),
        "visible_only": bool(args.visible_only),
        "num_candidates": args.num_candidates,
        "num_sampling_rounds": args.num_sampling_rounds,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "seed": args.seed,
    }
    if data is None:
        data = {"meta": meta, "entries": {}}
    else:
        data["meta"].update(meta)

    def flush_output(reason):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[INFO] flush ({reason}) -> total entries: {len(data['entries'])}")

    written = 0
    skipped_existing = 0
    for seq_name in seq_names:
        class_name = seq_name.split("-")[0]
        seq_path = root / class_name / seq_name
        boxes = read_gt(seq_path)
        visible = read_visible(seq_path)
        frame_dir = seq_path / "img"
        frame_files = sorted(frame_dir.glob("*.jpg"))
        anchor_ids = get_anchor_ids(len(frame_files), args.cache_interval, args.anchor_offset)
        if args.max_anchors_per_seq > 0:
            anchor_ids = anchor_ids[:args.max_anchors_per_seq]

        seq_new = 0
        for frame_id in anchor_ids:
            if frame_id >= len(boxes):
                continue
            parsed = nearest_parsed(parsed_cache, seq_name, frame_id)
            if not isinstance(parsed, dict):
                print(f"[WARN] {seq_name}:{frame_id} parsed fields unavailable, skip.")
                continue
            if not parsed.get("target"):
                parsed["target"] = class_name
            is_visible = bool(visible[frame_id]) if frame_id < len(visible) else True
            if args.visible_only and not is_visible:
                continue

            key = f"{seq_name}:{frame_id}"
            if args.resume and key in data["entries"]:
                skipped_existing += 1
                continue

            image_bgr = cv2.imread(str(frame_files[frame_id]))
            if image_bgr is None:
                print(f"[WARN] Failed to read image: {frame_files[frame_id]}")
                continue
            try:
                search_crop, _, _ = sample_target(image_bgr, boxes[frame_id], args.search_factor, output_sz=args.search_size)
                image_rgb = cv2.cvtColor(search_crop, cv2.COLOR_BGR2RGB)
            except Exception as exc:
                print(f"[WARN] {key} sample_target failed: {exc}")
                continue

            user_prompt = build_user_prompt(parsed, args.use_hints, is_visible)
            unique_candidates = {}
            invalid_samples = []
            total_raw_generations = 0
            for _ in range(max(1, args.num_sampling_rounds)):
                remain = args.num_candidates - len(unique_candidates)
                if remain <= 0:
                    break
                try:
                    raw_outputs = run_qwen_sample(
                        processor, model, image_rgb, system_prompt, user_prompt, args.device,
                        args.max_new_tokens, args.temperature, args.top_p, args.top_k, remain,
                    )
                except Exception as exc:
                    print(f"[WARN] {key} generation failed: {exc}")
                    break
                total_raw_generations += len(raw_outputs)
                for raw_text in raw_outputs:
                    try:
                        parsed_output = extract_json_object(raw_text)
                        validated = validate_output(parsed_output, parsed, args.max_field_words, args.max_growth)
                        sig = candidate_signature(validated)
                        if sig in unique_candidates:
                            continue
                        unique_candidates[sig] = {
                            "concept": validated["concept"],
                            "background": validated["background"],
                            "raw_response": raw_text,
                        }
                        if len(unique_candidates) >= args.num_candidates:
                            break
                    except Exception as exc:
                        invalid_samples.append({"raw_response": raw_text, "error": str(exc)})
                if len(unique_candidates) >= args.num_candidates:
                    break

            candidates = list(unique_candidates.values())
            if not candidates:
                print(f"[WARN] {key} has no valid candidates.")
                continue

            data["entries"][key] = {
                "seq_name": seq_name,
                "frame_id": frame_id,
                "image_path": str(frame_files[frame_id]),
                "visible": is_visible,
                "target": str(parsed.get("target", "")).strip(),
                "concepts": str(parsed.get("concepts", "")).strip(),
                "background": str(parsed.get("background", "")).strip(),
                "num_valid_candidates": len(candidates),
                "num_invalid_samples": len(invalid_samples),
                "num_raw_generations": total_raw_generations,
                "candidates": candidates,
                "invalid_samples": invalid_samples[:8],
            }
            written += 1
            seq_new += 1
            print(f"[INFO] candidates {key} -> valid={len(candidates)} invalid={len(invalid_samples)}")
            if args.limit_total > 0 and written >= args.limit_total:
                break

        flush_output(f"seq={seq_name}, new={seq_new}")
        if args.limit_total > 0 and written >= args.limit_total:
            break

    print(
        f"[INFO] run finished: new={written}, skipped_existing={skipped_existing}, "
        f"total_entries={len(data['entries'])}, output={output_path}"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import json
import math
import random
import re
import sys
from pathlib import Path

import cv2
import torch

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

INVALID_LITERAL_SET = {"high", "mid", "low"}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate multi-candidate Qwen refine outputs for TNLLT anchors.")
    parser.add_argument("--model-path", required=True, help="Path to a loadable Qwen2.5-VL snapshot directory.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--tnllt-root", default="", help="Optional TNLLT root override.")
    parser.add_argument("--parsed-text-path", required=True, help="Path to parsed TNLLT text json.")
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
    parser.add_argument("--num-sampling-rounds", type=int, default=2, help="Rounds of repeated sampling to improve unique candidate count.")
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
    decoded = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return decoded


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
            if not isinstance(data, dict):
                raise SystemExit(f"Invalid output format in {output_path}: root is not a dict.")
            if "entries" not in data or not isinstance(data["entries"], dict):
                raise SystemExit(f"Invalid output format in {output_path}: missing dict field 'entries'.")
            if "meta" not in data or not isinstance(data["meta"], dict):
                data["meta"] = {}
            print(f"[INFO] resume enabled: loaded {len(data['entries'])} existing entries from {output_path}")
        elif not args.overwrite:
            raise SystemExit(f"Output already exists: {output_path}. Use --overwrite to replace it or --resume to continue.")

    env = EnvironmentSettings()
    tnllt_root = args.tnllt_root or env.tnllt_dir
    dataset = TNLLT(root=tnllt_root, split=args.split, parsed_text_path=args.parsed_text_path)
    seq_ids = list(range(args.seq_start, dataset.get_num_sequences()))
    if args.max_seqs > 0:
        seq_ids = seq_ids[:args.max_seqs]

    processor, model = load_qwen(args.model_path, args.device)
    system_prompt = SYSTEM_PROMPTS[args.system_prompt_version]

    if data is None:
        data = {
            "meta": {
                "model_path": args.model_path,
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
            },
            "entries": {},
        }
    else:
        data["meta"].update({
            "model_path": args.model_path,
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
        })

    def flush_output(reason):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[INFO] flush ({reason}) -> total entries: {len(data['entries'])}")

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
            key = f"{seq_name}:{frame_id}"
            if args.resume and key in data["entries"]:
                skipped_existing += 1
                continue

            image_path = dataset._get_frame_path(seq_path, frame_id)
            image_bgr = cv2.imread(image_path)
            if image_bgr is None:
                print(f"[WARN] Failed to read image: {image_path}")
                continue
            bbox = info["bbox"][frame_id]
            try:
                search_crop, _, _ = sample_target(image_bgr, bbox, args.search_factor, output_sz=args.search_size)
                image_rgb = cv2.cvtColor(search_crop, cv2.COLOR_BGR2RGB)
            except Exception as exc:
                print(f"[WARN] {key} sample_target failed: {exc}")
                continue

            user_prompt = build_user_prompt(parsed, args.use_hints, visible)
            unique_candidates = {}
            invalid_samples = []
            total_raw_generations = 0
            for _ in range(max(1, args.num_sampling_rounds)):
                remain = args.num_candidates - len(unique_candidates)
                if remain <= 0:
                    break
                try:
                    raw_outputs = run_qwen_sample(
                        processor,
                        model,
                        image_rgb,
                        system_prompt,
                        user_prompt,
                        args.device,
                        args.max_new_tokens,
                        args.temperature,
                        args.top_p,
                        args.top_k,
                        remain,
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
                        invalid_samples.append({
                            "raw_response": raw_text,
                            "error": str(exc),
                        })
                if len(unique_candidates) >= args.num_candidates:
                    break

            candidates = list(unique_candidates.values())
            if not candidates:
                print(f"[WARN] {key} has no valid candidates.")
                continue

            data["entries"][key] = {
                "seq_name": seq_name,
                "frame_id": frame_id,
                "image_path": image_path,
                "visible": visible,
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

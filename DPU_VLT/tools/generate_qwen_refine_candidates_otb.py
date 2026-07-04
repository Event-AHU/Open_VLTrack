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
from lib.train.dataset.otb_lang import OTB99Lang


SYSTEM_PROMPTS = {
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--otb-root", default="")
    parser.add_argument("--split", default="train", choices=["train"])
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
    from transformers import AutoProcessor
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
        model_cls = Qwen2_5_VLForConditionalGeneration
    except Exception:
        from transformers import AutoModelForVision2Seq
        model_cls = AutoModelForVision2Seq
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = model_cls.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True).to(device).eval()
    return processor, model


def build_user_prompt(parsed, use_hints, visible):
    lines = [
        f"Target: {str(parsed.get('target', '')).strip()}",
        f"Concept: {str(parsed.get('concepts', '')).strip()}",
        f"Background: {str(parsed.get('background', '')).strip()}",
        "",
        "Task:",
        "Generate one concise candidate update for concept and background for the current search image.",
        "Keep target identity consistent with the input.",
        "Return JSON only as:",
        '{"concept":"...", "background":"..."}',
    ]
    return "\n".join(lines)


def extract_json_object(text):
    if not text:
        raise ValueError("Empty generation output.")
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            candidate = text[start:end + 1]
    if candidate is None:
        raise ValueError(f"No JSON in: {text[:200]}")
    return json.loads(candidate)


def validate_output(raw_output, parsed, max_field_words, max_growth):
    def _check(name, original, updated):
        if not updated:
            return ""
        if updated.strip().lower() in INVALID_LITERAL_SET:
            raise ValueError(f"{name} leaks hint: {updated!r}")
        words = updated.split()
        if len(words) > max_field_words:
            raise ValueError(f"{name} too long: {updated!r}")
        orig_words = max(1, len(original.split())) if original else 1
        if len(words) > math.ceil(orig_words * max_growth):
            raise ValueError(f"{name} grows too much")
        return updated
    return {
        "concept": _check("concept", str(parsed.get("concepts", "")), str(raw_output.get("concept", raw_output.get("concepts", ""))).strip()),
        "background": _check("background", str(parsed.get("background", "")), str(raw_output.get("background", "")).strip()),
    }


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
            **inputs, max_new_tokens=max_new_tokens, do_sample=True,
            temperature=temperature, top_p=top_p, top_k=top_k,
            num_return_sequences=num_return_sequences,
            pad_token_id=getattr(getattr(processor, "tokenizer", None), "pad_token_id", None),
        )
    prompt_len = inputs["input_ids"].shape[1]
    return processor.batch_decode(generated[:, prompt_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False)


def get_anchor_ids(num_frames, interval, offset):
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
            print(f"[INFO] resume: loaded {len(data['entries'])} existing entries")
        elif not args.overwrite:
            raise SystemExit(f"Output exists: {output_path}. Use --overwrite or --resume.")

    env = EnvironmentSettings()
    otb_root = args.otb_root or env.otb99lang_dir
    dataset = OTB99Lang(root=otb_root, split=args.split, parsed_text_path=args.parsed_text_path)
    seq_ids = list(range(args.seq_start, dataset.get_num_sequences()))
    if args.max_seqs > 0:
        seq_ids = seq_ids[:args.max_seqs]

    processor, model = load_qwen(args.model_path, args.device)
    system_prompt = SYSTEM_PROMPTS[args.system_prompt_version]

    meta = {
        "model_path": args.model_path, "dataset": "OTB99Lang", "split": args.split,
        "parsed_text_path": args.parsed_text_path, "search_factor": args.search_factor,
        "search_size": args.search_size, "cache_interval": args.cache_interval,
        "anchor_offset": args.anchor_offset, "num_candidates": args.num_candidates,
        "temperature": args.temperature, "top_p": args.top_p, "seed": args.seed,
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
    for seq_id in seq_ids:
        seq_name = dataset.sequence_list[seq_id]
        seq_path = dataset._get_sequence_path(seq_id)
        info = dataset.get_sequence_info(seq_id)
        num_frames = int(info["bbox"].shape[0])
        anchor_ids = get_anchor_ids(num_frames, args.cache_interval, args.anchor_offset)
        if args.max_anchors_per_seq > 0:
            anchor_ids = anchor_ids[:args.max_anchors_per_seq]

        parsed = dataset.get_language_description(seq_id)  # OTB: same text for all frames

        seq_new = 0
        for frame_id in anchor_ids:
            if not isinstance(parsed, dict):
                continue
            visible = bool(info["visible"][frame_id].item()) if hasattr(info["visible"][frame_id], "item") else bool(info["visible"][frame_id])
            if args.visible_only and not visible:
                continue
            key = f"{seq_name}:{frame_id}"
            if args.resume and key in data["entries"]:
                continue

            image_path = dataset._get_frame_path(seq_path, frame_id)
            image_bgr = cv2.imread(image_path)
            if image_bgr is None:
                print(f"[WARN] Failed to read: {image_path}")
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
            total_raw = 0
            for _ in range(max(1, args.num_sampling_rounds)):
                remain = args.num_candidates - len(unique_candidates)
                if remain <= 0:
                    break
                try:
                    raw_outputs = run_qwen_sample(processor, model, image_rgb, system_prompt, user_prompt,
                                                  args.device, args.max_new_tokens, args.temperature,
                                                  args.top_p, args.top_k, remain)
                except Exception as exc:
                    print(f"[WARN] {key} generation failed: {exc}")
                    break
                total_raw += len(raw_outputs)
                for raw_text in raw_outputs:
                    try:
                        parsed_output = extract_json_object(raw_text)
                        validated = validate_output(parsed_output, parsed, args.max_field_words, args.max_growth)
                        sig = (validated["concept"].lower(), validated["background"].lower())
                        if sig not in unique_candidates:
                            unique_candidates[sig] = {"concept": validated["concept"], "background": validated["background"], "raw_response": raw_text}
                    except Exception as exc:
                        invalid_samples.append({"raw_response": raw_text, "error": str(exc)})
                if len(unique_candidates) >= args.num_candidates:
                    break

            candidates = list(unique_candidates.values())
            if not candidates:
                print(f"[WARN] {key} no valid candidates.")
                continue

            data["entries"][key] = {
                "seq_name": seq_name, "frame_id": frame_id, "image_path": image_path,
                "visible": visible,
                "target": str(parsed.get("target", "")).strip(),
                "concepts": str(parsed.get("concepts", "")).strip(),
                "background": str(parsed.get("background", "")).strip(),
                "num_valid_candidates": len(candidates),
                "num_invalid_samples": len(invalid_samples),
                "num_raw_generations": total_raw,
                "candidates": candidates,
                "invalid_samples": invalid_samples[:8],
            }
            written += 1
            seq_new += 1
            print(f"[INFO] {key} -> valid={len(candidates)} invalid={len(invalid_samples)}")
            if args.limit_total > 0 and written >= args.limit_total:
                break

        flush_output(f"seq={seq_name}, new={seq_new}")
        if args.limit_total > 0 and written >= args.limit_total:
            break

    print(f"[INFO] done: new={written}, total_entries={len(data['entries'])}, output={output_path}")


if __name__ == "__main__":
    main()

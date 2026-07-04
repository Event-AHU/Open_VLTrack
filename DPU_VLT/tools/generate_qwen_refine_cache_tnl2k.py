#!/usr/bin/env python3
import argparse
import json
import math
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
from lib.train.dataset.tnl2k import TNL2k_Lang


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


def parse_args():
    parser = argparse.ArgumentParser(description="Generate offline Qwen refine cache for TNL2K.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--lora-path", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tnl2k-root", default="")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--parsed-text-path", default="")
    parser.add_argument("--search-factor", type=float, default=4.0)
    parser.add_argument("--search-size", type=int, default=256)
    parser.add_argument("--cache-interval", type=int, default=50)
    parser.add_argument("--anchor-offset", type=int, default=49)
    parser.add_argument("--max-seqs", type=int, default=0)
    parser.add_argument("--max-anchors-per-seq", type=int, default=0)
    parser.add_argument("--seq-start", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--system-prompt-version", default="v2", choices=["v2"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit-total", type=int, default=0)
    parser.add_argument("--max-field-words", type=int, default=6)
    parser.add_argument("--max-growth", type=float, default=2.0)
    return parser.parse_args()


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
        raise ValueError("No JSON object found in model output.")
    return json.loads(candidate)


def build_user_prompt(parsed):
    target = str(parsed.get("target", "")).strip()
    concept = str(parsed.get("concepts", "")).strip()
    background = str(parsed.get("background", "")).strip()
    return "\n".join(
        [
            f"Target: {target}",
            f"Concept: {concept}",
            f"Background: {background}",
            "",
            "Task:",
            "Update concept and background for the current search image.",
            "Return JSON only as:",
            '{"concept":"...", "background":"..."}',
        ]
    )


def validate_output(raw_output, parsed, max_field_words, max_growth):
    concept = str(parsed.get("concepts", "")).strip()
    background = str(parsed.get("background", "")).strip()
    out_concept = str(raw_output.get("concept", raw_output.get("concepts", ""))).strip()
    out_background = str(raw_output.get("background", "")).strip()

    def _check(original, updated, name):
        if not updated:
            return ""
        words = updated.split()
        if len(words) > max_field_words:
            raise ValueError(f"{name} too long")
        orig_words = max(1, len(original.split())) if original else 1
        if len(words) > math.ceil(orig_words * max_growth):
            raise ValueError(f"{name} grows too much")
        return updated

    return {
        "concept": _check(concept, out_concept, "concept"),
        "background": _check(background, out_background, "background"),
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
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.resume and args.overwrite:
        raise SystemExit("--resume and --overwrite cannot be used together.")
    data = None
    if output_path.exists():
        if args.resume:
            data = json.load(open(output_path, "r", encoding="utf-8"))
            if "entries" not in data:
                raise SystemExit("Invalid existing cache format.")
        elif not args.overwrite:
            raise SystemExit(f"Output exists: {output_path}. Use --overwrite or --resume.")

    env = EnvironmentSettings()
    tnl2k_root = args.tnl2k_root or env.tnl2k_dir
    dataset = TNL2k_Lang(root=tnl2k_root, split=args.split, parsed_text_path=args.parsed_text_path)
    seq_ids = list(range(args.seq_start, dataset.get_num_sequences()))

    # Load flat parsed text (seq_name -> {raw, target, concepts, background})
    # _get_cached_nlp expects frame_id-nested structure; flat test cache needs direct lookup
    flat_parsed = {}
    if args.parsed_text_path and Path(args.parsed_text_path).exists():
        with open(args.parsed_text_path, "r", encoding="utf-8") as f:
            raw_parsed = json.load(f)
        for seq_name, item in raw_parsed.items():
            if isinstance(item, dict) and "target" in item:
                concepts = item.get("concepts", [])
                background = item.get("background", [])
                flat_parsed[seq_name] = {
                    "raw": str(item.get("raw", "")),
                    "target": str(item.get("target", "")),
                    "concepts": " ".join(str(x) for x in concepts) if isinstance(concepts, list) else str(concepts),
                    "background": " ".join(str(x) for x in background) if isinstance(background, list) else str(background),
                }
    if args.max_seqs > 0:
        seq_ids = seq_ids[: args.max_seqs]

    processor, model = load_qwen(args.model_path, args.device, args.lora_path)
    system_prompt = SYSTEM_PROMPTS[args.system_prompt_version]

    if data is None:
        data = {
            "meta": {
                "model_path": args.model_path,
                "lora_path": args.lora_path,
                "dataset": "TNL2K",
                "split": args.split,
                "parsed_text_path": args.parsed_text_path,
                "cache_interval": args.cache_interval,
                "anchor_offset": args.anchor_offset,
            },
            "entries": {},
        }

    written = 0
    skipped_existing = 0
    for seq_id in seq_ids:
        seq_name = dataset.sequence_list[seq_id]
        seq_path = dataset._get_sequence_path(seq_id)
        info = dataset.get_sequence_info(seq_id)
        num_frames = int(info["bbox"].shape[0])
        anchor_ids = get_anchor_ids(num_frames, args.cache_interval, args.anchor_offset)
        if args.max_anchors_per_seq > 0:
            anchor_ids = anchor_ids[: args.max_anchors_per_seq]

        for frame_id in anchor_ids:
            key = f"{seq_name}:{frame_id}"
            if args.resume and key in data["entries"]:
                skipped_existing += 1
                continue

            parsed = flat_parsed.get(seq_name) or dataset.get_language_description(seq_id, int(frame_id))
            if not isinstance(parsed, dict) or not parsed.get("target") and not parsed.get("concepts"):
                continue
            image_path = dataset._get_frame_path(seq_path, seq_name, frame_id)
            if image_path is None:
                continue
            image_bgr = cv2.imread(image_path)
            if image_bgr is None:
                continue
            bbox = info["bbox"][frame_id]
            try:
                search_crop, _, _ = sample_target(image_bgr, bbox, args.search_factor, output_sz=args.search_size)
                image_rgb = cv2.cvtColor(search_crop, cv2.COLOR_BGR2RGB)
                raw_text = run_qwen(
                    processor, model, image_rgb, system_prompt, build_user_prompt(parsed), args.device, args.max_new_tokens
                )
                obj = extract_json_object(raw_text)
                valid = validate_output(obj, parsed, args.max_field_words, args.max_growth)
            except Exception:
                continue

            data["entries"][key] = {
                "seq_name": seq_name,
                "frame_id": frame_id,
                "image_path": image_path,
                "target": str(parsed.get("target", "")).strip(),
                "concepts": str(parsed.get("concepts", "")).strip(),
                "background": str(parsed.get("background", "")).strip(),
                "qwen_target": str(parsed.get("target", "")).strip(),
                "qwen_concept": valid["concept"],
                "qwen_background": valid["background"],
                "raw_response": raw_text,
            }
            written += 1
            if args.limit_total > 0 and written >= args.limit_total:
                break

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        if args.limit_total > 0 and written >= args.limit_total:
            break

    print(f"[DONE] written={written}, skipped_existing={skipped_existing}, total={len(data['entries'])}")
    print(f"[DONE] output={output_path}")


if __name__ == "__main__":
    main()

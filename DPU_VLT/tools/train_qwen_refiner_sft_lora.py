#!/usr/bin/env python3
import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset

from peft import LoraConfig, get_peft_model
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForVision2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA SFT for Qwen tracking refiner (multimodal: image + text).")
    parser.add_argument("--model-path", required=True, help="Base model path, e.g. Qwen2.5-VL-3B-Instruct local snapshot")
    parser.add_argument("--train-jsonl", required=True, help="SFT jsonl with messages + image_path")
    parser.add_argument("--output-dir", required=True, help="Directory to save LoRA adapter")
    parser.add_argument("--resume-from-checkpoint", default="", help="Path to checkpoint dir to resume from")

    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--train-samples", type=int, default=0, help="0 means use all")

    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-train-epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=3)

    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated module names",
    )

    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    return parser.parse_args()


def resolve_torch_dtype(name: str):
    if name == "auto":
        return "auto"
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported torch dtype: {name}")


def load_processor(model_path: str):
    return AutoProcessor.from_pretrained(model_path, trust_remote_code=True)


def load_tokenizer(model_path: str, processor):
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        return processor.tokenizer
    try:
        return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load tokenizer from processor/AutoTokenizer: {e}")


def load_model(model_path: str, torch_dtype):
    loaders = [
        ("AutoModelForImageTextToText", AutoModelForImageTextToText),
        ("AutoModelForVision2Seq", AutoModelForVision2Seq),
        ("AutoModelForCausalLM", AutoModelForCausalLM),
    ]
    errors = []
    for name, cls in loaders:
        try:
            model = cls.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
            )
            print(f"[INFO] loaded model with {name}")
            return model
        except Exception as e:
            errors.append((name, str(e)))

    msg = "\n".join([f"- {n}: {e}" for n, e in errors])
    raise RuntimeError(
        "Failed to load model with all supported auto classes. "
        "Please check model path and transformers version.\n"
        f"Tried:\n{msg}"
    )


def build_manual_qwen_vl_prompt(system_text: str, user_text: str, add_generation_prompt: bool = True):
    # Manual fallback template for local Qwen2.5-VL snapshots without chat_template.
    # This explicitly inserts image placeholder tokens so image features align with image tokens.
    parts = [
        "<|im_start|>system\n" + str(system_text).strip() + "<|im_end|>\n",
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\n" + str(user_text).strip() + "<|im_end|>\n",
    ]
    if add_generation_prompt:
        parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def build_chatml_with_processor(processor, system_text: str, user_text: str, assistant_text: str = "", add_generation_prompt: bool = True):
    messages = [
        {"role": "system", "content": [{"type": "text", "text": str(system_text).strip()}]},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": str(user_text).strip()},
            ],
        },
    ]
    if not add_generation_prompt:
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": str(assistant_text).strip()}]}
        )
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def load_image_any(path: str):
    # 1) Pillow
    try:
        from PIL import Image

        return Image.open(path).convert("RGB")
    except Exception:
        pass

    # 2) OpenCV
    try:
        import cv2

        bgr = cv2.imread(path)
        if bgr is None:
            raise RuntimeError(f"cv2.imread failed: {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        pass

    # 3) imageio
    try:
        import imageio.v3 as iio

        arr = iio.imread(path)
        if arr is None:
            raise RuntimeError(f"imageio failed: {path}")
        return arr
    except Exception as e:
        raise RuntimeError(
            f"Failed to load image {path}. Install pillow or opencv-python. Original error: {e}"
        )


def load_sft_records(path: Path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            msgs = obj.get("messages", None)
            image_path = str(obj.get("image_path", "")).strip()
            if not isinstance(msgs, list) or len(msgs) < 3:
                continue
            if not image_path:
                continue
            records.append(obj)
    return records


class MultimodalSFTDataset(Dataset):
    def __init__(self, records):
        self.records = records
        if len(self.records) == 0:
            raise RuntimeError("No valid records loaded.")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


@dataclass
class MultimodalSFTCollator:
    processor: object
    tokenizer: object
    max_length: int

    def __call__(self, features: List[Dict]):
        images = []
        full_texts = []
        prompt_texts = []

        eos = self.tokenizer.eos_token or ""

        for rec in features:
            image_path = str(rec.get("image_path", "")).strip()
            image = load_image_any(image_path)
            images.append(image)

            msgs = rec["messages"]
            system_text = str(msgs[0].get("content", ""))
            user_text = str(msgs[1].get("content", ""))
            assistant_text = str(msgs[2].get("content", ""))

            # Prefer processor chat template to keep image token count aligned with image features.
            try:
                prompt_text = build_chatml_with_processor(
                    self.processor,
                    system_text=system_text,
                    user_text=user_text,
                    add_generation_prompt=True,
                )
                full_text = build_chatml_with_processor(
                    self.processor,
                    system_text=system_text,
                    user_text=user_text,
                    assistant_text=assistant_text + eos,
                    add_generation_prompt=False,
                )
            except Exception:
                prompt_text = build_manual_qwen_vl_prompt(
                    system_text=system_text,
                    user_text=user_text,
                    add_generation_prompt=True,
                )
                full_text = prompt_text + assistant_text + "<|im_end|>" + eos

            prompt_texts.append(prompt_text)
            full_texts.append(full_text)

        # Do not truncate VL inputs. Truncation may cut image placeholder tokens and
        # trigger "image features and image tokens do not match" in Qwen2.5-VL.
        full_batch = self.processor(
            text=full_texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

        # Fail fast when image tokens are missing in text prompt (common cause of VL mismatch).
        image_token_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        if isinstance(image_token_id, int) and image_token_id >= 0:
            image_token_count = int((full_batch["input_ids"] == image_token_id).sum().item())
            if image_token_count == 0:
                raise RuntimeError(
                    "No image tokens found in input_ids. Check manual prompt token construction and processor compatibility."
                )
        prompt_batch = self.processor(
            text=prompt_texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

        labels = full_batch["input_ids"].clone()
        prompt_lens = prompt_batch["attention_mask"].sum(dim=1)

        for i, pl in enumerate(prompt_lens.tolist()):
            pl = int(pl)
            if pl > 0:
                labels[i, :pl] = -100

        labels[full_batch["attention_mask"] == 0] = -100
        full_batch["labels"] = labels
        return full_batch


def main():
    args = parse_args()
    set_seed(args.seed)
    random.seed(args.seed)

    model_path = args.model_path
    train_jsonl = Path(args.train_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not train_jsonl.exists():
        raise FileNotFoundError(f"Train jsonl not found: {train_jsonl}")

    torch_dtype = resolve_torch_dtype(args.torch_dtype)

    print(f"[INFO] loading processor from {model_path}")
    processor = load_processor(model_path)
    tokenizer = load_tokenizer(model_path, processor)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[INFO] loading model from {model_path}")
    model = load_model(model_path, torch_dtype=torch_dtype)

    target_modules = [x.strip() for x in args.lora_target_modules.split(",") if x.strip()]
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    if args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    print(f"[INFO] loading SFT data from {train_jsonl}")
    records = load_sft_records(train_jsonl)
    if args.train_samples > 0:
        records = records[: args.train_samples]
    print(f"[INFO] raw records={len(records)}")

    train_dataset = MultimodalSFTDataset(records)
    print(f"[INFO] train samples={len(train_dataset)}")

    data_collator = MultimodalSFTCollator(
        processor=processor,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    train_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=(args.torch_dtype == "bfloat16"),
        fp16=(args.torch_dtype == "float16"),
        report_to=[],
        dataloader_num_workers=0,
        remove_unused_columns=False,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print("[INFO] start training")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint if args.resume_from_checkpoint else None)

    print(f"[INFO] saving LoRA adapter to {output_dir}")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    with open(output_dir / "train_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_path": model_path,
                "train_jsonl": str(train_jsonl),
                "train_samples": len(train_dataset),
                "max_length": args.max_length,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "target_modules": target_modules,
                "multimodal": True,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("[DONE] training complete")


if __name__ == "__main__":
    main()

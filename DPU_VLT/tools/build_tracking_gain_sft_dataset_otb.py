#!/usr/bin/env python3
import argparse
import importlib
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.train.admin.local import EnvironmentSettings
from lib.train.dataset.otb_lang import OTB99Lang


SYSTEM_PROMPTS = {
    "v2": "You are refining structured tracking text for the current search region.",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates-path", required=True)
    parser.add_argument("--output-score-path", required=True)
    parser.add_argument("--output-sft-path", required=True)
    parser.add_argument("--tracker-name", default="lantrack")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seq-start", type=int, default=0)
    parser.add_argument("--max-seqs", type=int, default=0)
    parser.add_argument("--max-candidates-per-anchor", type=int, default=4)
    parser.add_argument("--gain-th", type=float, default=0.003)
    parser.add_argument("--concept-only", action="store_true")
    parser.add_argument("--otb-root", default="")
    parser.add_argument("--split", default="train", choices=["train"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--system-prompt-version", default="v2")
    return parser.parse_args()


def xywh_to_xyxy(box):
    x, y, w, h = [float(v) for v in box]
    return np.array([x, y, x + w, y + h], dtype=np.float64)


def compute_iou(box_a, box_b):
    a, b = xywh_to_xyxy(box_a), xywh_to_xyxy(box_b)
    inter_w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    inter_h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = inter_w * inter_h
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def build_seq_map(otb_root=""):
    env = EnvironmentSettings()
    root = otb_root or env.otb99lang_dir
    dataset = OTB99Lang(root=root, split="train")
    seq_map = {}
    for seq_id, seq_name in enumerate(dataset.sequence_list):
        info = dataset.get_sequence_info(seq_id)
        seq_path = dataset._get_sequence_path(seq_id)
        frames = [dataset._get_frame_path(seq_path, i) for i in range(int(info["bbox"].shape[0]))]
        seq_map[seq_name] = {
            "name": seq_name,
            "frames": frames,
            "ground_truth_rect": info["bbox"].cpu().numpy(),
            "object_class": seq_name,
        }
    return seq_map


def build_baseline_text(entry):
    return {
        "raw": "",
        "target": str(entry.get("target", "")).strip(),
        "concepts": str(entry.get("concepts", "")).strip(),
        "background": str(entry.get("background", "")).strip(),
    }


def build_candidate_text(entry, candidate, concept_only=False):
    out = build_baseline_text(entry)
    cand_concept = str(candidate.get("concept", "")).strip()
    cand_background = str(candidate.get("background", "")).strip()
    if cand_concept:
        out["concepts"] = cand_concept
    if not concept_only and cand_background != "":
        out["background"] = cand_background
    return out


def build_user_prompt(text_struct):
    return "\n".join([
        f"Target: {str(text_struct.get('target', '')).strip()}",
        f"Concept: {str(text_struct.get('concepts', '')).strip()}",
        f"Background: {str(text_struct.get('background', '')).strip()}",
        "", "Task:",
        "Update concept and background for the current search image.",
        "Return JSON only as:", '{"concept":"...", "background":"..."}',
    ])


def build_sft_record(entry, selected_text, gain, system_prompt_version):
    return {
        "seq_name": entry["seq_name"],
        "frame_id": entry["frame_id"],
        "image_path": entry["image_path"],
        "gain": float(gain),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPTS[system_prompt_version]},
            {"role": "user", "content": build_user_prompt({"target": entry.get("target", ""), "concepts": entry.get("concepts", ""), "background": entry.get("background", "")})},
            {"role": "assistant", "content": json.dumps({"concept": selected_text.get("concepts", ""), "background": selected_text.get("background", "")}, ensure_ascii=False)},
        ],
    }


def make_tracker(args):
    params = importlib.import_module(f"lib.test.parameter.{args.tracker_name}").parameters(args.config)
    if args.checkpoint:
        params.checkpoint = args.checkpoint
    params.debug = 0
    params.save_all_boxes = False
    tracker_cls = importlib.import_module(f"lib.test.tracker.{args.tracker_name}").get_tracker_class()
    return tracker_cls(params)


def run_tracking_window(tracker, seq, start_frame, text_struct, window_size):
    image = cv2.cvtColor(cv2.imread(seq["frames"][start_frame]), cv2.COLOR_BGR2RGB)
    init_bbox = list(seq["ground_truth_rect"][start_frame])
    tracker.initialize(image, {
        "init_bbox": init_bbox, "text_description": text_struct,
        "init_text_description": text_struct, "class": seq["object_class"],
        "path": seq["name"], "num": start_frame,
    })
    end_frame = min(len(seq["frames"]), start_frame + window_size + 1)
    pred_boxes = [init_bbox]
    gt_boxes = [list(seq["ground_truth_rect"][start_frame])]
    for frame_num in range(start_frame + 1, end_frame):
        image = cv2.cvtColor(cv2.imread(seq["frames"][frame_num]), cv2.COLOR_BGR2RGB)
        out = tracker.track(image, {"previous_output": {"target_bbox": pred_boxes[-1]}, "class": seq["object_class"], "path": seq["name"], "num": frame_num, "text_description": text_struct})
        pred_box = out.get("target_bbox")
        if pred_box is None:
            break
        pred_boxes.append(list(pred_box))
        gt_boxes.append(list(seq["ground_truth_rect"][frame_num]))
    ious = [compute_iou(p, g) for p, g in zip(pred_boxes, gt_boxes)]
    return float(np.mean(ious)) if ious else 0.0, ious


def score_anchor(entry, seq, tracker, args):
    frame_id = int(entry["frame_id"])
    baseline_text = build_baseline_text(entry)
    baseline_score, baseline_ious = run_tracking_window(tracker, seq, frame_id, baseline_text, args.window_size)
    scored_candidates = []
    for idx, candidate in enumerate(entry.get("candidates", [])[:args.max_candidates_per_anchor]):
        cand_text = build_candidate_text(entry, candidate, concept_only=args.concept_only)
        cand_score, cand_ious = run_tracking_window(tracker, seq, frame_id, cand_text, args.window_size)
        scored_candidates.append({
            "candidate_id": idx, "concept": str(candidate.get("concept", "")).strip(),
            "background": str(candidate.get("background", "")).strip(),
            "score": cand_score, "gain": float(cand_score - baseline_score),
            "ious": cand_ious, "text": cand_text,
        })
    best = max(scored_candidates, key=lambda x: x["gain"]) if scored_candidates else None
    return {
        "seq_name": entry["seq_name"], "frame_id": frame_id, "image_path": entry["image_path"],
        "baseline_text": baseline_text, "baseline_score": baseline_score, "baseline_ious": baseline_ious,
        "candidates": scored_candidates, "best_candidate": best,
        "best_gain": float(best["gain"]) if best else None,
    }


def make_sample_key(seq_name, frame_id):
    return f"{seq_name}:{int(frame_id)}"


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    score_path = Path(args.output_score_path)
    sft_path = Path(args.output_sft_path)
    if args.overwrite and args.resume:
        raise SystemExit("Use either --overwrite or --resume, not both.")
    if (score_path.exists() or sft_path.exists()) and not args.overwrite and not args.resume:
        raise SystemExit("Output exists. Use --overwrite or --resume.")
    score_path.parent.mkdir(parents=True, exist_ok=True)
    sft_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.candidates_path, "r", encoding="utf-8") as f:
        candidate_data = json.load(f)
    entries = list(candidate_data["entries"].values())
    seq_map = build_seq_map(otb_root=args.otb_root)

    selected_seq_names = sorted({item["seq_name"] for item in entries if item["seq_name"] in seq_map})
    if args.seq_start > 0:
        selected_seq_names = selected_seq_names[args.seq_start:]
    if args.max_seqs > 0:
        selected_seq_names = selected_seq_names[:args.max_seqs]
    selected_seq_names = set(selected_seq_names)
    filtered_entries = [item for item in entries if item["seq_name"] in selected_seq_names]
    if args.max_samples > 0:
        filtered_entries = filtered_entries[:args.max_samples]

    results = {"meta": {"candidates_path": args.candidates_path, "tracker_name": args.tracker_name, "config": args.config, "window_size": args.window_size, "gain_th": args.gain_th, "split": args.split}, "samples": []}
    sft_records = []

    if args.resume and score_path.exists():
        with open(score_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        if sft_path.exists():
            with open(sft_path, "r", encoding="utf-8") as f:
                sft_records = [json.loads(l) for l in f if l.strip()]

    scored_keys = {make_sample_key(item["seq_name"], item["frame_id"]) for item in results.get("samples", [])}
    sft_keys = {make_sample_key(item["seq_name"], item["frame_id"]) for item in sft_records}

    tracker = make_tracker(args)
    print("[INFO] Tracker ready.")

    def flush_outputs(reason):
        with open(score_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        with open(sft_path, "w", encoding="utf-8") as f:
            for item in sft_records:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"[INFO] flushed {reason}: samples={len(results['samples'])} sft={len(sft_records)}")

    current_seq = None
    for idx, entry in enumerate(filtered_entries, 1):
        seq = seq_map.get(entry["seq_name"])
        if seq is None:
            continue
        sample_key = make_sample_key(entry["seq_name"], entry["frame_id"])
        if entry["seq_name"] != current_seq:
            if current_seq is not None:
                flush_outputs(f"seq_end:{current_seq}")
            current_seq = entry["seq_name"]
        if sample_key in scored_keys:
            continue

        scored = score_anchor(entry, seq, tracker, args)
        best = scored.get("best_candidate")
        if best is not None and float(best.get("gain", 0.0)) > args.gain_th:
            best["selected"] = True
            if sample_key not in sft_keys:
                sft_records.append(build_sft_record(entry, best["text"], best["gain"], args.system_prompt_version))
                sft_keys.add(sample_key)

        results["samples"].append(scored)
        scored_keys.add(sample_key)
        if idx % 10 == 0:
            print(f"[INFO] scored {idx}/{len(filtered_entries)}; sft={len(sft_records)}")

    flush_outputs("final")
    print(f"[DONE] scores -> {score_path}")
    print(f"[DONE] {len(sft_records)} SFT records -> {sft_path}")


if __name__ == "__main__":
    main()

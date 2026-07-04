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


SYSTEM_PROMPTS = {
    "v1": "You are a structured language refiner for visual tracking.",
    "v2": "You are refining structured tracking text for the current search region.",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score LaSOT Qwen refine candidates with tracker gain and export SFT dataset."
    )
    parser.add_argument("--candidates-path", required=True)
    parser.add_argument("--output-score-path", required=True)
    parser.add_argument("--output-sft-path", required=True)
    parser.add_argument("--tracker-name", default="lantrack", choices=["lantrack", "dutrack"])
    parser.add_argument("--config", required=True, help="Experiment yaml name without .yaml")
    parser.add_argument("--checkpoint", default="", help="Optional checkpoint override")
    parser.add_argument("--lasot-root", default="", help="Optional LaSOTBenchmark root override")
    parser.add_argument("--split", default="", choices=["", "train", "test"], help="Empty means infer from candidates meta")
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seq-start", type=int, default=0)
    parser.add_argument("--max-seqs", type=int, default=0)
    parser.add_argument("--max-candidates-per-anchor", type=int, default=4)
    parser.add_argument("--gain-th", type=float, default=0.003)
    parser.add_argument("--concept-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--system-prompt-version", default="v2", choices=sorted(SYSTEM_PROMPTS.keys()))
    return parser.parse_args()


def xywh_to_xyxy(box):
    x, y, w, h = [float(v) for v in box]
    return np.array([x, y, x + w, y + h], dtype=np.float64)


def compute_iou(box_a, box_b):
    a = xywh_to_xyxy(box_a)
    b = xywh_to_xyxy(box_b)
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def load_candidates(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "entries" not in data or not isinstance(data["entries"], dict):
        raise ValueError(f"Invalid candidates file: {path}")
    return data


def resolve_split(args, candidate_data):
    if args.split:
        return args.split
    meta = candidate_data.get("meta", {}) if isinstance(candidate_data, dict) else {}
    split = str(meta.get("split", "train")).strip().lower()
    return split if split in {"train", "test"} else "train"


def read_csv_boxes(path):
    boxes = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                boxes.append([float(x) for x in line.split(",")[:4]])
    return np.asarray(boxes, dtype=np.float64)


def load_train_sequence_list():
    split_file = REPO_ROOT / "lib" / "train" / "data_specs" / "lasot_train_split.txt"
    with open(split_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def build_seq_map(split, lasot_root):
    if not lasot_root:
        from lib.test.evaluation.environment import env_settings

        lasot_root = env_settings().lasot_path
    root = Path(lasot_root)
    if split == "test":
        from lib.test.evaluation.lasotdataset import LaSOTDataset

        seq_names = list(LaSOTDataset().sequence_list)
    else:
        seq_names = load_train_sequence_list()

    seq_map = {}
    for seq_name in seq_names:
        class_name = seq_name.split("-")[0]
        seq_path = root / class_name / seq_name
        gt_path = seq_path / "groundtruth.txt"
        frame_dir = seq_path / "img"
        if not gt_path.is_file() or not frame_dir.is_dir():
            continue
        ground_truth = read_csv_boxes(gt_path)
        frames = [str(frame_dir / f"{i + 1:08d}.jpg") for i in range(ground_truth.shape[0])]
        seq_map[seq_name] = {
            "name": seq_name,
            "frames": frames,
            "ground_truth_rect": ground_truth,
            "object_class": class_name,
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
    target = str(text_struct.get("target", "")).strip()
    concept = str(text_struct.get("concepts", "")).strip()
    background = str(text_struct.get("background", "")).strip()
    return "\n".join([
        f"Target: {target}",
        f"Concept: {concept}",
        f"Background: {background}",
        "",
        "Task:",
        "Update concept and background for the current search image.",
        "Return JSON only as:",
        '{"concept":"...", "background":"..."}',
    ])


def build_sft_record(entry, selected_text, gain, args):
    return {
        "seq_name": entry["seq_name"],
        "frame_id": entry["frame_id"],
        "image_path": entry["image_path"],
        "gain": float(gain),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPTS[args.system_prompt_version]},
            {
                "role": "user",
                "content": build_user_prompt({
                    "target": entry.get("target", ""),
                    "concepts": entry.get("concepts", ""),
                    "background": entry.get("background", ""),
                }),
            },
            {
                "role": "assistant",
                "content": json.dumps({
                    "concept": selected_text.get("concepts", ""),
                    "background": selected_text.get("background", ""),
                }, ensure_ascii=False),
            },
        ],
    }


def make_tracker(args):
    param_module = importlib.import_module(f"lib.test.parameter.{args.tracker_name}")
    params = param_module.parameters(args.config)
    if args.checkpoint:
        params.checkpoint = args.checkpoint
    params.debug = 0
    params.save_all_boxes = False
    tracker_module = importlib.import_module(f"lib.test.tracker.{args.tracker_name}")
    tracker_cls = tracker_module.get_tracker_class()
    return tracker_cls(params)


def run_tracking_window(tracker, seq, start_frame, text_struct, window_size):
    image = cv2.cvtColor(cv2.imread(seq["frames"][start_frame]), cv2.COLOR_BGR2RGB)
    init_bbox = list(seq["ground_truth_rect"][start_frame])
    init_info = {
        "init_bbox": init_bbox,
        "text_description": text_struct,
        "init_text_description": text_struct,
        "class": seq["object_class"],
        "path": seq["name"],
        "num": start_frame,
    }
    tracker.initialize(image, init_info)

    end_frame = min(len(seq["frames"]), start_frame + window_size + 1)
    pred_boxes = [init_bbox]
    gt_boxes = [list(seq["ground_truth_rect"][start_frame])]

    for frame_num in range(start_frame + 1, end_frame):
        image = cv2.cvtColor(cv2.imread(seq["frames"][frame_num]), cv2.COLOR_BGR2RGB)
        info = {
            "previous_output": {"target_bbox": pred_boxes[-1]},
            "class": seq["object_class"],
            "path": seq["name"],
            "num": frame_num,
            "text_description": text_struct,
        }
        out = tracker.track(image, info)
        pred_box = out.get("target_bbox", None)
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
    candidates = entry.get("candidates", [])[: args.max_candidates_per_anchor]
    for idx, candidate in enumerate(candidates):
        cand_text = build_candidate_text(entry, candidate, concept_only=args.concept_only)
        cand_score, cand_ious = run_tracking_window(tracker, seq, frame_id, cand_text, args.window_size)
        gain = float(cand_score - baseline_score)
        scored_candidates.append({
            "candidate_id": idx,
            "concept": str(candidate.get("concept", "")).strip(),
            "background": str(candidate.get("background", "")).strip(),
            "score": cand_score,
            "gain": gain,
            "ious": cand_ious,
            "text": cand_text,
            "raw_response": candidate.get("raw_response", ""),
        })

    best = max(scored_candidates, key=lambda x: x["gain"]) if scored_candidates else None
    return {
        "seq_name": entry["seq_name"],
        "frame_id": frame_id,
        "image_path": entry["image_path"],
        "baseline_text": baseline_text,
        "baseline_score": baseline_score,
        "baseline_ious": baseline_ious,
        "candidates": scored_candidates,
        "best_candidate": best,
        "best_gain": float(best["gain"]) if best is not None else None,
    }


def load_existing_outputs(score_path, sft_path):
    existing_results = None
    existing_sft_records = []
    if score_path.exists():
        with open(score_path, "r", encoding="utf-8") as f:
            existing_results = json.load(f)
    if sft_path.exists():
        with open(sft_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    existing_sft_records.append(json.loads(line))
    return existing_results, existing_sft_records


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
        raise SystemExit("Output exists. Use --overwrite to replace it.")
    score_path.parent.mkdir(parents=True, exist_ok=True)
    sft_path.parent.mkdir(parents=True, exist_ok=True)

    candidate_data = load_candidates(args.candidates_path)
    split = resolve_split(args, candidate_data)
    seq_map = build_seq_map(split=split, lasot_root=args.lasot_root)
    entries = list(candidate_data["entries"].values())

    selected_seq_names = sorted({item["seq_name"] for item in entries if item["seq_name"] in seq_map})
    if args.seq_start > 0:
        selected_seq_names = selected_seq_names[args.seq_start:]
    if args.max_seqs > 0:
        selected_seq_names = selected_seq_names[:args.max_seqs]
    selected_seq_names = set(selected_seq_names)

    filtered_entries = [item for item in entries if item["seq_name"] in selected_seq_names]
    if args.max_samples > 0:
        filtered_entries = filtered_entries[:args.max_samples]

    results = {
        "meta": {
            "candidates_path": args.candidates_path,
            "tracker_name": args.tracker_name,
            "config": args.config,
            "checkpoint": args.checkpoint,
            "window_size": args.window_size,
            "gain_th": args.gain_th,
            "concept_only": bool(args.concept_only),
            "num_entries": len(filtered_entries),
            "split": split,
            "lasot_root": args.lasot_root,
        },
        "samples": [],
    }
    sft_records = []

    if args.resume:
        existing_results, existing_sft_records = load_existing_outputs(score_path, sft_path)
        if existing_results is not None:
            results = existing_results
            meta = results.setdefault("meta", {})
            meta.update({
                "candidates_path": args.candidates_path,
                "tracker_name": args.tracker_name,
                "config": args.config,
                "checkpoint": args.checkpoint,
                "window_size": args.window_size,
                "gain_th": args.gain_th,
                "concept_only": bool(args.concept_only),
                "num_entries": len(filtered_entries),
                "split": split,
                "lasot_root": args.lasot_root,
            })
            results.setdefault("samples", [])
        sft_records = existing_sft_records

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
        print(
            f"[INFO] flushed {reason}: samples={len(results.get('samples', []))} "
            f"sft_records={len(sft_records)}"
        )

    current_seq_name = None
    for idx, entry in enumerate(filtered_entries, start=1):
        seq = seq_map.get(entry["seq_name"], None)
        if seq is None:
            continue
        sample_key = make_sample_key(entry["seq_name"], entry["frame_id"])
        if sample_key in scored_keys:
            if current_seq_name is None:
                current_seq_name = entry["seq_name"]
            elif entry["seq_name"] != current_seq_name:
                flush_outputs(f"resume_seq_end:{current_seq_name}")
                current_seq_name = entry["seq_name"]
            continue

        if current_seq_name is None:
            current_seq_name = entry["seq_name"]
        elif entry["seq_name"] != current_seq_name:
            flush_outputs(f"seq_end:{current_seq_name}")
            current_seq_name = entry["seq_name"]

        scored = score_anchor(entry, seq, tracker, args)
        best = scored.get("best_candidate", None)
        if best is not None and float(best.get("gain", 0.0)) > args.gain_th:
            best["selected"] = True
            if sample_key not in sft_keys:
                sft_records.append(build_sft_record(entry, best["text"], best["gain"], args))
                sft_keys.add(sample_key)

        results["samples"].append(scored)
        scored_keys.add(sample_key)
        if idx % 10 == 0:
            print(f"[INFO] scored {idx}/{len(filtered_entries)} entries; sft_records={len(sft_records)}")

    flush_outputs("final")
    print(f"[DONE] wrote scores to {score_path}")
    print(f"[DONE] wrote {len(sft_records)} SFT records to {sft_path}")


if __name__ == "__main__":
    main()

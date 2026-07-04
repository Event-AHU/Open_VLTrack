#!/usr/bin/env bash
set -euo pipefail

cd /rydata/jinliye/LanTracking

PYTHON_BIN=${PYTHON_BIN:-python}
GPU_ID=${GPU_ID:-4}
CANDIDATES_PATH=${CANDIDATES_PATH:-/rydata/jinliye/LanTracking/tools/qwen_refine_cache/tnllt_qwen_refine_candidates_train_full.json}
TRACKER_NAME=${TRACKER_NAME:-lantrack}
CONFIG_NAME=${CONFIG_NAME:-lantrack_256_tnllt_struct_onlyv2_token_reweight_concept_target_span_topk8}
WINDOW_SIZE=${WINDOW_SIZE:-20}
MAX_CANDIDATES=${MAX_CANDIDATES:-4}
GAIN_TH=${GAIN_TH:-0.02}
OUTPUT_DIR=${OUTPUT_DIR:-/rydata/jinliye/LanTracking/tools/qwen_refine_cache}
LOG_DIR=${LOG_DIR:-/rydata/jinliye/LanTracking/tools/qwen_refine_cache/logs}
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

run_part() {
  local part_id="$1"
  local seq_start="$2"
  local max_seqs="$3"
  local score_path="$OUTPUT_DIR/tnllt_tracking_gain_part${part_id}.json"
  local sft_path="$OUTPUT_DIR/tnllt_tracking_gain_sft_part${part_id}.jsonl"
  local log_path="$LOG_DIR/tnllt_tracking_gain_part${part_id}.log"

  echo "[RUN] part=${part_id} seq_start=${seq_start} max_seqs=${max_seqs} gpu=${GPU_ID}"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" tools/build_tracking_gain_sft_dataset.py     --candidates-path "$CANDIDATES_PATH"     --output-score-path "$score_path"     --output-sft-path "$sft_path"     --tracker-name "$TRACKER_NAME"     --config "$CONFIG_NAME"     --split train     --window-size "$WINDOW_SIZE"     --max-candidates-per-anchor "$MAX_CANDIDATES"     --gain-th "$GAIN_TH"     --concept-only     --seq-start "$seq_start"     --max-seqs "$max_seqs"     --resume     > "$log_path" 2>&1 &
  echo $! > "$LOG_DIR/tnllt_tracking_gain_part${part_id}.pid"
  echo "[PID] part=${part_id} pid=$(cat "$LOG_DIR/tnllt_tracking_gain_part${part_id}.pid") log=$log_path"
}

run_part 0 0 38
run_part 1 38 38
run_part 2 76 37
run_part 3 113 37

echo "[DONE] started 4 processes on GPU ${GPU_ID}"
echo "[INFO] logs: ${LOG_DIR}"
echo "[INFO] outputs: ${OUTPUT_DIR}"

#!/bin/bash
set -euo pipefail

# Batch runner for tools/generate_qwen_refine_cache.py
# Usage examples:
# 1) Run internal jobs (default detached background):
#    bash tools/run_generate_qwen_cache.sh
# 2) Single custom job:
#    bash tools/run_generate_qwen_cache.sh "gpus=4|split=test|exp=qwen_lora_xxx|lora=/path/to/adapter|resume=1"
# 3) Multiple custom jobs:
#    bash tools/run_generate_qwen_cache.sh \
#      "gpus=4|split=test|exp=run_a|lora=/path/a|resume=1" \
#      "gpus=5|split=test|exp=run_b|lora=/path/b|resume=1"
# 4) Wait for all jobs in current shell:
#    RUN_MODE=parallel_wait bash tools/run_generate_qwen_cache.sh
# 5) Force serial mode:
#    RUN_MODE=foreground bash tools/run_generate_qwen_cache.sh

REPO_ROOT="/rydata/jinliye/LanTracking"
PYTHON_BIN="/rydata/jinliye/condaenv/easyr12/bin/python"
SCRIPT_PATH="${REPO_ROOT}/tools/generate_qwen_refine_cache.py"
LOG_ROOT="${REPO_ROOT}/mylog/qwen_cache"

DEFAULT_MODEL_PATH="/rydata/jinliye/llmmodel/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/1b989f2c63999d7344135894d3cfa8f494116743"
DEFAULT_CACHE_ROOT="${REPO_ROOT}/output/test/qwencache"
DEFAULT_DATASET_NAME="TNLLT"
DEFAULT_SPLIT="test"
DEFAULT_GPU="4"
DEFAULT_DEVICE_PREFIX="cuda"
DEFAULT_DEVICE_INDEX="0"
DEFAULT_INTERVAL="50"
DEFAULT_OFFSET="49"
DEFAULT_PROMPT_VER="v2"
DEFAULT_USE_HINTS="1"
DEFAULT_RESUME="1"
DEFAULT_VISIBLE_ONLY="0"
DEFAULT_MAX_SEQS="0"
DEFAULT_MAX_ANCHORS="0"
DEFAULT_LIMIT_TOTAL="0"
DEFAULT_PARSED_TEST="${REPO_ROOT}/tools/tnllt_test_parsed_text_v2.json"
DEFAULT_PARSED_TRAIN="${REPO_ROOT}/tools/tnllt_train_parsed_text_v2.json"
DEFAULT_RUN_MODE="background_detach"

RUN_TAG=$(date +"%Y%m%d_%H%M%S")
BATCH_DIR="${LOG_ROOT}/${RUN_TAG}"
mkdir -p "${BATCH_DIR}"
PID_FILE="${BATCH_DIR}/batch_pids.txt"
: > "${PID_FILE}"

# Run mode:
# - background_detach (default): launch all jobs with nohup, return immediately.
# - parallel_wait: launch all jobs in parallel, wait all, aggregate status.
# - foreground: run jobs one by one and stream logs to terminal.
RUN_MODE="${RUN_MODE:-${DEFAULT_RUN_MODE}}"
if [[ "${RUN_MODE}" != "background_detach" && "${RUN_MODE}" != "parallel_wait" && "${RUN_MODE}" != "foreground" ]]; then
  echo "Invalid RUN_MODE=${RUN_MODE}, expected background_detach|parallel_wait|foreground" >&2
  exit 1
fi

# If USE_INTERNAL_JOBS=1 and no CLI args are provided, jobs below are used directly.
USE_INTERNAL_JOBS=1
  INTERNAL_JOBS=(
    "gpus=0|split=test|exp=qwen_lora_strict437_r16_lr3e5_e12_ckpt120|lora=/wangx_nas/JLY/Code/lantrack/LLMcheckpoint/qwen_lora_strict437_r16_lr3e5_e12/checkpoint-120|resume=1|use_hints=1"
  )
    # "gpus=4|split=test|exp=qwen_lora_strict437_r16_lr3e5_e12_ckpt60|lora=/wangx_nas/JLY/Code/lantrack/LLMcheckpoint/qwen_lora_strict437_r16_lr3e5_e12/checkpoint-60|resume=1|use_hints=1"
    # "gpus=5|split=test|exp=qwen_lora_strict437_r16_lr3e5_e12_ckpt90|lora=/wangx_nas/JLY/Code/lantrack/LLMcheckpoint/qwen_lora_strict437_r16_lr3e5_e12/checkpoint-90|resume=1|use_hints=1"
JOBS=()
if [[ "$#" -gt 0 ]]; then
  JOBS=("$@")
elif [[ "${USE_INTERNAL_JOBS}" == "1" ]]; then
  JOBS=("${INTERNAL_JOBS[@]}")
else
  echo "No jobs provided. Pass jobs as args or set USE_INTERNAL_JOBS=1 and edit INTERNAL_JOBS in script." >&2
  exit 1
fi

pids=()
pid_job_names=()
pid_log_dirs=()

for i in "${!JOBS[@]}"; do
  job="${JOBS[$i]}"

  gpus="${DEFAULT_GPU}"
  split="${DEFAULT_SPLIT}"
  exp=""
  lora=""
  model_path="${DEFAULT_MODEL_PATH}"
  cache_root="${DEFAULT_CACHE_ROOT}"
  dataset_name="${DEFAULT_DATASET_NAME}"
  interval="${DEFAULT_INTERVAL}"
  offset="${DEFAULT_OFFSET}"
  prompt_ver="${DEFAULT_PROMPT_VER}"
  use_hints="${DEFAULT_USE_HINTS}"
  resume="${DEFAULT_RESUME}"
  visible_only="${DEFAULT_VISIBLE_ONLY}"
  max_seqs="${DEFAULT_MAX_SEQS}"
  max_anchors="${DEFAULT_MAX_ANCHORS}"
  limit_total="${DEFAULT_LIMIT_TOTAL}"

  IFS='|' read -r -a parts <<< "${job}"
  for part in "${parts[@]}"; do
    key="${part%%=*}"
    val="${part#*=}"
    case "${key}" in
      gpus) gpus="${val}" ;;
      split) split="${val}" ;;
      exp) exp="${val}" ;;
      lora) lora="${val}" ;;
      model) model_path="${val}" ;;
      cache_root) cache_root="${val}" ;;
      dataset) dataset_name="${val}" ;;
      interval) interval="${val}" ;;
      offset) offset="${val}" ;;
      prompt) prompt_ver="${val}" ;;
      use_hints) use_hints="${val}" ;;
      resume) resume="${val}" ;;
      visible_only) visible_only="${val}" ;;
      max_seqs) max_seqs="${val}" ;;
      max_anchors) max_anchors="${val}" ;;
      limit_total) limit_total="${val}" ;;
      "") ;;
      *)
        echo "Unknown key in job: ${key}" >&2
        exit 1
        ;;
    esac
  done

  if [[ -z "${exp}" ]]; then
    echo "Missing required field: exp=... in job: ${job}" >&2
    exit 1
  fi

  if [[ "${split}" != "train" && "${split}" != "test" && "${split}" != "val" ]]; then
    echo "Invalid split=${split}, expected train|val|test" >&2
    exit 1
  fi

  if [[ "${split}" == "train" ]]; then
    parsed_text_path="${DEFAULT_PARSED_TRAIN}"
  else
    parsed_text_path="${DEFAULT_PARSED_TEST}"
  fi

  # With CUDA_VISIBLE_DEVICES, process-visible GPU index starts from 0.
  device="${DEFAULT_DEVICE_PREFIX}:${DEFAULT_DEVICE_INDEX}"

  job_name="${split}-${dataset_name}-${exp}"
  log_dir="${BATCH_DIR}/${job_name}"
  mkdir -p "${log_dir}"

  stdout_log="${log_dir}/stdout.log"
  stderr_log="${log_dir}/stderr.log"
  config_txt="${log_dir}/job_config.txt"

  cmd=("${PYTHON_BIN}" "${SCRIPT_PATH}"
    --model-path "${model_path}"
    --cache-root "${cache_root}"
    --dataset-name "${dataset_name}"
    --exp-name "${exp}"
    --split "${split}"
    --parsed-text-path "${parsed_text_path}"
    --device "${device}"
    --cache-interval "${interval}"
    --anchor-offset "${offset}"
    --system-prompt-version "${prompt_ver}"
  )

  if [[ -n "${lora}" ]]; then
    cmd+=(--lora-path "${lora}")
  fi
  if [[ "${use_hints}" == "1" ]]; then
    cmd+=(--use-hints)
  fi
  if [[ "${resume}" == "1" ]]; then
    cmd+=(--resume)
  fi
  if [[ "${visible_only}" == "1" ]]; then
    cmd+=(--visible-only)
  fi
  if [[ "${max_seqs}" != "0" ]]; then
    cmd+=(--max-seqs "${max_seqs}")
  fi
  if [[ "${max_anchors}" != "0" ]]; then
    cmd+=(--max-anchors-per-seq "${max_anchors}")
  fi
  if [[ "${limit_total}" != "0" ]]; then
    cmd+=(--limit-total "${limit_total}")
  fi

  {
    echo "job: ${job}"
    echo "gpus: ${gpus}"
    echo "split: ${split}"
    echo "exp: ${exp}"
    echo "lora: ${lora}"
    echo "model: ${model_path}"
    echo "cache_root: ${cache_root}"
    echo "dataset: ${dataset_name}"
    echo "parsed_text_path: ${parsed_text_path}"
    echo "device: ${device}"
    echo "interval: ${interval}"
    echo "offset: ${offset}"
    echo "prompt: ${prompt_ver}"
    echo "resume: ${resume}"
    echo "use_hints: ${use_hints}"
    echo "run_mode: ${RUN_MODE}"
    printf "command: "
    printf "%q " "${cmd[@]}"
    printf "\n"
  } > "${config_txt}"

  if [[ "${RUN_MODE}" == "foreground" ]]; then
    echo "Running cache job: ${job_name}"
    echo "  gpus: ${gpus}"
    echo "  log_dir: ${log_dir}"

    set +e
    CUDA_VISIBLE_DEVICES="${gpus}" "${cmd[@]}" \
      > >(tee -a "${stdout_log}") \
      2> >(tee -a "${stderr_log}" >&2)
    rc=$?
    set -e

    echo "${rc}" > "${log_dir}/exit_code.txt"
    if [[ "${rc}" -ne 0 ]]; then
      echo "Job failed: ${job_name}, exit_code=${rc}. See logs under ${log_dir}" >&2
      exit "${rc}"
    fi

    echo "Finished cache job: ${job_name}"
  else
    if [[ "${RUN_MODE}" == "background_detach" ]]; then
      CUDA_VISIBLE_DEVICES="${gpus}" nohup "${cmd[@]}" > "${stdout_log}" 2> "${stderr_log}" &
    else
      CUDA_VISIBLE_DEVICES="${gpus}" "${cmd[@]}" > "${stdout_log}" 2> "${stderr_log}" &
    fi
    pid=$!

    echo "${pid}" > "${log_dir}/pid.txt"
    echo "${job_name} ${pid}" >> "${PID_FILE}"

    pids+=("${pid}")
    pid_job_names+=("${job_name}")
    pid_log_dirs+=("${log_dir}")

    echo "Started cache job: ${job_name}"
    echo "  pid: ${pid}"
    echo "  gpus: ${gpus}"
    echo "  log_dir: ${log_dir}"
  fi
done

if [[ "${RUN_MODE}" == "parallel_wait" ]]; then
  failed=0
  for i in "${!pids[@]}"; do
    pid="${pids[$i]}"
    job_name="${pid_job_names[$i]}"
    log_dir="${pid_log_dirs[$i]}"

    if wait "${pid}"; then
      rc=0
      echo "${rc}" > "${log_dir}/exit_code.txt"
      echo "Finished cache job: ${job_name}"
    else
      rc=$?
      echo "${rc}" > "${log_dir}/exit_code.txt"
      echo "Job failed: ${job_name}, exit_code=${rc}. See logs under ${log_dir}" >&2
      failed=1
    fi
  done

  if [[ "${failed}" -ne 0 ]]; then
    echo "At least one parallel job failed. Batch logs: ${BATCH_DIR}" >&2
    exit 1
  fi
fi

echo "Batch logs: ${BATCH_DIR}"
if [[ "${RUN_MODE}" != "foreground" ]]; then
  echo "PID list: ${PID_FILE}"
fi

#!/bin/bash
# Usage:
#   bash tools/run_generate_qwen_cache_tnl2k.sh                  # run INTERNAL_JOBS
#   bash tools/run_generate_qwen_cache_tnl2k.sh "gpu=0|seq_start=0|max_seqs=178|exp=myrun|lora=/path/to/adapter|split=test"
#   RUN_MODE=foreground bash tools/run_generate_qwen_cache_tnl2k.sh

set -euo pipefail

REPO_ROOT="/rydata/jinliye/LanTracking"
PYTHON_BIN="/rydata/jinliye/condaenv/easyr12/bin/python"
SCRIPT_PATH="${REPO_ROOT}/tools/generate_qwen_refine_cache_tnl2k.py"
LOG_ROOT="${REPO_ROOT}/mylog/qwen_cache_tnl2k"

DEFAULT_MODEL_PATH="/rydata/jinliye/llmmodel/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/1b989f2c63999d7344135894d3cfa8f494116743"
DEFAULT_CACHE_DIR="${REPO_ROOT}/tools/qwen_refine_cache"
DEFAULT_PARSED_TEST="${REPO_ROOT}/tools/tnl2k_test_parsed_text_v1.json"
DEFAULT_PARSED_TRAIN="${REPO_ROOT}/tools/tnl2k_train_parsed_text_v1.json"
RUN_MODE="${RUN_MODE:-background_detach}"

# Edit INTERNAL_JOBS to define parallel shards
USE_INTERNAL_JOBS=1
# 下面这个还没跑 显存不够空了跑下
INTERNAL_JOBS=(
  "gpu=2|split=test|seq_start=0|max_seqs=178|exp=tnl2k_qwen_refine_test_mixed_r16_e5_part0|lora=/wangx_nas/JLY/Code/lantrack/LLMcheckpoint/mixed_all_r16_lr3e5_e5|overwrite=1"
)
  # "gpu=1|split=test|seq_start=178|max_seqs=178|exp=tnl2k_qwen_refine_test_mixed_r16_e5_part1|lora=/wangx_nas/JLY/Code/lantrack/LLMcheckpoint/mixed_all_r16_lr3e5_e5|overwrite=1"
  # "gpu=2|split=test|seq_start=356|max_seqs=178|exp=tnl2k_qwen_refine_test_mixed_r16_e5_part2|lora=/wangx_nas/JLY/Code/lantrack/LLMcheckpoint/mixed_all_r16_lr3e5_e5|overwrite=1"
  # "gpu=3|split=test|seq_start=534|max_seqs=175|exp=tnl2k_qwen_refine_test_mixed_r16_e5_part3|lora=/wangx_nas/JLY/Code/lantrack/LLMcheckpoint/mixed_all_r16_lr3e5_e5|overwrite=1"
  # "gpu=3|split=test|seq_start=0|max_seqs=178|exp=tnl2k_qwen_refine_test_r16_e5_part0|lora=/wangx_nas/JLY/Code/lantrack/LLMcheckpoint/tnl2k_r16_lr3e-5_e5|resume=1"
  # "gpu=4|split=test|seq_start=178|max_seqs=178|exp=tnl2k_qwen_refine_test_r16_e5_part1|lora=/wangx_nas/JLY/Code/lantrack/LLMcheckpoint/tnl2k_r16_lr3e-5_e5|resume=1"
  # "gpu=4|split=test|seq_start=356|max_seqs=178|exp=tnl2k_qwen_refine_test_r16_e5_part2|lora=/wangx_nas/JLY/Code/lantrack/LLMcheckpoint/tnl2k_r16_lr3e-5_e5|resume=1"
  # "gpu=7|split=test|seq_start=534|max_seqs=175|exp=tnl2k_qwen_refine_test_r16_e5_part3|lora=/wangx_nas/JLY/Code/lantrack/LLMcheckpoint/tnl2k_r16_lr3e-5_e5|resume=1"

JOBS=()
if [[ "$#" -gt 0 ]]; then
  JOBS=("$@")
elif [[ "${USE_INTERNAL_JOBS}" == "1" ]]; then
  JOBS=("${INTERNAL_JOBS[@]}")
else
  echo "No jobs provided." >&2; exit 1
fi

RUN_TAG=$(date +"%Y%m%d_%H%M%S")
BATCH_DIR="${LOG_ROOT}/${RUN_TAG}"
mkdir -p "${BATCH_DIR}"

pids=()
pid_job_names=()
pid_log_dirs=()

for job in "${JOBS[@]}"; do
  gpu="0"; split="test"; exp=""; lora=""
  model_path="${DEFAULT_MODEL_PATH}"; seq_start="0"; max_seqs="0"; resume="0"; overwrite="0"

  IFS='|' read -r -a parts <<< "${job}"
  for part in "${parts[@]}"; do
    key="${part%%=*}"; val="${part#*=}"
    case "${key}" in
      gpu) gpu="${val}" ;;
      split) split="${val}" ;;
      exp) exp="${val}" ;;
      lora) lora="${val}" ;;
      model) model_path="${val}" ;;
      seq_start) seq_start="${val}" ;;
      max_seqs) max_seqs="${val}" ;;
      resume) resume="${val}" ;;
      overwrite) overwrite="${val}" ;;
      "") ;;
      *) echo "Unknown key: ${key}" >&2; exit 1 ;;
    esac
  done

  [[ -z "${exp}" ]] && { echo "Missing exp= in job: ${job}" >&2; exit 1; }

  if [[ "${split}" == "train" ]]; then
    parsed_path="${DEFAULT_PARSED_TRAIN}"
  else
    parsed_path="${DEFAULT_PARSED_TEST}"
  fi

  output="${DEFAULT_CACHE_DIR}/${exp}.json"
  log_dir="${BATCH_DIR}/${exp}"
  mkdir -p "${log_dir}"

  cmd=("${PYTHON_BIN}" "${SCRIPT_PATH}"
    --model-path "${model_path}"
    --output "${output}"
    --split "${split}"
    --parsed-text-path "${parsed_path}"
    --device "cuda:0"
    --seq-start "${seq_start}"
  )
  [[ "${max_seqs}" != "0" ]] && cmd+=(--max-seqs "${max_seqs}")
  [[ -n "${lora}" ]] && cmd+=(--lora-path "${lora}")
  [[ "${resume}" == "1" ]] && cmd+=(--resume) || cmd+=(--overwrite)

  echo "Starting: ${exp} (gpu=${gpu})"
  echo "  output: ${output}"
  echo "  log: ${log_dir}/stdout.log"

  if [[ "${RUN_MODE}" == "foreground" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}" 2>&1 | tee "${log_dir}/stdout.log"
  else
    CUDA_VISIBLE_DEVICES="${gpu}" nohup "${cmd[@]}" \
      > "${log_dir}/stdout.log" 2> "${log_dir}/stderr.log" &
    pid=$!
    echo "${pid}" > "${log_dir}/pid.txt"
    pids+=("${pid}")
    pid_job_names+=("${exp}")
    pid_log_dirs+=("${log_dir}")
    echo "  pid: ${pid}"
  fi
done

if [[ "${RUN_MODE}" == "parallel_wait" ]]; then
  failed=0
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}" && echo "Done: ${pid_job_names[$i]}" || { echo "Failed: ${pid_job_names[$i]}" >&2; failed=1; }
  done
  [[ "${failed}" -ne 0 ]] && exit 1
fi

echo "Batch logs: ${BATCH_DIR}"

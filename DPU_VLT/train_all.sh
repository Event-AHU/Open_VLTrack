#!/bin/bash
set -euo pipefail

LOG_ROOT="/rydata/jinliye/LanTracking/mylog/train"
BASE_PORT=29500
DEFAULT_SAVE_DIR="./output"
DEFAULT_MODE="multiple"
DEFAULT_USE_WANDB=0

JOBS=(
  "gpus=1,3|script=lantrack|config=lantrack_256_all_struct_onlyv2_token_reweight_concept_target_span_topk8_ep50|nproc=2|port=29500"
)

if [[ "$#" -gt 0 ]]; then
  JOBS=("$@")
fi

export HF_ENDPOINT="https://hf-mirror.com"

RUN_TAG=$(date +"%Y%m%d_%H%M%S")
BATCH_DIR="${LOG_ROOT}/${RUN_TAG}"
mkdir -p "${BATCH_DIR}"

batch_pid_file="${BATCH_DIR}/batch_pids.txt"
: > "${batch_pid_file}"

for i in "${!JOBS[@]}"; do
  job="${JOBS[$i]}"
  gpus=""
  script=""
  config=""
  nproc=""
  port=""
  save_dir="${DEFAULT_SAVE_DIR}"
  mode="${DEFAULT_MODE}"
  use_wandb="${DEFAULT_USE_WANDB}"

  IFS='|' read -r -a parts <<< "${job}"
  for part in "${parts[@]}"; do
    key="${part%%=*}"
    val="${part#*=}"
    case "${key}" in
      gpus) gpus="${val}" ;;
      script) script="${val}" ;;
      config) config="${val}" ;;
      nproc) nproc="${val}" ;;
      port) port="${val}" ;;
      save_dir) save_dir="${val}" ;;
      mode) mode="${val}" ;;
      use_wandb) use_wandb="${val}" ;;
      "") ;;
      *)
        echo "Unknown key: ${key}" >&2
        exit 1
        ;;
    esac
  done

  if [[ -z "${gpus}" || -z "${script}" || -z "${config}" ]]; then
    echo "Missing required fields in job: ${job}" >&2
    echo "Required: gpus=...|script=...|config=..." >&2
    exit 1
  fi

  if [[ -z "${nproc}" ]]; then
    IFS=',' read -r -a gpu_arr <<< "${gpus}"
    nproc="${#gpu_arr[@]}"
  fi

  if [[ -z "${port}" ]]; then
    port=$((BASE_PORT + i))
  fi

  log_dir="${BATCH_DIR}/${script}-${config}"
  mkdir -p "${log_dir}"

  stdout_log="${log_dir}/train_stdout.log"
  stderr_log="${log_dir}/train_stderr.log"
  config_file="${log_dir}/experiment_config.txt"

  cmd=(torchrun --nproc_per_node="${nproc}" --master_port="${port}" tracking/train.py
      --script "${script}" --config "${config}" --save_dir "${save_dir}" --mode "${mode}" --use_wandb "${use_wandb}")

  {
    echo "script: ${script}"
    echo "config: ${config}"
    echo "save_dir: ${save_dir}"
    echo "mode: ${mode}"
    echo "gpus: ${gpus}"
    echo "nproc_per_node: ${nproc}"
    echo "master_port: ${port}"
    echo "use_wandb: ${use_wandb}"
    printf "command: "
    printf "%q " "${cmd[@]}"
    printf "\n"
  } > "${config_file}"

  CUDA_VISIBLE_DEVICES="${gpus}" nohup "${cmd[@]}" > "${stdout_log}" 2> "${stderr_log}" &
  pid=$!
  echo "${pid}" > "${log_dir}/pid.txt"
  echo "${script} ${config} ${pid}" >> "${batch_pid_file}"

  echo "Started: ${script} ${config}"
  echo "  log_dir: ${log_dir}"
  echo "  pid: ${pid}"
done

echo "Batch logs: ${BATCH_DIR}"
echo "PID list: ${batch_pid_file}"

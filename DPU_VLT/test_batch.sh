#!/bin/bash
set -euo pipefail

LOG_ROOT="/rydata/jinliye/LanTracking/mylog/test"
DEFAULT_TRACKER_NAME="lantrack"
DEFAULT_THREADS=1
DEFAULT_NUM_GPUS=1
DEFAULT_DEBUG=0

JOBS=(
  "gpus=2|tracker_name=lantrack|tracker_param=lantrack_256_all_struct_onlyv2_token_reweight_concept_target_span_topk8_ep45_lasot_qwen_refine_targetfilter|dataset=lasot|threads=4|num_gpus=1|runid=45"
)

if [[ "$#" -gt 0 ]]; then
  JOBS=("$@")
fi

RUN_TAG=$(date +"%Y%m%d_%H%M%S")
BATCH_DIR="${LOG_ROOT}/${RUN_TAG}"
mkdir -p "${BATCH_DIR}"

batch_pid_file="${BATCH_DIR}/batch_pids.txt"
: > "${batch_pid_file}"

for i in "${!JOBS[@]}"; do
  job="${JOBS[$i]}"
  gpus=""
  tracker_name="${DEFAULT_TRACKER_NAME}"
  tracker_param=""
  dataset=""
  threads="${DEFAULT_THREADS}"
  num_gpus="${DEFAULT_NUM_GPUS}"
  runid=""
  sequence=""
  debug="${DEFAULT_DEBUG}"

  IFS='|' read -r -a parts <<< "${job}"
  for part in "${parts[@]}"; do
    key="${part%%=*}"
    val="${part#*=}"
    case "${key}" in
      gpus) gpus="${val}" ;;
      tracker_name) tracker_name="${val}" ;;
      tracker_param|config) tracker_param="${val}" ;;
      dataset|dataset_name) dataset="${val}" ;;
      threads) threads="${val}" ;;
      num_gpus) num_gpus="${val}" ;;
      runid) runid="${val}" ;;
      sequence) sequence="${val}" ;;
      debug) debug="${val}" ;;
      "") ;;
      *)
        echo "Unknown key: ${key}" >&2
        exit 1
        ;;
    esac
  done

  if [[ -z "${gpus}" || -z "${tracker_param}" || -z "${dataset}" ]]; then
    echo "Missing required fields in job: ${job}" >&2
    echo "Required: gpus=...|tracker_param=...|dataset=..." >&2
    exit 1
  fi

  log_dir="${BATCH_DIR}/${tracker_name}-${tracker_param}-${dataset}"
  mkdir -p "${log_dir}"

  stdout_log="${log_dir}/test_stdout.log"
  stderr_log="${log_dir}/test_stderr.log"
  config_file="${log_dir}/experiment_config.txt"

  cmd=(python tracking/test.py
      --tracker_name "${tracker_name}"
      --tracker_param "${tracker_param}"
      --dataset_name "${dataset}"
      --threads "${threads}"
      --num_gpus "${num_gpus}"
      --debug "${debug}")
  if [[ -n "${runid}" ]]; then
    cmd+=(--runid "${runid}")
  fi
  if [[ -n "${sequence}" ]]; then
    cmd+=(--sequence "${sequence}")
  fi

  {
    echo "tracker_name: ${tracker_name}"
    echo "tracker_param: ${tracker_param}"
    echo "dataset_name: ${dataset}"
    echo "threads: ${threads}"
    echo "num_gpus: ${num_gpus}"
    echo "debug: ${debug}"
    echo "runid: ${runid}"
    echo "sequence: ${sequence}"
    echo "gpus: ${gpus}"
    printf "command: "
    printf "%q " "${cmd[@]}"
    printf "\n"
  } > "${config_file}"

  CUDA_VISIBLE_DEVICES="${gpus}" nohup "${cmd[@]}" > "${stdout_log}" 2> "${stderr_log}" &
  pid=$!
  echo "${pid}" > "${log_dir}/pid.txt"
  echo "${tracker_name} ${tracker_param} ${dataset} ${pid}" >> "${batch_pid_file}"

  echo "Started: ${tracker_name} ${tracker_param} ${dataset}"
  echo "  log_dir: ${log_dir}"
  echo "  pid: ${pid}"
done

echo "Batch logs: ${BATCH_DIR}"
echo "PID list: ${batch_pid_file}"

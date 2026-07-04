#!/bin/bash
set -euo pipefail

GPU_IDS="0"
CACHE_DIR="/rydata/jinliye/treeTrack/Dependency Parser/dataset"
SAVE_DIR="/rydata/jinliye/LanTracking/DPModel2/ckpt"
GLOVE_PATH="/rydata/jinliye/LanTracking/DPModel2/glove.6B.300d.txt"
GLOVE_DIM=300
BATCH_SIZE=64
EPOCHS=300
LR=5e-4
DEVICE="cuda:0"
HIDDEN_DIM=256
LSTM_LAYERS=3
FFNN_DIM=256
SAVE_EVERY=30
MIN_LR=1e-5
PATIENCE=3
FACTOR=0.5
LOG_DIR="/rydata/jinliye/LanTracking/DPModel2/log"

mkdir -p "${LOG_DIR}" "${SAVE_DIR}"
TS=$(date +"%Y%m%d_%H%M%S")
STDOUT_LOG="${LOG_DIR}/train_${TS}.out"
STDERR_LOG="${LOG_DIR}/train_${TS}.err"

CUDA_VISIBLE_DEVICES="${GPU_IDS}" nohup python /rydata/jinliye/LanTracking/DPModel2/train.py \
  --cache_dir "${CACHE_DIR}" \
  --save_dir "${SAVE_DIR}" \
  --glove_path "${GLOVE_PATH}" \
  --glove_dim "${GLOVE_DIM}" \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --device "${DEVICE}" \
  --hidden_dim "${HIDDEN_DIM}" \
  --lstm_layers "${LSTM_LAYERS}" \
  --ffnn_dim "${FFNN_DIM}" \
  --save_every "${SAVE_EVERY}" \
  --min_lr "${MIN_LR}" \
  --patience "${PATIENCE}" \
  --factor "${FACTOR}" \
  > "${STDOUT_LOG}" 2> "${STDERR_LOG}" &

echo "Started DPModel2 training"
echo "GPU_IDS: ${GPU_IDS}"
echo "STDOUT: ${STDOUT_LOG}"
echo "STDERR: ${STDERR_LOG}"
echo "PID: $!"

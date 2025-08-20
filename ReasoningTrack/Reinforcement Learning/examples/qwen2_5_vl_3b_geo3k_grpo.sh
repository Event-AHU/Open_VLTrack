#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH="/rydata/wengchaoliu/qwen2.5vl-3b/"  # replace it with your local file path

CUDA_VISIBLE_DEVICES=4 python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=hiyouga/geometry3k@train \
    data.val_files=hiyouga/geometry3k@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    worker.actor.optim.strategy=adamw_bf16 \
    trainer.experiment_name=qwen2_5_vl_3b_geo_grpo \
    trainer.n_gpus_per_node=1 \
    # trainer.logger=['console']\ 
    # > /rydata/jinliye/RL/vltracking/EasyR1/log/training.log 2>&1

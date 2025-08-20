#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH="Qwen/Qwen2.5-0.5B-Instruct"  # replace it with your local file path

CUDA_VISIBLE_DEVICES=4 python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=xiaodongguaAIGC/X-R1-750@train \
    data.val_files=xiaodongguaAIGC/X-R1-750@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    worker.actor.optim.strategy=adamw_bf16\
    trainer.experiment_name=qwen2_5_vl_3b_geo_grpo \
    trainer.n_gpus_per_node=1\
    # trainer.logger=['console']\ 
    # > /rydata/jinliye/RL/vltracking/EasyR1/log/training.log 2>&1

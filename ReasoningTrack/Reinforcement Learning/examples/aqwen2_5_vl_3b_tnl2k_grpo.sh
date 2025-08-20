#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH="/rydata/jinliye/RL/vltracking/EasyR1/LongTimeTracking/LLaMA-Factory/saves/Qwen2.5-VL-3B-Instruct/full/train_2025-05-09-13-12-15/checkpoint-250"  # replace it with your local file path
# MODEL_PATH="/rydata/wengchaoliu/qwen2.5vl-3b/"
CUDA_VISIBLE_DEVICES=2 nohup python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=Jinliye/TNL2KLTRLDataset2@train \
    data.val_files=Jinliye/TNL2KLTRLDataset2@test \
    data.rollout_batch_size=256 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    worker.actor.optim.strategy=adamw_bf16\
    worker.rollout.limit_images=2 \
    trainer.experiment_name=qwen2_5_vl_3b_tnllt_grpo \
    trainer.n_gpus_per_node=1 \
    trainer.logger=['console','tensorboard']\ 
    
    # trainer.save_checkpoint_path= "/rydata/jinliye/RL/vltracking/EasyR1/checkpoint/TNLLT" \
    # data.format_prompt= ./examples/format_prompt/LTracking_format.jinja \
    # trainer.total_epochs = 1 \
    # > /rydata/jinliye/RL/vltracking/EasyR1/log/training.log 2>&1

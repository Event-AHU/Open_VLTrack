import os
import argparse
import random
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Parse args for training')
    # for train
    parser.add_argument('--script', type=str, help='training script name')
    parser.add_argument('--config', type=str, default='baseline', help='yaml configure file name')
    parser.add_argument('--save_dir', type=str, help='root directory to save checkpoints, logs, and tensorboard')
    parser.add_argument('--mode', type=str, choices=["single", "multiple", "multi_node"], default="multiple",
                        help="train on single gpu or multiple gpus")
    parser.add_argument('--nproc_per_node', type=int,default=2, help="number of GPUs per node")  # specify when mode is multiple
    parser.add_argument('--use_lmdb', type=int, choices=[0, 1], default=0)  # whether datasets are in lmdb format
    parser.add_argument('--script_prv', type=str, help='training script name')
    parser.add_argument('--config_prv', type=str, default='baseline', help='yaml configure file name')
    parser.add_argument('--use_wandb', type=int, choices=[0, 1], default=0)  # whether to use wandb
    # for knowledge distillation
    parser.add_argument('--distill', type=int, choices=[0, 1], default=0)  # whether to use knowledge distillation
    parser.add_argument('--script_teacher', type=str, help='teacher script name')
    parser.add_argument('--config_teacher', type=str, help='teacher yaml configure file name')

    # for multiple machines
    parser.add_argument('--rank', type=int, help='Rank of the current process.')
    parser.add_argument('--world-size', type=int, help='Number of processes participating in the job.')
    parser.add_argument('--ip', type=str, default='127.0.0.1', help='IP of the current rank 0.')
    parser.add_argument('--port', type=int, default='20000', help='Port of the current rank 0.')
    parser.add_argument("--local-rank", type=int, default=-1, help="Local rank for distributed training")  # 必须添加
    args = parser.parse_args()

    return args

import torch
def main():
    args = parse_args()
    # 3. 关键：从环境变量读取 LOCAL_RANK（torchrun 自动注入，无需命令行传递）
    local_rank = os.environ.get("LOCAL_RANK")
    local_rank = int(os.environ.get("LOCAL_RANK", -1)) 
    # 验证是否获取到（可选，用于调试）
    print(f"train.py 读取到的 LOCAL_RANK: {local_rank}", flush=True)
    # 1. 根据 local_rank 绑定 GPU（核心步骤）
    if args.local_rank != -1:
        # 绑定当前进程到 args.local_rank 对应的 GPU
        torch.cuda.set_device(args.local_rank)
        # 初始化分布式进程组（使用 NCCL 后端，GPU 训练推荐）
        torch.distributed.init_process_group(
            backend="nccl",  # GPU 分布式训练优先用 NCCL
            init_method="env://"  # 从环境变量读取 MASTER_ADDR/MASTER_PORT（torchrun 自动设置）
        )

    # 验证：打印当前进程绑定的 GPU
    print(f"Local rank: {args.local_rank}, 绑定的 GPU: {torch.cuda.current_device()}", flush=True)
    if args.mode == "single":
        train_cmd = "python lib/train/run_training.py --script %s --config %s --save_dir %s --use_lmdb %d " \
                    "--script_prv %s --config_prv %s --distill %d --script_teacher %s --config_teacher %s --use_wandb %d"\
                    % (args.script, args.config, args.save_dir, args.use_lmdb, args.script_prv, args.config_prv,
                       args.distill, args.script_teacher, args.config_teacher, args.use_wandb)
        print(train_cmd)
    elif args.mode == "multiple":
        # train_cmd = "python -m torch.distributed.launch --nproc_per_node %d --master_port %d lib/train/run_training.py " \
        #             "--script %s --config %s --save_dir %s --use_lmdb %d --script_prv %s --config_prv %s --use_wandb %d " \
        #             "--distill %d --script_teacher %s --config_teacher %s" \
        #             % (args.nproc_per_node, random.randint(10000, 50000), args.script, args.config, args.save_dir, args.use_lmdb, args.script_prv, args.config_prv, args.use_wandb,
        #                args.distill, args.script_teacher, args.config_teacher)
        train_cmd = "python lib/train/run_training.py " \
            "--script %s --config %s --save_dir %s --use_lmdb %d --script_prv %s --config_prv %s --use_wandb %d " \
            "--distill %d --script_teacher %s --config_teacher %s --local_rank %d" \
            % ( args.script, args.config, args.save_dir, args.use_lmdb, args.script_prv, args.config_prv, args.use_wandb,
                args.distill, args.script_teacher, args.config_teacher, local_rank)
    elif args.mode == "multi_node":
        train_cmd = "python -m torch.distributed.launch --nproc_per_node %d --master_addr %s --master_port %d --nnodes %d --node_rank %d lib/train/run_training.py " \
                    "--script %s --config %s --save_dir %s --use_lmdb %d --script_prv %s --config_prv %s --use_wandb %d " \
                    "--distill %d --script_teacher %s --config_teacher %s" \
                    % (args.nproc_per_node, args.ip, args.port, args.world_size, args.rank, args.script, args.config, args.save_dir, args.use_lmdb, args.script_prv, args.config_prv, args.use_wandb,
                       args.distill, args.script_teacher, args.config_teacher)
    else:
        raise ValueError("mode should be 'single' or 'multiple'.")
    os.system(train_cmd)


if __name__ == "__main__":
    main()

#python tracking/train.py --script odtrack --config baseline --save_dir ./output --mode multiple --nproc_per_node 1 --use_wandb 0

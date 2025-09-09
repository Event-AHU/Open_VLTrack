CUDA_VISIBLE_DEVICES=2 vllm serve /wangx_nas/JLY/Code/LongTimeTracking/RLModels/easyr1/TNLLT_ioubf16/global_step_90/actor/huggingface \
  --port 8000 \
  --host 0.0.0.0 \
  --dtype bfloat16 \
  --limit-mm-per-prompt image=5,video=5 \
  --gpu-memory-utilization 0.8

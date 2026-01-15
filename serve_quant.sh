#!/bin/bash
source env.sh

LARGE_GPU_MEM=0.80
MAX_MODEL_LEN=32768 # 写代码不太可能要 32k 以上

export MAIN_MODEL="cyankiwi/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit"
vllm serve $MAIN_MODEL --port 8000 --max_model_len $MAX_MODEL_LEN --gpu-memory-utilization $LARGE_GPU_MEM --logits-processors logits_processors.vllm_adapters.watermark_adapter:WatermarkVLLMAdapter
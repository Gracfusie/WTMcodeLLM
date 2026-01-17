#!/bin/bash
source env.sh

LARGE_GPU_MEM=0.80
MAX_MODEL_LEN=32768 # 写代码不太可能要 32k 以上

export VLLM_USE_MODELSCOPE=0
export MAIN_MODEL="NVFP4/Qwen3-Coder-30B-A3B-Instruct-FP4"

echo "使用自己的代理.."
unset http_proxt https_proxy
proxychains4 -q vllm serve $MAIN_MODEL --port 8000 --max_model_len $MAX_MODEL_LEN --gpu-memory-utilization $LARGE_GPU_MEM --logits-processors logits_processors.vllm_adapters.watermark_adapter:WatermarkVLLMAdapter
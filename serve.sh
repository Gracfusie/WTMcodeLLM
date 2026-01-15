#!/bin/bash
source env.sh

LARGE_GPU_MEM=0.70

# 每个 Large Model Batch 也需要跑 Small Model

sleep 1

vllm serve $MAIN_MODEL --port 8000 --gpu-memory-utilization $LARGE_GPU_MEM --logits-processors logits_processors.vllm_adapters.watermark_adapter:WatermarkVLLMAdapter
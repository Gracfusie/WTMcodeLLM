#!/usr/bin/env bash

source env.sh

export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
export OPENAI_KEY="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"
export OPENAI_SUPPORT_N=0

cd third_party/LiveCodeBench

python -m lcb_runner.runner.main \
  --model "qwen/qwen3-coder-30b-a3b-instruct" \
  --scenario codegeneration \
  --evaluate \
  --release_version v6 \
  --n 3 \
  --multiprocess 10 \
  --temperature 0.7 \
  --max_tokens 16384 \
  --openai_timeout 7200 \
  --eval_limit 50

echo "完成。输出位于 output/<model_repr>/codegeneration_*.json 及评测文件。"


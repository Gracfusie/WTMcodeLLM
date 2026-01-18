#!/usr/bin/env bash

source env.sh

export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}"
export OPENAI_KEY="${OPENAI_KEY:-EMPTY}"

cd third_party/LiveCodeBench

python -m lcb_runner.runner.main \
  --model "NVFP4/Qwen3-Coder-30B-A3B-Instruct-FP4" \
  --scenario codegeneration \
  --evaluate \
  --release_version v6 \
  --n 3 \
  --multiprocess 10 \
  --temperature 0.7 \
  --max_tokens 16384 \
  --openai_timeout 14400 \
  --eval_limit 50 

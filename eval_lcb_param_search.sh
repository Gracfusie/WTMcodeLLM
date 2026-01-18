#!/usr/bin/env bash

source env.sh

export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}"
export OPENAI_KEY="${OPENAI_KEY:-EMPTY}"

cd third_party/LiveCodeBench

param_grid=(
  "0.5 2.0"
  "0.5 2.0"
  "0.5 2.0"
  "1.0 2.0"
  "2.0 2.0"
  "0.5 4.0"
  "0.5 1.0"
)

for pair in "${param_grid[@]}"; do
  read -r entropy delta <<< "${pair}"

  export WATERMARK_ENTROPY_THRESHOLD="${entropy}"
  export WATERMARK_DELTA="${delta}"

  name_entropy="${entropy//./p}"
  name_delta="${delta//./p}"
  custom_name="test2-entropy${name_entropy}_delta${name_delta}"

  echo "运行参数搜索: entropy=${entropy}, delta=${delta}, name=${custom_name}"

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
    --eval_limit 50 \
    --custom_output_save_name "${custom_name}"
done
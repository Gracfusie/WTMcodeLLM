#!/usr/bin/env bash

# 一键评测：通过本地/自建 OpenAI 兼容推理接口（需支持 reasoning_effort）

source /etc/network_turbo
source /root/autodl-tmp/.venv/bin/activate

# 确保权重和缓存都落在大盘（与 interact.sh 保持一致）
export HF_HOME=/root/autodl-tmp/hf
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/hf
export VLLM_WORKDIR=/root/autodl-tmp/vllm_workdir
export HF_HUB_DISABLE_XET=1
export TIKTOKEN_RS_CACHE_DIR=/root/autodl-tmp/tiktoken_rs_cache
export HF_DATASETS_ALLOW_CODE=1            # 允许加载带脚本的数据集

# OpenAI 兼容端点与密钥
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}"
export OPENAI_KEY="${OPENAI_KEY:-EMPTY}"

MODEL_NAME="${MODEL_NAME:-openai/gpt-oss-20b__high}"   # 带 __high/medium/low 传 reasoning_effort
RELEASE_VERSION="${RELEASE_VERSION:-release_latest}"
STOP_TOKEN="${STOP_TOKEN:-}"                           # 与 interact 一致，默认空

cd /root/autodl-tmp/WTMcodeLLM/third_party/LiveCodeBench

python -m lcb_runner.runner.main \
  --model "${MODEL_NAME}" \
  --scenario codegeneration \
  --evaluate \
  --release_version "${RELEASE_VERSION}" \
  --n 1 \
  --multiprocess 8 \
  --temperature 1.0 \
  --top_p 1.0 \
  --max_tokens 32768 \
  --multiprocess 0 \
  --stop "${STOP_TOKEN}" \
  "$@"

echo "完成。输出位于 output/<model_repr>/codegeneration_*.json 及评测文件。"


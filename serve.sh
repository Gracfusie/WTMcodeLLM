source /etc/network_turbo
source /root/autodl-tmp/.venv/bin/activate

# 确保权重和缓存都落在大盘
export HF_HOME=/root/autodl-tmp/hf
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/hf
export VLLM_WORKDIR=/root/autodl-tmp/vllm_workdir   # 可选，编译/缓存目录
export HF_HUB_DISABLE_XET=1  # 临时解决 bug
export TIKTOKEN_ENCODINGS_BASE=/root/autodl-tmp/tiktoken_rs_cache
export PYTHONPATH=/root/autodl-tmp/WTMcodeLLM:${PYTHONPATH}
python -c 'from openai_harmony import load_harmony_encoding; load_harmony_encoding("HarmonyGptOss")'
# 启动 gpt-oss-20b（首次会下载到上面的目录）
# 通过 FQCN 加载 toy 调试 logits processor，打印前缀与 top-k logits 信息
vllm serve "openai/gpt-oss-20b" --async-scheduling \
  --logits-processors logits_processors.vllm_adapters.watermark_adapter:WatermarkVLLMAdapter
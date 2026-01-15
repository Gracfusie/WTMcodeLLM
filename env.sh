source /etc/network_turbo
source /root/autodl-tmp/WTMcodeLLM/.venv/bin/activate

# 确保权重和缓存都落在大盘
export HF_HOME=/root/autodl-tmp/hf
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/hf
export VLLM_USE_MODELSCOPE=1
export MODELSCOPE_CACHE=/root/autodl-tmp/modelscope_cache
export VLLM_WORKDIR=/root/autodl-tmp/vllm_workdir   # 可选，编译/缓存目录
export HF_HUB_DISABLE_XET=1  # 临时解决 bug
export TIKTOKEN_ENCODINGS_BASE=/root/autodl-tmp/tiktoken_rs_cache
export PYTHONPATH=/root/autodl-tmp/WTMcodeLLM:${PYTHONPATH}

# 确保 Redis 服务正在运行
# if ! pgrep -x "redis-server" > /dev/null; then
#     echo "Redis 服务未运行,正在启动..."
#     redis-server --daemonize yes
#     if [ $? -eq 0 ]; then
#         echo "Redis 服务已成功启动"
#     else
#         echo "警告: Redis 服务启动失败"
#     fi
# else
#     echo "Redis 服务已在运行中"
# fi


export MAIN_MODEL="Qwen/Qwen2.5-Coder-1.5B-Instruct"
echo 现在 main model 是小的。之后 Qwen/Qwen3-Coder-30B-A3B-Instruct 之后记得改！

export WATERMARK_PROXY_MODEL="Qwen/Qwen2.5-Coder-1.5B-Instruct"
export WATERMARK_PROXY_TEMPLATE_PREFIX="<|fim_prefix|>"
export WATERMARK_PROXY_TEMPLATE_SUFFIX="<|fim_suffix|>\n<|fim_middle|>"

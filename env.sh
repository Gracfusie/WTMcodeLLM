# AutoDL-specific environment variables
# if [ -z "$http_proxy" ] && [ -z "$https_proxy" ] && [ -z "$HTTP_PROXY" ] && [ -z "$HTTPS_PROXY" ]; then
#     source /etc/network_turbo
# fi


# export HF_HOME=/root/autodl-tmp/hf
# export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/hf
# export VLLM_USE_MODELSCOPE=true
# export MODELSCOPE_CACHE=/root/autodl-tmp/modelscope_cache
# export VLLM_WORKDIR=/root/autodl-tmp/vllm_workdir
# export TIKTOKEN_ENCODINGS_BASE=/root/autodl-tmp/tiktoken_rs_cache

source .venv/bin/activate
export WATERMARK_PROXY_MODEL="Qwen/Qwen2.5-Coder-1.5B-Instruct"
export WATERMARK_PROXY_TEMPLATE_PREFIX="<|fim_prefix|>"
export WATERMARK_PROXY_TEMPLATE_SUFFIX="<|fim_suffix|>\n<|fim_middle|>"

export WATERMARK_ENTROPY_THRESHOLD="0.5"
export WATERMARK_DELTA="2.0"
export WATERMARK_SECRET_KEY="42"

# Context Window For Proxy Model
export WATERMARK_PROXY_WINDOW_SIZE="256"

source /etc/network_turbo
source .venv/bin/activate

# 确保权重和缓存都落在大盘
export HF_HOME=/root/autodl-tmp/hf
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/hf
export VLLM_USE_MODELSCOPE=1
export MODELSCOPE_CACHE=/root/autodl-tmp/modelscope_cache
export VLLM_WORKDIR=/root/autodl-tmp/vllm_workdir   # 可选，编译/缓存目录
export HF_HUB_DISABLE_XET=1  # 临时解决 bug
export TIKTOKEN_ENCODINGS_BASE=/root/autodl-tmp/tiktoken_rs_cache
export PYTHONPATH=$(pwd):${PYTHONPATH}

export WATERMARK_PROXY_MODEL="Qwen/Qwen2.5-Coder-1.5B-Instruct"
export WATERMARK_PROXY_TEMPLATE_PREFIX="<|fim_prefix|>"
export WATERMARK_PROXY_TEMPLATE_SUFFIX="<|fim_suffix|>\n<|fim_middle|>"

# 可能要调的参数

# ACW 水印参数
export WATERMARK_ENTROPY_THRESHOLD="0.5"
export WATERMARK_DELTA="2.0"
export WATERMARK_SECRET_KEY="42"

# Proxy 模型的 Window Size. -1 表示不使用 Window
# 启动的话可能会让水印更 robust, i.e. 不容易靠截断/移动代码片段攻击
export WATERMARK_PROXY_WINDOW_SIZE="-1"
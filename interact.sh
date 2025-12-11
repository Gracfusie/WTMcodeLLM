source /etc/network_turbo
source /root/autodl-tmp/.venv/bin/activate

# 确保权重和缓存都落在大盘
export HF_HOME=/root/autodl-tmp/hf
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/hf
export VLLM_WORKDIR=/root/autodl-tmp/vllm_workdir   # 可选，编译/缓存目录
export HF_HUB_DISABLE_XET=1  # 临时解决 bug
export TIKTOKEN_RS_CACHE_DIR=/root/autodl-tmp/tiktoken_rs_cache

python interact.py
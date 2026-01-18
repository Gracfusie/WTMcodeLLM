uv venv .venv
uv pip install vllm --torch-backend=auto
uv pip install modelscope accelerate
cd third_party/LiveCodeBench
uv pip install -e .
cd ../../
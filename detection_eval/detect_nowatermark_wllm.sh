export MAIN_MODEL="NVFP4/Qwen3-Coder-30B-A3B-Instruct-FP4"
export Z_THRESHOLD=4.0

export WATERMARK_ENTROPY_THRESHOLD="-1.0"
export WATERMARK_GAMMA="0.5"

python detector.py --method wllm --z-threshold $Z_THRESHOLD --input-dir nowatermark-input/ --output-dir nowatermark-wllm-output/
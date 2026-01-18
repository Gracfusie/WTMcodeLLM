export MAIN_MODEL="NVFP4/Qwen3-Coder-30B-A3B-Instruct-FP4"
export Z_THRESHOLD=4.0

python detector.py --z-threshold $Z_THRESHOLD --input-dir proxy-attack-trunc-input/ --output-dir proxy-attack-trunc-zscores/
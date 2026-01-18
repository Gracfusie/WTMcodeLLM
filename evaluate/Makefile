# Makefile to run evaluate/runner.py with hardcoded parameter lists

PY ?= python
RUNNER := runner_new.py

# Watermark algorithm to include in output filenames and pass-through (default: proxy)
# 现在仅用于标记输出文件名、传递参数，因此在不跑proxy模型的时候要注意 make grid WATERMARK_ALGORITHM=...
WATERMARK_ALGORITHM ?= proxy

# Core runtime controls (override via environment if needed)
# Provide sensible defaults; user can override via environment.
MODEL ?= None
ENDPOINT ?= http://127.0.0.1:8000/v1
API_KEY ?= EMPTY
INPUT_DIR ?= input
OUTPUT_DIR ?= output
ATTACK ?= none
REASONING ?= high
CONCURRENCY ?= 64

# Hardcoded parameter lists (entropy, delta, window) - this is for grid runs of PROXY
# USAGE: 1,2,3,4,...
PARAM_ENT := 0.25,0.5,1.0
PARAM_DEL := 1,2,3,10
PARAM_WIN := 128,256

PARAM_ENT_B := 0.5
PARAM_DEL_B := 2
PARAM_WIN_B := 256

# Midput -> output conversion (format_adapter)
MIDPUT_INPUT ?= midput/Scenario.codegeneration_1_0.7_eval_all.json
CONVERT_OUTPUT_DIR ?= output/converted
CONVERT_BASE_STEM ?= scenario
CONVERT_LIMIT ?=

.PHONY: all grid vary default debug convert clean

all: grid

# Grid run over hardcoded lists
grid:
	WATERMARK_ALGORITHM=$(WATERMARK_ALGORITHM) $(PY) $(RUNNER) \
	  --endpoint $(ENDPOINT) \
	  --api-key $(API_KEY) \
	  --reasoning-effort $(REASONING) \
	  --input-dir $(INPUT_DIR) \
	  --output-dir $(OUTPUT_DIR) \
	  --concurrency $(CONCURRENCY) \
	  --entropy-list $(PARAM_ENT) \
	  --delta-list $(PARAM_DEL) \
	  --window-list $(PARAM_WIN)

# Partial sweeps: vary one parameter at a time, others fixed to baseline (PARAM_*_B)
vary:
	# Vary entropy only
	WATERMARK_ALGORITHM=$(WATERMARK_ALGORITHM) $(PY) $(RUNNER) \
	  --endpoint $(ENDPOINT) \
	  --api-key $(API_KEY) \
	  --reasoning-effort $(REASONING) \
	  --input-dir $(INPUT_DIR) \
	  --output-dir $(OUTPUT_DIR) \
	  --concurrency $(CONCURRENCY) \
	  --entropy-list $(PARAM_ENT) \
	  --delta-list $(PARAM_DEL_B) \
	  --window-list $(PARAM_WIN_B)
	# Vary delta only
	WATERMARK_ALGORITHM=$(WATERMARK_ALGORITHM) $(PY) $(RUNNER) \
	  --endpoint $(ENDPOINT) \
	  --api-key $(API_KEY) \
	  --reasoning-effort $(REASONING) \
	  --input-dir $(INPUT_DIR) \
	  --output-dir $(OUTPUT_DIR) \
	  --concurrency $(CONCURRENCY) \
	  --entropy-list $(PARAM_ENT_B) \
	  --delta-list $(PARAM_DEL) \
	  --window-list $(PARAM_WIN_B)
	# Vary window only
	WATERMARK_ALGORITHM=$(WATERMARK_ALGORITHM) $(PY) $(RUNNER) \
	  --endpoint $(ENDPOINT) \
	  --api-key $(API_KEY) \
	  --reasoning-effort $(REASONING) \
	  --input-dir $(INPUT_DIR) \
	  --output-dir $(OUTPUT_DIR) \
	  --concurrency $(CONCURRENCY) \
	  --entropy-list $(PARAM_ENT_B) \
	  --delta-list $(PARAM_DEL_B) \
	  --window-list $(PARAM_WIN)

# Small built-in sweep when no explicit lists are provided
default:
	WATERMARK_ALGORITHM=$(WATERMARK_ALGORITHM) $(PY) $(RUNNER) \
	  --endpoint $(ENDPOINT) \
	  --api-key $(API_KEY) \
	  --reasoning-effort $(REASONING) \
	  --input-dir $(INPUT_DIR) \
	  --output-dir $(OUTPUT_DIR) \
	  --concurrency $(CONCURRENCY) \
	  --grid-default

# Debug: single prompt + single param set (sampling + cap)
debug:
	WATERMARK_ALGORITHM=$(WATERMARK_ALGORITHM) $(PY) $(RUNNER) \
	  --endpoint $(ENDPOINT) \
	  --api-key $(API_KEY) \
	  --reasoning-effort $(REASONING) \
	  --input-dir $(INPUT_DIR) \
	  --output-dir $(OUTPUT_DIR) \
	  --concurrency 1 \
	  --sample-count 1 \
	  --sample-seed 0 \
	  --max-param-sets 1

# Convert midput JSON to analyze_outputs-compatible per-sample JSONs
convert:
	$(PY) format_adapter.py \
	  --input $(MIDPUT_INPUT) \
	  --output-dir $(CONVERT_OUTPUT_DIR) \
	  --base-stem $(CONVERT_BASE_STEM) \
	  $(if $(CONVERT_LIMIT),--limit $(CONVERT_LIMIT),)

# Clean outputs
clean:
	rm -f $(OUTPUT_DIR)/*Z.json

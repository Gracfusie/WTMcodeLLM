# Makefile to run evaluate/runner.py with hardcoded parameter lists

PY ?= python
RUNNER := runner.py

# Core runtime controls (override via environment if needed)
# Provide sensible defaults; user can override via environment.
MODEL ?= None
ENDPOINT ?= http://127.0.0.1:8000/v1
API_KEY ?= EMPTY
INPUT_DIR ?= input
OUTPUT_DIR ?= output
ATTACK ?= none
REASONING ?= high

# Hardcoded parameter lists (entropy, delta, window) - this is for grid runs of PROXY
PARAM_ENT := 2.0,3.0
PARAM_DEL := 0.0,0.2
PARAM_WIN := 128,256,512

.PHONY: all grid default clean

all: grid

# Grid run over hardcoded lists
grid:
	$(PY) $(RUNNER) \
	  --model $(MODEL) \
	  --endpoint $(ENDPOINT) \
	  --api-key $(API_KEY) \
	  --reasoning-effort $(REASONING) \
	  --attack $(ATTACK) \
	  --input-dir $(INPUT_DIR) \
	  --output-dir $(OUTPUT_DIR) \
	  --entropy-list $(PARAM_ENT) \
	  --delta-list $(PARAM_DEL) \
	  --window-list $(PARAM_WIN)

# Small built-in sweep when no explicit lists are provided
default:
	$(PY) $(RUNNER) \
	  --model $(MODEL) \
	  --endpoint $(ENDPOINT) \
	  --api-key $(API_KEY) \
	  --reasoning-effort $(REASONING) \
	  --attack $(ATTACK) \
	  --input-dir $(INPUT_DIR) \
	  --output-dir $(OUTPUT_DIR) \
	  --grid-default

# Clean outputs
clean:
	rm -f $(OUTPUT_DIR)/*Z.json

#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"

# Deployment configuration. Paths are relative to the project directory.
MODEL_DIR="$PROJECT_DIR/checkpoint"
HOTWORD_FILE="" # Optional: set a project-relative vocabulary path.
MODE=ctc_rag
GPU_ID=0
HOST=127.0.0.1
PORT=8900
MAX_HOTWORDS=100
RAG_WORKERS=8
GPU_MEMORY_UTILIZATION=0.08
KV_CACHE_MIB=1024
MAX_MODEL_LEN=1024
MAX_NEW_TOKENS=96
MAX_BATCH_SIZE=8
MAX_BATCH_WAIT_MS=20

if [[ -n "$HOTWORD_FILE" && "$HOTWORD_FILE" != /* ]]; then
  HOTWORD_FILE="$PROJECT_DIR/$HOTWORD_FILE"
fi

for file in config.json preprocessor_config.json tokenizer.json; do
  if [[ ! -f "$MODEL_DIR/$file" ]]; then
    printf 'Model file not found: %s\n' "$MODEL_DIR/$file" >&2
    printf 'Place the complete CTC checkpoint in %s or edit MODEL_DIR in scripts/serve.sh.\n' "$MODEL_DIR" >&2
    exit 1
  fi
done
if [[ ! -f "$MODEL_DIR/model.safetensors" && ! -f "$MODEL_DIR/model.safetensors.index.json" ]]; then
  printf 'Model weights or shard index not found in %s.\n' "$MODEL_DIR" >&2
  exit 1
fi
if [[ -n "$HOTWORD_FILE" && ! -f "$HOTWORD_FILE" ]]; then
  printf 'Hotword file not found: %s\n' "$HOTWORD_FILE" >&2
  exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
  printf 'uv is not installed. See https://docs.astral.sh/uv/getting-started/installation/\n' >&2
  exit 1
fi
if [[ ! -x "$PROJECT_DIR/.venv-uv/bin/python" ]]; then
  printf 'Environment not found. Run: UV_PROJECT_ENVIRONMENT=.venv-uv uv sync --locked --extra vllm --extra server\n' >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export VLLM_USE_FLASHINFER_SAMPLER=0
export HF_HUB_OFFLINE=1
export UV_PROJECT_ENVIRONMENT="$PROJECT_DIR/.venv-uv"

cd "$PROJECT_DIR"
hotword_args=()
if [[ -n "$HOTWORD_FILE" ]]; then
  hotword_args=(--hot-txt "$HOTWORD_FILE")
fi
exec uv run --no-sync python -m qwen3_asr_vllm.server \
  --model "$MODEL_DIR" \
  --mode "$MODE" "${hotword_args[@]}" \
  --host "$HOST" --port "$PORT" \
  --max-ctx-hotwords "$MAX_HOTWORDS" --rag-workers "$RAG_WORKERS" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" --kv-cache-mib "$KV_CACHE_MIB" \
  --max-model-len "$MAX_MODEL_LEN" --max-new-tokens "$MAX_NEW_TOKENS" \
  --max-batch-size "$MAX_BATCH_SIZE" --max-batch-wait-ms "$MAX_BATCH_WAIT_MS"

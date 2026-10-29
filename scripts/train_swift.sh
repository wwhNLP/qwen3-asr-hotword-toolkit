#!/usr/bin/env bash
set -euo pipefail

# 默认路径相对于项目目录，也可以通过环境变量覆盖。
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/artifacts/aishell}"
SFT_DIR="${PROJECT_DIR}/artifacts/sft-0.6b"
STAGE=ctc
if (($#)) && [[ "$1" != -* ]]; then
  STAGE="${1:-sft}"
  shift
fi

export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

# 默认使用物理 GPU 3、6；外部设置可覆盖，进程数自动跟随卡数。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3,6}"
IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-${#GPU_IDS[@]}}"
export MASTER_PORT="${MASTER_PORT:-29531}"
if [[ ! "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]] || ((NPROC_PER_NODE > ${#GPU_IDS[@]})); then
  printf 'NPROC_PER_NODE must be between 1 and the number of selected GPUs.\n' >&2
  exit 1
fi

# CTC 阶段默认使用步数最大的 SFT 检查点。
latest_sft_checkpoint() {
  local checkpoint
  for checkpoint in "${SFT_DIR}"/checkpoint-*; do
    [[ "${checkpoint}" =~ /checkpoint-[0-9]+$ ]] || continue
    [[ -f "${checkpoint}/config.json" ]] || continue
    [[ -f "${checkpoint}/model.safetensors" || -f "${checkpoint}/model.safetensors.index.json" ]] || continue
    printf '%s\n' "${checkpoint}"
  done | sort -V | tail -n 1
}

# 第一阶段：热词 SFT；第二阶段：冻结主模型，只训练 CTC。
case "${STAGE}" in
  sft)
    MODEL="${MODEL_PATH:-${PROJECT_DIR}/model/Qwen3-ASR-0.6B}"
    OUTPUT="${OUTPUT_DIR:-${SFT_DIR}}"
    STAGE_ARGS=(--learning_rate 2e-6 --num_train_epochs 2)
    ;;
  ctc)
    MODEL="${MODEL_PATH:-$(latest_sft_checkpoint)}"
    if [[ -z "${MODEL}" ]]; then
      printf 'No SFT checkpoint with config and weights found under %s; run sft first or set MODEL_PATH.\n' "${SFT_DIR}" >&2
      exit 1
    fi
    printf 'Using SFT checkpoint: %s\n' "${MODEL}" >&2
    OUTPUT="${OUTPUT_DIR:-${PROJECT_DIR}/artifacts/ctc-0.6b}"
    STAGE_ARGS=(
      --freeze_parameters thinker --trainable_parameters thinker.ctc_branch
      --freeze_llm true --gradient_checkpointing false
      --learning_rate 1e-4 --num_train_epochs 5
    )
    ;;
  *)
    printf 'Unknown stage: %s (use sft or ctc)\n' "${STAGE}" >&2
    exit 2
    ;;
esac

# 两卡时：每卡 batch 8 × 2 卡 × 累积 8 步 = 有效 batch 128。
# 脚本末尾传入的参数可以覆盖以下默认值。
exec swift sft \
  --model "${MODEL}" \
  --output_dir "${OUTPUT}" \
  --dataset "${DATA_DIR}/train_${STAGE}.jsonl" \
  --val_dataset "${DATA_DIR}/dev_${STAGE}.jsonl" \
  --external_plugins "${PROJECT_DIR}/training/swift_plugin.py" \
  --model_type "qwen3_asr_context_${STAGE}" \
  --template "qwen3_asr_context_${STAGE}" \
  --tuner_type full \
  --use_hf true \
  --torch_dtype float32 \
  --bf16 true \
  --fp16 false \
  --attn_impl sdpa \
  --strict true \
  --label_names labels \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 8 \
  --max_length 2048 \
  --dataloader_num_workers 0 \
  --eval_steps 500 \
  --save_total_limit 2 \
  --logging_steps 10 \
  --warmup_ratio 0.05 \
  --weight_decay 0.01 \
  --report_to none \
  --add_version false \
  "${STAGE_ARGS[@]}" "$@"

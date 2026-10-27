#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
stage="${1:?Usage: bash scripts/train_swift.sh sft|ctc [additional swift arguments]}"
shift
export PYTHONPATH="${repo_dir}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
data_dir="${DATA_DIR:-${repo_dir}/artifacts/aishell}"
options=(
  --external_plugins "${repo_dir}/training/swift_plugin.py"
  --model_type "qwen3_asr_context_${stage}"
  --template "qwen3_asr_context_${stage}"
  --tuner_type full --use_hf true --torch_dtype float32 --bf16 true --fp16 false --attn_impl sdpa
  --dataset "${data_dir}/train_${stage}.jsonl"
  --val_dataset "${data_dir}/dev_${stage}.jsonl" --split_dataset_ratio 0
  --strict true
  --per_device_train_batch_size 1 --per_device_eval_batch_size 1
  --gradient_accumulation_steps 16 --max_length 2048
  --padding_free false --packing false --loss_scale default
  --label_names labels
  --dataset_num_proc 1 --dataloader_num_workers 0
  --eval_strategy steps --eval_steps 500 --save_steps 500 --save_total_limit 2
  --logging_steps 10 --warmup_ratio 0.05 --weight_decay 0.01
  --report_to none --seed 42 --data_seed 42 --add_version false
)
case "${stage}" in
  sft)
    options+=(
      --model "${MODEL_PATH:-Qwen/Qwen3-ASR-0.6B}"
      --output_dir "${OUTPUT_DIR:-${repo_dir}/artifacts/sft-0.6b}"
      --freeze_vit true --freeze_aligner true --freeze_llm false
      --gradient_checkpointing true --learning_rate 2e-6 --num_train_epochs 2
    )
    ;;
  ctc)
    options+=(
      --model "${MODEL_PATH:?Set MODEL_PATH to the completed stage-one SFT checkpoint}"
      --output_dir "${OUTPUT_DIR:-${repo_dir}/artifacts/ctc-0.6b}"
      --freeze_parameters thinker --trainable_parameters thinker.ctc_branch
      --freeze_vit true --freeze_aligner true --freeze_llm true
      --gradient_checkpointing false --learning_rate 1e-4 --num_train_epochs 5
    )
    ;;
  *) printf 'Unknown stage: %s\n' "${stage}" >&2; exit 2 ;;
esac
exec swift sft "${options[@]}" "$@"

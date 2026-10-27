"""Validate and export an ms-swift full checkpoint for the vLLM adapter."""

import argparse
from pathlib import Path

import torch

from qwen3_asr_model.configuration_qwen3_asr import Qwen3ASRConfig
from qwen3_asr_model.modeling_qwen3_asr import Qwen3ASRForConditionalGeneration
from qwen3_asr_model.processing_qwen3_asr import Qwen3ASRProcessor
from utils.paraformer_vocab import ParaformerVocab


def export(checkpoint: str, output: str, require_ctc: bool = True):
    source, destination = Path(checkpoint).resolve(), Path(output).resolve()
    if source == destination or (destination.exists() and any(destination.iterdir())):
        raise ValueError("Export to an empty directory different from the training checkpoint.")
    if (source / "adapter_config.json").exists():
        raise ValueError("Merge LoRA adapters into full weights before exporting.")
    config = Qwen3ASRConfig.from_pretrained(source)
    if require_ctc and not config.thinker_config.add_ctc_branch:
        raise ValueError("ctc_rag requires a checkpoint after CTC training.")
    if config.thinker_config.add_ctc_branch:
        vocab = ParaformerVocab()
        if (config.thinker_config.ctc_vocab_size, config.thinker_config.ctc_blank_id) != (vocab.vocab_size, vocab.blank_id):
            raise ValueError("Checkpoint CTC vocabulary differs from the deployment vocabulary.")
    model, info = Qwen3ASRForConditionalGeneration.from_pretrained(
        source, config=config, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", output_loading_info=True,
    )
    failures = {key: value for key, value in info.items() if value}
    if failures:
        raise ValueError(f"Checkpoint did not load exactly; refusing random/missing weights: {failures}")
    model.config.thinker_config.text_config.use_cache = True
    processor = Qwen3ASRProcessor.from_pretrained(source)
    model.save_pretrained(destination, safe_serialization=True)
    processor.save_pretrained(destination)
    print(f"Exported complete checkpoint: {destination}")
    print("CTC enabled:", bool(model.config.thinker_config.add_ctc_branch))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--asr-only", action="store_true", help="Allow a checkpoint without CTC.")
    args = parser.parse_args()
    export(args.checkpoint, args.output, require_ctc=not args.asr_only)


if __name__ == "__main__":
    main()

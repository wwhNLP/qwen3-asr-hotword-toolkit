---
license: Apache-2.0
---

# Qwen3-ASR 0.6B CTC Hotword

A Chinese automatic speech recognition research checkpoint for experiments with external hotword vocabularies. It combines hotword-aware fine-tuning of the Qwen3-ASR-0.6B variant with a trained CTC branch. The CTC branch produces a rough transcript; the companion application uses phoneme-based retrieval to select relevant words from a supplied vocabulary and passes them to the ASR decoder.

- Project and inference code: [Qwen3-ASR Hotword Toolkit](https://github.com/wwhNLP/qwen3-asr-hotword-toolkit)
- Original context-ASR project: [Qwen3-ASR Context Deploy](https://github.com/Xiaoxx18/qwen3_asr_context_deploy)
- Base model: [Qwen/Qwen3-ASR-0.6B](https://modelscope.ai/models/Qwen/Qwen3-ASR-0.6B)
- Training data: AISHELL-1 training split; the hotword prompts were constructed from training data.

## Checkpoint

This is the merged checkpoint after two training stages: hotword-prompt SFT (step 1,878) and CTC branch training (step 4,695, five epochs). It includes the ASR model and a five-layer CTC branch; the CTC vocabulary has 8,404 tokens. The repository contains inference weights and processor/tokenizer files, not optimizer states or training audio.

The companion code supports vLLM 0.23.0 with a custom model adapter. CTC+RAG requires that application and an external hotword file; using stock `vllm serve` alone does not run the hotword retrieval pipeline. Refer to the GitHub project for installation, audio segmentation, online serving, batching, and vocabulary updates.

```bash
git clone --branch dev https://github.com/wwhNLP/qwen3-asr-hotword-toolkit.git
cd qwen3-asr-hotword-toolkit
modelscope download --model wwhnlp/Qwen3-ASR-0.6B-CTC-Hotword --local_dir ./checkpoint
```

The model is intended for Chinese ASR and domain vocabulary experiments. Corpus-level CER and hotword accuracy have not yet been established; a successful inference or latency benchmark is not evidence of an accuracy improvement. Test with representative audio, vocabulary and deployment conditions before relying on the output.

This checkpoint and the Qwen base model are published under Apache License 2.0; see `LICENSE`. See the linked code repositories and training-data sources for their respective terms.

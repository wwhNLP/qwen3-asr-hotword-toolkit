<div align="center">

# 🎙️ Qwen3-ASR Hotword Toolkit

**Hotword-aware speech recognition with CTC, phoneme retrieval, and vLLM**

[Background](#-background) · [Quick start](#-quick-start) · [Training](#-training) · [Tests](#-tests) · [Original repository](https://github.com/Xiaoxx18/qwen3_asr_context_deploy)

</div>

## 💡 Background

Qwen3-ASR can struggle with names and domain-specific terms, and providing hotwords in the prompt alone may not reliably improve their recognition. This project explores hotword-aware fine-tuning and retrieval to make better use of an external vocabulary.

The original [Qwen3-ASR Context Deploy](https://github.com/Xiaoxx18/qwen3_asr_context_deploy) combines a hotword-fine-tuned model with a CTC branch and phoneme-based retrieval: it creates a rough transcript, selects relevant terms from a hotword list, and supplies them to Qwen3-ASR for final transcription.

This repository builds on that work with:

- A **vLLM backend** for the CTC, retrieval, and transcription pipeline, reusing the same audio encoding for CTC and final decoding.
- A **two-stage ms-swift workflow** to fine-tune Qwen3-ASR-0.6B for hotword prompts, train a CTC branch, and export the checkpoint for vLLM.
- **Robustness fixes and tests** for training, inference, and hotword retrieval, plus JSONL batch evaluation with corpus CER.

The vLLM pipeline has been tested with the original author's 1.7B checkpoint. The 0.6B training-to-deployment path has only been smoke-tested for two steps per stage; full training and recognition quality remain unverified.

```mermaid
flowchart LR
    A[Audio] --> E[Audio encoder]
    E --> C[CTC rough transcript]
    C --> R[Phoneme retrieval]
    H[Hotword list] --> R
    R --> P[Hotword prompt]
    E --> D[Qwen3-ASR decoder]
    P --> D
    D --> T[Transcript]
```

## ✨ Features

- **Three modes:** `asr` for transcription, `ctc` for a rough transcript, and `ctc_rag` for retrieval-assisted transcription.
- **Shared audio encoding:** the vLLM worker reuses encoded features for CTC and final decoding, avoiding a second encoder pass.
- **External hotwords:** phoneme-based retrieval selects relevant terms from a UTF-8 text file, one term per line.
- **Training and evaluation:** prepare AISHELL-1 data, fine-tune a 0.6B model with ms-swift, train a CTC branch, and evaluate JSONL batches with corpus CER.

## 🚀 Quick start

Use a separate Python 3.10+ GPU environment. The vLLM adapter targets **vLLM 0.22.1** and **Transformers 4.57.6**; install CUDA-compatible vLLM and PyTorch builds for your system.

```bash
git clone https://github.com/wwhNLP/qwen3-asr-hotword-toolkit.git
cd qwen3-asr-hotword-toolkit
python -m pip install -e '.[vllm]'
```

Download a checkpoint with a trained CTC branch, such as the original author's [1.7B hotword model](https://huggingface.co/habagou/qwen3-asr-hotword-1.7B-ctc):

```bash
hf download habagou/qwen3-asr-hotword-1.7B-ctc \
  --local-dir ./ckpt/qwen3-asr-hotword-1.7B-ctc

CUDA_VISIBLE_DEVICES=0 python infer_vllm.py \
  --model ./ckpt/qwen3-asr-hotword-1.7B-ctc \
  --audio /path/to/audio.wav \
  --mode ctc_rag --hot-txt /path/to/hotwords.txt \
  --verify-encoder-reuse
```

`ctc` and `ctc_rag` require a checkpoint with a CTC branch. The included `hotword/hot.txt` is an example; replace it with a list relevant to your audio. `--verify-encoder-reuse` checks that each audio segment is encoded once.

For a JSONL batch, each record needs exactly one audio path; an optional assistant message supplies the reference for CER:

```json
{"uttid":"example","audios":["/path/to/audio.wav"],"messages":[{"role":"assistant","content":"reference transcript"}]}
```

```bash
CUDA_VISIBLE_DEVICES=0 python infer_vllm.py \
  --model ./ckpt/qwen3-asr-hotword-1.7B-ctc \
  --test-json /path/to/test.jsonl --mode ctc_rag \
  --hot-txt /path/to/hotwords.txt --batch-size 8 --out-json results.jsonl
```

The original Transformers entry point is `infer_ctc.py`; configure its hotword settings in the script and set `MODEL_PATH`, `MODE`, and optionally `TEST_JSON` via environment variables.

Both inference entry points reject output paths that alias the input manifest or any input audio, including symbolic and hard links. In `infer_ctc.py`, each batch result has a `status` (`ok` or `error`); failures include the exception type and message in `error` and use `hyp: null`. CER counts only successful records with a reference transcript. Missing references and failed records have `err: null` and `ref_len: null`; an explicitly empty reference still counts insertions. The console reports failed and scored sample counts separately. When there are no scored reference characters, CER is `N/A` and `run_file` returns `None`.

## 🏋️ Training

The [ms-swift + AISHELL-1 guide](docs/ms_swift_aishell.md) (Chinese) covers data preparation, hotword-aware SFT, CTC-only training, checkpoint export, and vLLM inference. It starts from the official Qwen3-ASR-0.6B base model. Use AISHELL-1 **train** for training and reserve the SeACo dev/test subsets for evaluation.

The 0.6B training → export → vLLM pipeline was initially smoke-tested for **two steps per training stage**. A completed local AISHELL-1 checkpoint was later used for [latency measurements](docs/vllm_optimization.md); corpus-level recognition quality has not yet been evaluated. The [upstream README](https://github.com/Xiaoxx18/qwen3_asr_context_deploy#readme) reports the original author's recognition benchmarks.

## 🧪 Tests

```bash
python -m unittest discover -s tests -v
```

GPU integration tests require real weights and audio, and run when configured in a vLLM environment:

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0 \
  VLLM_TEST_MODEL=/path/to/ctc-checkpoint VLLM_TEST_AUDIO=/path/to/audio.wav \
  python -m unittest discover -s tests -p test_vllm_integration.py -v
```

## 📌 Current limits

The vLLM adapter supports a **single GPU** and an **unquantized, merged checkpoint**. TP/PP, mixed multimodal requests, and loading LoRA adapters directly are not supported. Audio segments default to a maximum of 30 seconds; split longer recordings or adjust `--max-audio-seconds`. Use `infer_vllm.py` for the complete hotword pipeline; `vllm serve` alone does not run its context preparation.

Audio encoding and CTC run in microbatches (`--prepare-batch-size 4`). For larger batches, CPU hotword retrieval overlaps preparation of the next microbatch. Features stay on the worker GPU by default; only IDs and lengths cross the process boundary. A 256 MiB request cache is released after generation, including on errors. `--embedding-transport cpu` provides the original feature-transfer path for comparison. `--batch-size` controls the number of utterances submitted together for final decoding.

Hotword coarse scoring uses up to four native CPU workers (`--rag-workers 4`), including for single audio. Use `--rag-workers 1` for serial scoring. Matching scores and candidate order are preserved; each score matrix is limited to 16 MiB.

See the [latency measurements and vLLM 0.26 migration assessment](docs/vllm_optimization.md) for validation, limits, and a repeatable benchmark. The supported inference version remains **vLLM 0.22.1**.

## 🙏 Credits and license

The original model implementation, retrieval approach, and reported experiments come from [Xiaoxx18/qwen3_asr_context_deploy](https://github.com/Xiaoxx18/qwen3_asr_context_deploy). This project also builds on [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR), [SeACo-Paraformer](https://github.com/R1ckShi/SeACo-Paraformer), [asr-hotword](https://github.com/HaujetZhao/asr-hotword), and [FunASR](https://github.com/modelscope/FunASR).

The upstream repository currently has no declared license. Confirm the relevant permissions before using or redistributing its code; dependencies, datasets, and model weights have their own terms.

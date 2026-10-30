# vLLM 0.23.0: setup and CTC+RAG inference

Use a separate Python 3.10+ environment. The project offers `.[vllm]` for
vLLM 0.22.1 and `.[vllm023]` for 0.23.0; install one of these alternatives.
Both pin Transformers 4.57.6. vLLM 0.23 still accepts Transformers 4, although
it logs a deprecation warning. Releases beyond 0.23.0 are not enabled here.

Select a vLLM wheel matching your platform and CUDA stack. For example, on
Linux x86-64, the official CUDA 12.9 wheel can be installed before the extra:

```bash
python -m pip install \
  'https://github.com/vllm-project/vllm/releases/download/v0.23.0/vllm-0.23.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl' \
  --extra-index-url https://download.pytorch.org/whl/cu129
python -m pip install -e '.[vllm023]'
```

This is an installation recipe, not a validation of every CUDA and driver
combination. The `vllm023` extra pins cuDNN frontend to 1.19.1 to avoid
newer frontend dependencies conflicting with the 0.23.0 CUDA requirements.

The PyPI 0.23 wheel encountered during testing links CUDA 13; it cannot be
loaded with the local CUDA 12 runtime. The local A800 validation instead uses
vLLM 0.23.0+cu129, PyTorch 2.11.0+cu128, Transformers 4.57.6 and NVIDIA driver
570.148.08. That mixed CUDA-minor combination was tested on this machine; choose
appropriate supported binaries for a different system. See the official
[0.23 GPU installation guide](https://docs.vllm.ai/en/v0.23.0/getting_started/installation/gpu/)
and [release requirements](https://github.com/vllm-project/vllm/blob/v0.23.0/requirements/cuda.txt).

## Transcribe one audio file

Use a merged checkpoint with a trained CTC branch and a UTF-8 hotword file,
one entry per line. Entries can specify aliases and context blacklists, for
example `康辉|康灰~~~央视`.

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0 python infer_vllm.py \
  --model /path/to/ctc-checkpoint \
  --audio /path/to/audio.wav \
  --mode ctc_rag --hot-txt /path/to/hotwords.txt \
  --max-ctx-hotwords 100 --rag-workers 8 --out-json result.jsonl
```

This uses the PyTorch sampler tested locally. FlashInfer sampling may require
additional build tools and CUDA components. `--rag-workers 8` trades more CPU
cores for throughput; the portable default is 4.

The JSONL result includes `text` (final transcript), `rough` (CTC transcript)
and `hotwords` (the actual selected words). The limit is an upper bound; fewer
words may pass retrieval. The default is 10. Pick a GPU with sufficient free
memory and set `--gpu-memory-utilization` when sharing it. Input audio is
resampled to 16 kHz; segments default to at most 30 seconds.

For repeated requests, create one `VLLMContextASR` instance and reuse it:

```python
from qwen3_asr_vllm.runtime import VLLMContextASR

engine = VLLMContextASR("/path/to/ctc-checkpoint", mode="ctc_rag",
                        hot_txt="/path/to/hotwords.txt", max_ctx_hotwords=100)
result = engine.transcribe("/path/to/audio.wav")
print(result["text"], result["rough"], result["hotwords"])
```

`transcribe_batch` accepts a list of audio paths; per-audio vocabularies and
generation limits are documented in [request options](request_options.md).
The Python adapter orchestrates CTC and retrieval: `vllm serve` alone does not
execute this project's hotword pipeline. The current adapter supports one
vLLM worker and GPU per engine instance.

## Compatibility and optimization

The model adapter retains batched audio/CTC preparation, overlapped CPU
retrieval, worker-local audio embeddings, request-vocabulary reuse and native
phoneme refinement. CTC sequence boundaries and maximum length are already
known from the encoded features. Cumulative offsets are built on the CPU and
the maximum is passed as a CPU scalar. The vLLM attention wrapper calls
`.item()` on this scalar for each CTC layer, so this avoids repeated GPU-to-CPU
synchronization; see the [v0.23.0 attention wrapper](https://github.com/vllm-project/vllm/blob/v0.23.0/vllm/v1/attention/ops/vit_attn_wrappers.py).

The test matrix covers real CTC weight loading, audio-encoding reuse,
variable-length packed CTC, per-audio hotword isolation and cache cleanup.
Real-GPU tests passed on vLLM 0.22.1 and 0.23.0 with the local 0.6B CTC
checkpoint, default torch.compile/CUDA Graphs and PyTorch sampling.

### Local latency measurements

On one shared A800, using the 0.6B checkpoint, a 12,286-entry lexicon, up
to 100 retrieved words per audio, four RAG workers, 3 warmups and 20 measured
runs, median CTC+RAG latency was:

| Audio batch | vLLM 0.22.1 | vLLM 0.23.0 |
| --- | ---: | ---: |
| 1 | 62.91 ms | 63.19 ms |
| 4 | 129.73 ms | 121.96 ms |
| 8 | 198.56 ms | 200.91 ms |

Versions were measured sequentially on a shared machine with different
dependency versions, so these numbers do not isolate the effect of vLLM.
The 0.23 timings used cuDNN frontend 1.32.0; the final CLI and GPU integration
checks were repeated with the pinned 1.19.1. The transcripts and selected
words matched in these runs. A separate 0.23
experiment found that passing the CTC maximum as a CPU scalar saved roughly
0.2 ms in the CTC stage; the end-to-end difference was within measurement
noise.

In another 0.23 run under the same workload, increasing RAG workers from 4
to 8 reduced batch-8 median latency from 197.14 to 186.10 ms. Results exclude
model loading and first compilation; tune workers for the available CPU.

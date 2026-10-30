# vLLM 0.23.0: setup and CTC+RAG inference

The locked deployment targets Linux x86-64, Python 3.12, vLLM 0.23.0+cu129,
PyTorch 2.11.0+cu129 and Transformers 4.57.6. The `vllm` and `vllm023` extras
select the same stack. vLLM 0.23 logs a Transformers 4 deprecation warning.

Install with [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
UV_PROJECT_ENVIRONMENT=.venv-uv uv sync --locked --extra vllm --extra server --extra client
source .venv-uv/bin/activate
```

The `.python-version` and `uv.lock` fix the interpreter series and dependency
versions. `tool.uv.sources` selects the official CUDA 12.9 vLLM wheel and
PyTorch index. Do not use plain pip for this lock: it does not apply uv's wheel
sources. The project pins cuDNN frontend 1.19.1 and xgrammar 0.2.3 for the
vLLM 0.23 dependency constraints and Python 3.12 wheel availability.

The PyPI 0.23 wheel encountered during testing links CUDA 13. The lock uses the
official CUDA 12.9 build; the NVIDIA driver must support the selected CUDA
stack. Earlier A800 tests used vLLM 0.23.0+cu129 and PyTorch 2.11.0+cu128 with
driver 570.148.08; they do not establish compatibility for every system. See the official
[0.23 GPU installation guide](https://docs.vllm.ai/en/v0.23.0/getting_started/installation/gpu/)
and [release requirements](https://github.com/vllm-project/vllm/blob/v0.23.0/requirements/cuda.txt).

`uv sync` needs access to the configured indexes and wheel URL unless the uv
cache is already populated. On an offline machine, prepare the matching Linux
x86-64/Python 3.12 dependencies in a uv cache on a connected machine and
transfer that cache. With Python 3.12 installed, run:

```bash
UV_PROJECT_ENVIRONMENT=.venv-uv UV_CACHE_DIR=/path/to/cache \
  uv sync --offline --locked --extra vllm --extra server --extra client
```

Copy the checkpoint and hotword file separately.

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

# Online CTC+RAG service

The HTTP service keeps one `VLLMContextASR` engine loaded. It is a custom API,
not a drop-in OpenAI-compatible `vllm serve` endpoint. Install the server
extra in the same environment as one of the vLLM extras:

```bash
python -m pip install -e '.[vllm023,server]'
```

On a client machine without a GPU, install only `.[client]`.

For CUDA wheel selection see [vLLM 0.23 setup](vllm_023.md). For the
validated local CUDA 12 environment, use the PyTorch sampler:

```bash
CUDA_VISIBLE_DEVICES=3 VLLM_USE_FLASHINFER_SAMPLER=0 \
python -m qwen3_asr_vllm.server \
  --model artifacts/ctc-0.6b/checkpoint-4695 \
  --hot-txt artifacts/aishell/train_hotwords.txt \
  --max-ctx-hotwords 100 --rag-workers 8 \
  --gpu-memory-utilization 0.12 --max-model-len 1024 --max-new-tokens 96 \
  --max-batch-size 8 --max-batch-wait-ms 20 --max-queue-size 128
```

Wait for the server startup message. By default it listens on
`127.0.0.1:8900`; `/health` checks readiness and `/docs` shows the API schema.
The process hosts a single engine on one GPU. Concurrent requests are queued
and grouped into batches of up to `--max-batch-size`. The first request waits
up to `--max-batch-wait-ms` for others; a busy engine can add queueing time.
The default queue holds 128 waiting requests and returns HTTP 503 when full.
One batch executes at a time, and each request retains its own hotwords and
generation limits. `/v1/batching` reports batch and queue counts.
If a batch raises `ValueError`, it is split to isolate the invalid request;
other inference failures are returned to that batch without retrying.
This endpoint does not provide authentication. If it must be reached
from another machine, use `--host 0.0.0.0` only behind appropriate access
control and request-size limits at a reverse proxy.

To cap the KV cache explicitly, add `--kv-cache-mib 1024` to the server
command. This sets the KV cache size, not total GPU memory. vLLM uses that
value instead of automatic KV sizing from `--gpu-memory-utilization`; model
weights, activations, CUDA graphs, and other allocations remain additional.
An undersized KV cache can reduce throughput or prevent startup for a long
context. See the [vLLM cache configuration](https://docs.vllm.ai/en/v0.23.0/api/vllm/config/cache/).
In vLLM 0.23, `--gpu-memory-utilization` still participates in the startup
free-memory check even when KV cache size is set explicitly; it is not a
hard limit on total allocation in that mode.

On a shared A800, with the local 0.6B checkpoint, batch size 8, maximum model
length 1024, 100 hotwords and `gpu_memory_utilization=0.12`, setting the KV
cache to 1024 MiB reduced the process's sampled GPU-memory peak from
10,084 to 5,002 MiB. Startup memory dropped from 8,826 to 3,840 MiB. These
are sampled process measurements, not guaranteed maximums. With fixed
`PYTHONHASHSEED=0`, three repeated runs over eight AISHELL clips and eight
30-second meeting clips returned identical final transcripts, rough transcripts
and selected hotwords under both configurations. A prior run without a fixed
hash seed had differing results on the meeting clips. The runs do not establish
accuracy equivalence for other audio or loads.

The same 1024 MiB KV configuration also passed those tests with
`--gpu-memory-utilization 0.08`; the sampled peak was still 5,002 MiB. This
lower ratio reduces the startup free-memory requirement on an 80 GiB GPU from
about 9.6 to 6.4 GiB, but is not a guarantee that 6.4 GiB is sufficient for
every workload. Benchmark with your own maximum audio length and concurrency.

## Transcribe

Upload an audio file supported by libsndfile. It is converted to mono 16 kHz;
the default duration limit is 30 seconds (`--max-audio-seconds`). The response
contains `text`, `rough`, and the retrieved `hotwords`.

```bash
curl -s http://127.0.0.1:8900/v1/audio/transcriptions \
  -F 'file=@/path/to/audio.wav'
```

To give a request its own vocabulary, submit a JSON list as the multipart
form field `hotwords`:

```bash
curl -s http://127.0.0.1:8900/v1/audio/transcriptions \
  -F 'file=@/path/to/audio.wav' \
  -F 'hotwords=["广州市","康辉"]' \
  -F 'max_ctx_hotwords=100'
```

An omitted `hotwords` field uses the global vocabulary; a supplied list
**replaces** it for that request, and `[]` disables hotwords for that request.
The list is a retrieval vocabulary: CTC+RAG selects matching entries from it,
so it does not guarantee every supplied word enters the prompt or transcript.
`max_ctx_hotwords` is an upper bound; `max_new_tokens` can also be set per
request. The global and other requests' vocabularies are unchanged.

Run the client without arguments to transcribe the configured default audio:

```bash
python qwen3_asr_vllm/client.py
```

Set `DEFAULT_URL`, `DEFAULT_AUDIO` and `DEFAULT_HOTWORDS` near the top of the
client file. `None` uses the server vocabulary; a list supplies request
hotwords. Audio longer than 30 seconds is divided into 30-second segments,
with up to four requests in flight, and results are joined in order. Segment
boundaries can affect recognition; the returned `segments` include timestamps.

The Python and command-line client also supports explicit options:

```bash
python -m qwen3_asr_vllm.client transcribe --audio /path/to/audio.wav \
  --hotword 广州市 --hotword 康辉 --max-ctx-hotwords 100
python -m qwen3_asr_vllm.client transcribe-many \
  --audio /path/to/one.wav --audio /path/to/two.wav --concurrency 8
python -m qwen3_asr_vllm.client update-hotwords --file /path/to/new-hotwords.txt
python -m qwen3_asr_vllm.client batching-info
```

Use `--url http://host:8900` before the subcommand for a remote server. In
Python, create an `ASRClient` and call `transcribe(path, hotwords=[...])`;
`transcribe_many(paths, concurrency=8)` submits concurrent requests and returns
results in input order. It applies one shared request vocabulary to those
requests; use independent `transcribe` calls for different vocabularies. The client raises an
HTTP error on failure and defaults to a 120-second timeout. It connects
directly by default; use `--use-env-proxy` (or `trust_env=True` in Python) if
the connection must use environment proxy settings.

On the shared local A800, with eight AISHELL clips, 12,286 vocabulary entries,
up to 100 hotwords, 8 RAG workers and a 20 ms batch window, two five-run
experiments gave median total HTTP completion times of 817–1,040 ms when the
eight requests were sent sequentially and 228–304 ms when sent concurrently.
The sequential case also incurs the batch window for each request. Outputs
matched; this compares two request patterns, not individual-request latency.

## Replace the global vocabulary without restarting

Send UTF-8 text, one entry per line. Alias and blacklist syntax is supported,
for example `康辉|康灰~~~央视`.

```bash
curl -s -X PUT http://127.0.0.1:8900/v1/hotwords \
  -H 'Content-Type: text/plain; charset=utf-8' \
  --data-binary @/path/to/new-hotwords.txt
curl -s http://127.0.0.1:8900/v1/hotwords
```

The response contains the active `version` and indexed `size`. Index creation
runs in a background thread using CPU and temporarily retains both vocabularies.
Inference already underway uses the old index; after a successful response,
new requests without a per-request vocabulary use the replacement. A request
already waiting in the queue uses the global index active when its inference
call begins; an isolated retry also uses the then-current index. Build or
encoding errors leave the current index active. Updates are serialized. An empty
body clears the global vocabulary.

The replacement lives in this server process only; it is not written back to
`--hot-txt`. To make it survive a restart, save the intended vocabulary to that
file before starting the next process. The default upload limit is 64 MiB
(`--max-vocabulary-mib`); audio upload limit is 64 MiB (`--max-audio-mib`).
Memory, index-build time and search latency also grow with the vocabulary and
must be measured for the actual data. During a rebuild, both indexes consume
memory. Multipart audio may be buffered to temporary storage before the
application checks its limit, so enforce ingress limits at the proxy as well.

On the shared local A800 host, one HTTP smoke test replaced a one-entry index
with 100,000 synthetic entries in 14.39 seconds. A concurrent transcription
completed using the old index during the build; the new index was then used
for transcription. This is a single measurement, not a capacity or latency
guarantee for a different vocabulary or deployment.

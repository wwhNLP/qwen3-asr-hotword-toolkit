# Request-scoped inference

`VLLMContextASR` accepts a vocabulary and limits for each audio. In `ctc_rag`
mode, request hotwords replace the engine's `hot_txt` vocabulary for that audio;
they are candidates for phoneme retrieval, not an unconditional prompt. In
`asr` mode they are sent directly as prompt words. Pure `ctc` mode has no
hotword retrieval or text generation and rejects these request options.

```python
from qwen3_asr_vllm.runtime import VLLMContextASR

engine = VLLMContextASR("/path/to/checkpoint", mode="ctc_rag")
one = engine.transcribe("one.wav", hotwords=["广州", "康辉|康灰"],
                        max_ctx_hotwords=2, max_new_tokens=64)
batch = engine.transcribe_batch(["one.wav", "two.wav", "three.wav"], request_options=[
    {"hotwords": ["广州"], "max_ctx_hotwords": 1},
    {"hotwords": ["康辉"], "max_new_tokens": 32},
    {"hotwords": []},
])
```

`request_options` contains one object or `None` for each audio. Supported fields:

| Field | Meaning |
| --- | --- |
| `hotwords` | A list of single-line entries, preserving the existing `target|alias~~~blacklist` syntax. Omitted or `null` selects the engine vocabulary; `[]` explicitly disables hotwords. |
| `max_ctx_hotwords` | Nonnegative maximum number of retrieved words; omitted or `null` uses the engine default. Zero skips retrieval. In `asr` mode, an explicit value caps direct prompt words. |
| `max_new_tokens` | Positive per-audio generation limit; omitted or `null` uses the engine default. |

`transcribe_batch(..., hotwords=[...])` supplies a shared request vocabulary;
an individual object's `hotwords` overrides it. In particular, `null` in that
object selects the engine vocabulary and `[]` disables hotwords for that item.
Mode, RAG threshold, worker count and model settings remain engine-level
configuration. Invalid options are rejected before dispatching GPU work.
Explicit entries beginning with `#` are words; `#` starts a comment only in a
vocabulary text file.

JSONL records use the same field names:

```json
{"uttid":"a","audios":["one.wav"],"hotwords":["广州","康辉"],"max_ctx_hotwords":2,"max_new_tokens":64}
{"uttid":"b","audios":["two.wav"],"hotwords":[]}
{"uttid":"c","audios":["three.wav"]}
```

Run `infer_vllm.py --model /path/to/checkpoint --test-json requests.jsonl
--mode ctc_rag --hot-txt default-hotwords.txt`. Repeated `--hotword` arguments
provide a shared vocabulary; a JSONL record overrides that vocabulary. Output
`hotwords` in `ctc_rag` is the selected list, and `request_options` records any
per-item options supplied by the input. Existing assistant messages remain
optional references for CER. The JSONL input is checked before creating output
when the CLI preflights its input paths.

## Reuse and limits

Repeated request vocabularies use a per-engine LRU cache: the default is eight
entries and at most 512 KiB of vocabulary text. This is not a bound on Python
or native index memory, which is larger than the source text. Words and aliases
across different vocabularies also reuse up to 8,192 immutable phoneme entries;
entries over 256 UTF-8 bytes bypass that cache. A second bounded temporary cache
allows reuse across microbatches within a single call. All caches hold only
request vocabulary data; scoring and matching are performed for every audio.

Set `request_cache_size=0` in Python or `--request-cache-size 0` in the CLI to
disable reuse across calls. Within-call reuse still applies within its bounds.
`engine.request_cache_info()` reports vocabulary and phoneme cache counters;
the vocabulary hit count excludes temporary within-call reuse.
`engine.clear_request_cache()` clears both. Clear them after changing the
process's pinyin dictionaries. Engine calls remain serialized; this is an
offline/Python interface, not a concurrent HTTP service.

Coarse retrieval selects serial scoring for fewer than 128 vocabulary-entry /
nonempty-query comparisons, avoiding native worker startup for small requests.
Larger workloads use up to the configured `rag_workers`. The same thresholds,
candidate ranking and refinement algorithms are used in either path.

The benchmark accepts `--request-hot-txt path/to/words.txt` and
`--request-cache-size 0` or `8`, for example:

```bash
python scripts/benchmark_vllm.py --model /path/to/checkpoint --audio one.wav \
  --request-hot-txt request-words.txt --request-cache-size 8 \
  --batch-sizes 1 --warmup 3 --repeats 15 --output request-benchmark.json
```

The request hotword file is read before timing. Vocabulary construction is
included in a cache miss and the initial warmups; repeated cached measurements
exclude the first construction. Prefix caching stays disabled. Output and
input paths cannot refer to the same file.

## Local latency measurements

Measured on a shared NVIDIA A800 GPU with the trained 0.6B step-4695 checkpoint,
vLLM 0.22.1, Numba refinement, four RAG workers, preparation microbatch 4, and
eight AISHELL dev inputs. The global vocabulary has 12,286 terms. These cases
pass the same 1,000-word request vocabulary for every audio, limit the selected
prompt to ten words, disable prefix caching, and exclude model startup and file
reading. A single resident engine alternates cache configurations in shuffled
rounds after three warmup batches, with fifteen timed batches per case.

| Utterances per batch | Request cache disabled, P50 | Request cache warm, P50 | Reduction |
| --- | ---: | ---: | ---: |
| 1 | 198.7 ms | 57.0 ms | 71.3% |
| 4 | 232.1 ms | 87.6 ms | 62.2% |
| 8 | 260.4 ms | 121.4 ms | 53.4% |

The disabled case builds the index once per batch, including when the audio is
split into multiple preparation microbatches. The warm and disabled cases
produce identical transcripts and selected words. A first request for a new
vocabulary still pays index construction; the warm numbers do not include it.

To check the unchanged default-vocabulary path, the preceding runtime and the
new runtime alternated in the same resident engine for twenty timed rounds
after three warmups per case. Whole-batch P50 before → after was 70.89 → 70.91 ms
for one input, 137.48 → 137.65 ms for four, and 203.54 → 203.60 ms for eight;
outputs matched. These measurements use the same shared A800 and 12,286-word
engine vocabulary. They do not establish a tail or concurrency guarantee.

A separate single-audio comparison alternates 15 distinct 1,000-word request
vocabularies with 800 words in common. Whole-vocabulary reuse is disabled in
both cases to isolate phoneme reuse. P50 was 192.3 ms without phoneme reuse and
86.5 ms with it. P95 was 236.1 versus 276.9 ms respectively; this small shared-
machine sample does not establish a tail-latency improvement.

On the same eight rough transcripts, an interleaved CPU-only check with ten
request words measured single-input retrieval at 1.08 ms when always starting
native workers versus 0.92 ms with automatic serial scoring. These small
measurements are not a corpus accuracy or concurrent-serving evaluation. Raw
reports and comparison scripts are stored locally in
`artifacts/context-optimization/request-options/` (excluded from Git).

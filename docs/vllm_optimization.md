# Context inference latency and vLLM 0.26 assessment

## Implemented behavior

`VLLMContextASR.transcribe_batch` now prepares audio in microbatches rather than
issuing one worker RPC per utterance. Within a microbatch:

1. Audio features are extracted together, stripped of padding, and concatenated
   with the true feature lengths for one native audio-encoder forward.
2. CTC runs once on packed encoder outputs. Cumulative sequence lengths keep
   attention and decoding separate for each utterance.
3. The driver performs phoneme retrieval and builds prompts on the CPU. For
   multiple microbatches, a background thread handles the preceding microbatch
   while the worker prepares the next. Single-microbatch requests avoid thread
   creation.
4. Encoder outputs stay in the same vLLM worker's GPU memory. Generation receives
   only context IDs and lengths and resolves these to the original tensors.
   IDs never repeat during a worker's lifetime, preventing stale multimodal or
   prefix-cache hits. The request cache has a 256 MiB limit and is released in a
   `finally` block after generation or a failure.

`--prepare-batch-size` defaults to 4. `--batch-size` remains the final decoding
batch size. A larger preparation microbatch reduces RPC overhead but leaves
fewer opportunities to overlap retrieval. `--embedding-transport cpu` retains
lossless CPU payloads for comparison or for batches exceeding the device cache
limit. Pure `ctc` mode does not cache or transfer encoder outputs.

Engine calls are still serialized. CPU retrieval overlaps GPU preparation
within one call; independent requests and GPU decoding are not asynchronously
scheduled by this wrapper. This is an offline/batch optimization, not a streaming
ASR API. End-to-end latency still includes CTC, retrieval and prompt prefill.

## Validation and measurement

The regression suite checks ordering, overlap using synchronization events,
failure cleanup, monotonic cache IDs, byte limits, and exact CPU transport.
Real-GPU checks additionally compare variable-length batch embeddings against
individual embeddings, CTC rough transcripts, CPU/device final transcriptions,
loaded CTC weights, and one encoding per utterance.

Benchmark command (run in the project's vLLM 0.22.1 inference environment):

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/benchmark_vllm.py \
  --model artifacts/ctc-0.6b/checkpoint-<step> \
  --audio /path/to/sample.wav \
  --hot-txt artifacts/aishell/train_hotwords.txt \
  --batch-sizes 1 4 8 --warmup 2 --repeats 10 \
  --max-model-len 1024 --max-new-tokens 96 \
  --gpu-memory-utilization 0.18 \
  --output artifacts/context-benchmark.json
```

The benchmark reports warmed batch P50/P95, utterances/second and real-time
factor. Model initialization is excluded; prefix caching is disabled to avoid
repeated test inputs receiving an unrealistic cache advantage. Audio is read
before timing. A batch P50 is the time to return the entire batch, not per-request
latency under concurrent traffic. Compare the same hardware, model, vocabulary,
inputs, decoding limit and compiler configuration across revisions.
For heterogeneous batches, pass multiple audio paths to `--audio`; the benchmark
cycles through them in the supplied order.

Measured on an NVIDIA A800-SXM4-80GB with vLLM 0.22.1+cu129 and the completed
local 0.6B CTC checkpoint (step 4695). The inputs were the first eight utterances
of the prepared AISHELL dev manifest (`BAC009S0724W0121` through `...W0128`),
with the 12,286-term train vocabulary. Compilation and CUDA Graphs were enabled;
preparation microbatch size was 4. Each case used two warmup batches and ten timed
batches. Vocabulary, prompts, decode limits and the checkpoint were held fixed.

| Utterances per batch | Original P50 | Optimized P50 | P50 change | Original P95 | Optimized P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 135.1 ms | 138.1 ms | 2.3% slower | 153.2 ms | 143.5 ms |
| 4 | 449.2 ms | 409.8 ms | 8.8% faster | 477.5 ms | 415.2 ms |
| 8 | 896.6 ms | 738.5 ms | 17.6% faster | 955.8 ms | 747.4 ms |

Final transcripts, CTC rough transcripts and retrieved hotword lists matched the
original pipeline on these cases. These are small warmed batch measurements,
not a corpus accuracy evaluation or a concurrent-serving latency guarantee.
Ten samples provide only a preliminary P95 estimate. Batching alone did not
improve single-utterance latency; its observed benefit was mainly in batches.
The retrieval optimization below was measured separately after this experiment.

A separate single-utterance profile with a roughly 4.2-second public sample and
the same vocabulary spent approximately 68 ms on CPU retrieval, 13 ms on worker
preparation, 31 ms on final generation, and 3 ms on cache-release RPC. These are
phase medians from a different run and should not be added to the AISHELL table.
Retrieval remains the largest measured cost in that profile.

The six real-GPU regression checks passed with default compilation/CUDA Graphs.
The first five also passed in eager mode. The tests include a rejected oversized
prompt followed by a successful request, with an empty context cache after the
failure. Raw measurements and GPU logs are saved locally under
`artifacts/context-optimization/`; that directory is excluded from Git.

## Added latency versus ASR and retrieval optimization

The second experiment compares three paths using identical resident step-4695
weights: `asr` skips CTC and retrieval; `ctc_no_rag` runs CTC and greedy decoding
with an empty hotword prompt; `ctc_rag` also retrieves up to ten hotwords and
generates with their prompt. This isolates the inference overhead of enabling
the branch. The ASR baseline still has unused CTC weights resident, so this
does not measure the memory savings of removing the branch from a checkpoint.

CPU profiling identified repeated scans of all similar-phoneme groups in the
weighted edit-distance loop. The new implementation builds a direct peer lookup
once at import, preserving each group's membership independently. It also keeps
two distance rows and their start positions instead of full distance/path
matrices. Scoring, boundary constraints, tie order and retrieval thresholds stay
the same. Query results are recomputed on every call.

The following warmed **whole-batch P50** measurements use an A800 on physical
GPU 1, vLLM 0.22.1, the same eight AISHELL dev inputs and 12,286-term vocabulary
as above, preparation microbatches of four, CUDA Graphs/compilation enabled,
prefix caching disabled, `max_model_len=1024`, and `max_new_tokens=96`.
GPU memory utilization is 0.12. Each case has three warmup batches and fifteen
timed batches. The three cases are interleaved in shuffled rounds (seed 42) in
one resident engine. Before/after retrieval measurements use separate engine
runs on the same GPU. Another idle process occupies approximately 1.5 GiB;
these are shared-machine measurements, with possible scheduling noise.

| Batch | ASR, CTC skipped | CTC, retrieval skipped | CTC+RAG before | CTC+RAG after | Current extra time vs ASR |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 47.3 ms | 49.3 ms | 144.5 ms | 104.2 ms | +56.9 ms (+120.5%) |
| 4 | 76.0 ms | 79.3 ms | 442.1 ms | 282.6 ms | +206.6 ms (+271.8%) |
| 8 | 111.3 ms | 115.5 ms | 755.7 ms | 469.6 ms | +358.3 ms (+321.9%) |

ASR and CTC-without-retrieval columns come from the after run. Enabling CTC
without retrieval adds approximately 2.0 / 3.2 / 4.2 ms per batch. The full
retrieval optimization reduces CTC+RAG P50 by 27.9% / 36.1% / 37.9%; retrieval
and the resulting prompt remain the main additional cost. Current CTC+RAG P95
is 111.1 / 289.3 / 584.6 ms; fifteen observations are insufficient for a reliable
tail-latency guarantee. Input duration is 4.281 seconds for the single case.
These timings exclude model startup and audio-file reading.

Validation compares the optimized matcher with the preceding implementation on
20,000 generated cases, including mixed languages, tones, empty sequences and
word boundaries. Full correction results also match on 135 queries (134 distinct)
(eight CTC outputs plus a spread of AISHELL dev reference texts). Interleaved
CPU timing across those queries reduces median retrieval time from 54.4 to
34.2 ms, and mean time from 60.7 to 36.6 ms. All three GPU ablation cases retain
identical outputs before/after, including rough transcripts, ranked hotwords
and final transcriptions. This equivalence check is not a corpus-level measure
of recognition accuracy.

An additional sweep kept the final decoding batch at eight and changed only
the preparation microbatch. P50 was 472.5 / 460.1 / 462.5 / 508.2 ms for sizes
1 / 2 / 4 / 8, with identical outputs. Size 2 differs from the default 4 by only
0.5%, so the default remains 4. The same run profiled a single 4.281-second
input at approximately 101.5 ms total: preparation RPC 14.2 ms, retrieval
54.0 ms, prompt rendering 0.2 ms, generation 30.2 ms and release RPC 2.6 ms.
These are separate phase medians and need not sum to the total median.
CPU retrieval remains the first priority for further optimization. A separate
CPU prototype batches the coarse scoring with RapidFuzz `process.cdist`, using
float64 scores and stable ranking. On these eight CTC texts and 12,286 terms,
coarse scoring takes 153.6 ms with sequential `extract`, versus 143.5 / 76.1 /
41.3 ms with `cdist` using 1 / 2 / 4 CPU workers. Candidate scores and order
match exactly on those inputs. These timings include matrix filtering and
ranking, but exclude alignment/masking, weighted refinement, and ASR. The
3.7x coarse-stage improvement is therefore **not an end-to-end speedup**.
That prototype only parallelized across utterances. The integrated version
below also parallelizes across vocabulary entries, including for single audio.
Another option is moving weighted refinement out of Python. Python threads
alone cannot parallelize that loop while it holds the GIL.

To reproduce the ablation, add `--ablation` to the benchmark command above,
with `--warmup 3 --repeats 15 --gpu-memory-utilization 0.12`, and supply the
eight audio paths to `--audio`. For a single path, use `--mode asr`, `--mode ctc`,
or `--mode ctc_rag`; `--mode ctc_rag --max-ctx-hotwords 0` runs CTC without
retrieval. Raw reports, query comparisons and logs are saved locally under
`artifacts/context-optimization/round2/` (excluded from Git).

## Native CPU retrieval and GPU tradeoffs

The 68 ms retrieval figure above comes from an earlier public-audio profile;
it is not a fixed cost for every input. Vocabulary size, phoneme length and
the number of coarse matches determine the work. This retrieval operates on
phoneme sequences: text normalization/pinyin conversion, full-vocabulary fuzzy
scoring, alignment and masking with OSA distance, then weighted edit-distance
refinement with word boundaries, aliases and contextual blacklists.

The integrated CPU optimization adds `PhonemeCorrector.correct_batch` and
`FastRAG.search_batch`. Native RapidFuzz `cdist` workers divide the vocabulary
dimension, so one utterance can also use multiple cores. `partial_ratio`
scores are symmetric; alignment/masking retains the original input/hotword
orientation. Float64 scores and stable sorting retain scalar ranking and ties.
The accelerated path holds one vocabulary/blacklist snapshot for the microbatch;
`workers=1` retains the original per-utterance serial behavior. Score matrices
are chunked to at most 16 MiB each, with scalar fallback if a single query would
exceed that budget. Each call uses at most `--rag-workers` native workers
(default 4), rather than all available CPU cores.

Measured on physical GPU 3 (A800-SXM4-80GB) with the same completed 0.6B
checkpoint, eight AISHELL dev inputs, 12,286-term vocabulary, preparation
microbatch 4, compiled vLLM 0.22.1, disabled prefix caching, model length 1024,
96-token generation cap and GPU memory utilization 0.12. CPU worker counts
1 / 2 / 4 are shuffled within each of fifteen timed rounds after three warmup
batches per case, in one resident engine. Audio reading and startup are excluded.

| Batch | Serial retrieval P50 | 4-worker retrieval P50 | Serial full pipeline P50 | 4-worker full pipeline P50 | Full pipeline reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 51.2 ms | 38.8 ms | 96.7 ms | 84.3 ms | 12.8% |
| 4 | 189.3 ms | 116.9 ms | 263.9 ms | 191.5 ms | 27.4% |
| 8 | 358.9 ms | 234.9 ms | 453.5 ms | 329.3 ms | 27.4% |

Times are for the whole batch. Retrieval phase medians sum the retrieval calls
within a request; preparation can overlap those calls. For the single input,
serial → 4-worker phase P50 is approximately: phoneme conversion 0.54 → 0.54 ms,
coarse scoring 22.44 → 9.16 ms, alignment 5.41 → 5.41 ms, weighted refinement
22.67 → 22.56 ms. Phase medians need not sum to the total median.
Weighted refinement is now the largest remaining retrieval stage.

All timed cases produce identical CTC text, ranked hotwords and final
transcriptions. Full correction results also match the preceding source
snapshot on 135 queries (134 distinct), with another 1,100 randomized retrieval
queries covering long phoneme sequences and varied cutoffs. Unit checks cover
aliases, blacklists, vocabulary replacement, empty inputs, ties, matrix memory
limits, overlap, and cleanup on errors. Validation passed 61 general CPU checks,
eight phoneme/compatibility checks in the inference environment, and six real
GPU integration tests with default compilation/CUDA Graphs. These small measurements establish P50
improvements, not corpus accuracy or concurrent-serving guarantees. In
particular, single-input P95 was 108.2 ms serial versus 109.7 ms with four
workers; a tail-latency improvement has not been demonstrated.

Pass `--rag-workers 1` or `--rag-workers 4` to either `infer_vllm.py` or
`scripts/benchmark_vllm.py` for serial/native scoring. Reports and the resident
engine comparison script are stored locally in
`artifacts/context-optimization/round3/` (excluded from Git).

### Why CPU, and what would GPU execution require?

From the implementation, the engineering advantages of CPU retrieval are
reuse of RapidFuzz and pinyin/string-processing libraries, direct handling of
variable-length text and rule checks, and overlap with other audio's GPU
encoding. This is an explanation of the current structure, not a documented
statement of the upstream author's intent. RapidFuzz's native scorers can
release the GIL and use CPU workers; the Python refinement loop still runs
serially. See the [RapidFuzz process API](https://rapidfuzz.github.io/RapidFuzz/Usage/process.html).

GPU string matching is possible: [NVIDIA cuDF exposes Levenshtein edit
distance](https://docs.nvidia.com/cudf/26.10/cudf/api_docs/api/cudf.core.accessors.string.StringMethods.edit_distance/).
That primitive does not implement this pipeline's partial-ratio/OSA filtering,
phoneme-specific costs, start/end constraints and stable candidate selection.
Replacing these with plain edit distance changes the retrieval algorithm.

A compatible GPU path would pre-encode the vocabulary, pack variable-length
phoneme IDs and boundary flags, and batch coarse scoring or weighted refinement
in a CUDA/Triton kernel. Pinyin conversion and blacklist processing can remain
on the CPU; the measured conversion cost is only around 0.5 ms for this input.
Integration must schedule the new kernel in the vLLM worker and measure its
RPC/synchronization costs and contention with encoding/decoding. Keeping the
dictionary resident and batching transfers/kernels follows the [CUDA best
practices guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html).
Neither a GPU implementation nor a GPU speedup is established by the current
CPU measurements. The next useful targets are native weighted refinement or a
batched GPU refinement kernel, preserving current scoring and boundary rules.

## vLLM 0.26: scope and difficulty

Assessment date: 2026-10-29. Source comparison uses the official **v0.26.0** tag.
The machine's separate `vllm` environment is actually
`0.26.1rc1.dev1141+g0ecc28479.cu128`, with PyTorch 2.13.0 and Transformers 5.15.1;
it must not be treated as an exact 0.26.0 installation.

**Estimated difficulty: moderate overall; small adapter changes and more
substantial dependency/regression work.** No model retraining is expected from
the interfaces inspected. This is a source/probe assessment, not a declaration
of supported 0.26 inference.

| Area | Evidence and remaining work |
| --- | --- |
| Neural model and CTC | Native Qwen3-ASR, packed audio input, `Qwen3OmniMoeAudioEncoderLayer(hidden_states, cu_seqlens, max_seqlen)`, and weight-loader hooks remain available in v0.26.0. Recheck all CTC tensor hashes and attention boundaries on the actual release. |
| RPC and multimodal input | `LLM.collective_rpc`, `DictEmbeddingItems`, `MultiModalFieldConfig`, processor registration and MRoPE hooks remain available. Verify device context IDs through hashing, encoder-cache eviction, preemption, and error recovery. These are internal extension interfaces. |
| Processor and checkpoint | The local 0.26 development environment imports this adapter and loads the completed 0.6B checkpoint's configuration/processor. CTC settings survive (`add_ctc_branch`, 8404 labels, 5 layers), and a hotword/audio prompt can be rendered. Full tensor loading and generation on exact 0.26.0 remain untested. |
| Dependencies | v0.26.0 requires Transformers >=5.5.3 and quack-kernels >=0.4.0; the current inference extra pins Transformers 4.57.6 and quack-kernels 0.3.3. CUDA requirements also pin several newer kernels. The published source pins PyTorch 2.11.0, so a PyTorch upgrade is not inherently required from this project's current 0.22.1 environment. Use a separate environment and resolve a matching binary stack. |
| Version guard | Runtime currently rejects every version except 0.22.1. Replace this with explicit tested releases only after the 0.26 regression matrix passes; changing the version string alone is insufficient. |
| Training | Keep ms-swift/Transformers 4 training and export separate from Transformers 5 inference. The export/training code has not been validated under Transformers 5. |

A practical migration sequence is: create an isolated exact-0.26.0 environment;
check imports/config/processor; run eager ASR, CTC and CTC+RAG; verify CTC weights
and batch/transport equivalence; test default torch.compile/CUDA Graphs and
failure recovery; then compare recognition metrics and warmed latency. Only
then update dependency extras, the version guard and documentation.

Planning estimate: approximately **1–3 engineering days** if compatible wheels
are available, including regression and measurement. This is an estimate from
the inspected interfaces; native-build or dependency failures could extend it.
Neither a speedup from upgrading nor numerical equivalence across versions is
established by source compatibility.

Sources: [v0.26.0 Qwen3-ASR implementation](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/model_executor/models/qwen3_asr.py),
[CTC encoder layer dependency](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/model_executor/models/qwen3_omni_moe_thinker.py),
[common requirements](https://github.com/vllm-project/vllm/blob/v0.26.0/requirements/common.txt),
[CUDA requirements](https://github.com/vllm-project/vllm/blob/v0.26.0/requirements/cuda.txt),
[release notes](https://github.com/vllm-project/vllm/releases/tag/v0.26.0).

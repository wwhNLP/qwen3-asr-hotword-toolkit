"""Measure warm batch latency and throughput with a resident vLLM engine."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import random
import time

import numpy as np

from qwen3_asr_vllm.runtime import VLLMContextASR, read_audio
from utils.inference_io import ensure_output_not_input


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--audio", type=Path, nargs="+", required=True)
    parser.add_argument("--hot-txt", type=Path)
    parser.add_argument("--mode", choices=("asr", "ctc", "ctc_rag"), default="ctc_rag")
    parser.add_argument("--max-ctx-hotwords", type=int, default=10)
    parser.add_argument("--ablation", action="store_true",
                        help="Compare ASR, CTC without retrieval, and CTC+RAG in randomized rounds.")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--prepare-batch-size", type=int)
    parser.add_argument("--rag-workers", type=int, help="Native CPU retrieval workers; 1 selects serial scoring")
    parser.add_argument("--rag-refiner", choices=("auto", "python", "numba"))
    parser.add_argument("--embedding-transport", choices=("device", "cpu"))
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.2)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0 or any(size < 1 for size in args.batch_sizes):
        parser.error("Batch sizes/repeats must be positive and warmup nonnegative.")
    if args.max_ctx_hotwords < 0 or (args.ablation and args.max_ctx_hotwords == 0):
        parser.error("Hotword limit must be nonnegative (positive for ablation).")
    if args.rag_workers is not None and args.rag_workers < 1:
        parser.error("--rag-workers must be positive.")
    ensure_output_not_input(args.output, [*args.audio, *([args.hot_txt] if args.hot_txt else [])])
    extra = {}
    if args.prepare_batch_size is not None:
        extra["prepare_batch_size"] = args.prepare_batch_size
    if args.embedding_transport is not None:
        extra["embedding_transport"] = args.embedding_transport
    if args.rag_workers is not None:
        extra["rag_workers"] = args.rag_workers
    if args.rag_refiner is not None:
        extra["rag_refiner"] = args.rag_refiner
    engine = VLLMContextASR(
        args.model, mode="ctc_rag" if args.ablation else args.mode,
        hot_txt=args.hot_txt, max_ctx_hotwords=args.max_ctx_hotwords,
        max_num_seqs=max(args.batch_sizes),
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len, max_new_tokens=args.max_new_tokens,
        enforce_eager=args.enforce_eager, enable_prefix_caching=False,
        disable_log_stats=True, **extra,
    )
    waveforms = [read_audio(path) for path in args.audio]
    report = {"vllm": version("vllm"), "model": args.model,
              "audio": [str(path) for path in args.audio],
              "hot_txt": str(args.hot_txt) if args.hot_txt else None,
              "prepare_batch_size": getattr(engine, "prepare_batch_size", None),
              "embedding_transport": getattr(engine, "embedding_transport", None),
              "rag_workers": getattr(engine, "rag_workers", None),
              "rag_refiner": getattr(getattr(engine, "_corrector", None), "refiner_backend", None),
              "max_model_len": args.max_model_len,
              "max_new_tokens": args.max_new_tokens,
              "gpu_memory_utilization": args.gpu_memory_utilization,
              "enforce_eager": args.enforce_eager, "prefix_caching": False,
              "warmup": args.warmup, "repeats": args.repeats,
              "ablation": args.ablation, "random_order_seed": 42, "runs": []}
    cases = [(args.mode, args.mode, args.max_ctx_hotwords)]
    if args.ablation:
        # All cases use identical resident weights and KV-cache reservation.
        # ASR skips CTC; ctc_no_rag executes CTC but supplies no hotword prompt.
        cases = [("asr", "asr", 0), ("ctc_no_rag", "ctc_rag", 0),
                 ("ctc_rag", "ctc_rag", args.max_ctx_hotwords)]
    rng = random.Random(42)
    for size in args.batch_sizes:
        batch = [waveforms[index % len(waveforms)] for index in range(size)]
        measured = {name: {"seconds": []} for name, _, _ in cases}
        for _, mode, limit in cases:
            engine.mode, engine.max_ctx_hotwords = mode, limit
            for _ in range(args.warmup):
                engine.transcribe_batch(batch)
        for _ in range(args.repeats):
            order = list(cases)
            rng.shuffle(order)
            for name, mode, limit in order:
                engine.mode, engine.max_ctx_hotwords = mode, limit
                start = time.perf_counter()
                results = engine.transcribe_batch(batch)
                measured[name]["seconds"].append(time.perf_counter() - start)
                measured[name]["results"] = results
        audio_seconds = sum(len(wave) for wave in batch) / 16000
        for name, mode, limit in cases:
            timings = measured[name]["seconds"]
            report["runs"].append({
                "case": name, "mode": mode, "max_ctx_hotwords": limit,
                "batch": size, "batch_audio_seconds": audio_seconds,
                "seconds": timings, "p50_ms": float(np.percentile(timings, 50) * 1000),
                "p95_ms": float(np.percentile(timings, 95) * 1000),
                "utterances_per_second": size * len(timings) / sum(timings),
                "real_time_factor": sum(timings) / (len(timings) * audio_seconds),
                "results": measured[name]["results"],
            })
    report["stats"] = engine.runtime_stats()
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()

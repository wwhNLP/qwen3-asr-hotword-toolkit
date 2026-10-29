"""File and JSONL inference entry point for the native vLLM context backend."""

import argparse
import json
import os
import time
from itertools import chain
from pathlib import Path

from .runtime import VLLMContextASR
from .requests import normalize_options, validate_mode_options
from utils.asr_metrics import cer_counts
from utils.inference_io import ensure_output_not_input


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("MODEL_PATH"))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--audio", type=Path)
    source.add_argument("--test-json", type=Path, help="JSONL with audios/messages/uttid fields")
    parser.add_argument("--mode", choices=("asr", "ctc", "ctc_rag"), default="ctc_rag")
    parser.add_argument("--hot-txt", type=Path)
    parser.add_argument("--hotword", action="append", help="Request hotword; repeat for multiple words")
    parser.add_argument("--rag-threshold", type=float, default=0.85)
    parser.add_argument("--max-ctx-hotwords", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-audio-seconds", type=float, default=30)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--prepare-batch-size", type=int, default=4, help="Audio encoder/CTC microbatch size")
    parser.add_argument("--rag-workers", type=int, default=4, help="Native CPU retrieval workers; 1 uses serial scoring")
    parser.add_argument("--rag-refiner", choices=("auto", "python", "numba"), default="auto", help="Weighted matching backend; auto uses Numba when available")
    parser.add_argument("--request-cache-size", type=int, default=8, help="Maximum cached request vocabularies; 0 disables caching")
    parser.add_argument("--embedding-transport", choices=("device", "cpu"), default="device", help="Keep features on the worker GPU or transfer through CPU")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--verify-encoder-reuse", action="store_true")
    parser.add_argument("--out-json", type=Path)
    return parser


def iter_samples(path, mode=None):
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            sample = json.loads(line)
            audios = sample.get("audios", [])
            if len(audios) != 1:
                raise ValueError(f"{path}:{line_number}: exactly one audio per record is required.")
            record = {
                "wav": audios[0],
                "uttid": sample.get("uttid", Path(audios[0]).stem),
                "ref": next(
                    (message["content"] for message in reversed(sample.get("messages", []))
                     if message.get("role") == "assistant"),
                    None,
                ),
            }
            options = {name: sample[name] for name in ("hotwords", "max_ctx_hotwords", "max_new_tokens")
                       if name in sample}
            if options:
                try:
                    normalized = normalize_options(1, request_options=[options])
                    validate_mode_options(mode, normalized)
                except ValueError as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
                record["request_options"] = options
            yield record


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.model:
        parser.error("--model or MODEL_PATH is required.")
    if args.batch_size < 1 or args.prepare_batch_size < 1:
        parser.error("--batch-size and --prepare-batch-size must be positive.")
    if args.rag_workers < 1:
        parser.error("--rag-workers must be positive.")
    if args.request_cache_size < 0:
        parser.error("--request-cache-size must be nonnegative.")
    try:
        validate_mode_options(args.mode, normalize_options(1, hotwords=args.hotword))
    except ValueError as exc:
        parser.error(str(exc))
    out_path = args.out_json
    if out_path is None and args.test_json:
        out_path = args.test_json.with_name(f"{args.test_json.stem}_{args.mode}_vllm.jsonl")
    if out_path:
        # Scan the entire manifest before opening output, including later batches.
        inputs = [args.audio] if args.audio else chain(
            [args.test_json], (sample["wav"] for sample in iter_samples(args.test_json, args.mode))
        )
        try:
            ensure_output_not_input(out_path, inputs)
        except ValueError as exc:
            parser.error(str(exc))
    engine = VLLMContextASR(
        args.model,
        mode=args.mode,
        hot_txt=args.hot_txt,
        rag_threshold=args.rag_threshold,
        max_ctx_hotwords=args.max_ctx_hotwords,
        max_new_tokens=args.max_new_tokens,
        max_audio_seconds=args.max_audio_seconds,
        prepare_batch_size=args.prepare_batch_size,
        rag_workers=args.rag_workers,
        rag_refiner=args.rag_refiner,
        request_cache_size=args.request_cache_size,
        embedding_transport=args.embedding_transport,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.batch_size,
        dtype=args.dtype,
        enforce_eager=args.enforce_eager,
    )
    samples = iter([{"wav": str(args.audio), "uttid": args.audio.stem, "ref": None}]) if args.audio else iter_samples(args.test_json, args.mode)
    stream = out_path.open("w", encoding="utf-8") if out_path else None
    before = engine.runtime_stats()["encoder_forward_calls"] if args.verify_encoder_reuse else None
    total = 0
    scored = total_errors = total_characters = 0
    started = time.perf_counter()
    try:
        while True:
            batch = []
            for _ in range(args.batch_size):
                sample = next(samples, None)
                if sample is None:
                    break
                batch.append(sample)
            if not batch:
                break
            outputs = engine.transcribe_batch(
                [sample["wav"] for sample in batch], hotwords=args.hotword,
                request_options=[sample.get("request_options") for sample in batch],
            )
            for sample, output in zip(batch, outputs):
                record = {**sample, **output}
                if sample["ref"] is not None:
                    errors, characters = cer_counts(sample["ref"], output["text"])
                    record.update(err=errors, ref_len=characters)
                    total_errors += errors
                    total_characters += characters
                    scored += 1
                line = json.dumps(record, ensure_ascii=False)
                print(line, flush=True)
                if stream:
                    stream.write(line + "\n")
            total += len(batch)
    finally:
        if stream:
            stream.close()
    summary = {"utterances": total, "inference_seconds": round(time.perf_counter() - started, 3)}
    if scored:
        summary.update(scored_utterances=scored, character_errors=total_errors,
                       reference_characters=total_characters,
                       cer=total_errors / total_characters if total_characters else None)
    if args.verify_encoder_reuse:
        calls = engine.runtime_stats()["encoder_forward_calls"] - before
        summary["encoder_forward_calls"] = calls
        if calls != total:
            raise RuntimeError(f"Expected {total} encoder calls, observed {calls}.")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

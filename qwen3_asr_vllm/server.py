"""Single-process HTTP service for the vLLM CTC+RAG pipeline."""

import argparse
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from .batching import BatchQueueFull, DynamicBatcher
from .requests import normalize_options
from .runtime import VLLMContextASR, read_audio


def _decode_audio(upload, max_samples, max_bytes):
    import numpy as np
    import soundfile as sf

    upload.file.seek(0, 2)
    size = upload.file.tell()
    upload.file.seek(0)
    if size > max_bytes:
        raise ValueError("Audio file exceeds the configured size limit.")
    try:
        info = sf.info(upload.file)
        if info.samplerate <= 0 or info.frames <= 0:
            raise ValueError("Audio must contain samples at a valid sample rate.")
        if info.frames / info.samplerate > max_samples / 16000:
            raise ValueError("Audio exceeds the configured duration limit.")
        upload.file.seek(0)
        waveform = read_audio(upload.file)
        if not len(waveform) or len(waveform) > max_samples or not np.isfinite(waveform).all():
            raise ValueError("Audio must be finite, nonempty, and within the duration limit.")
        return waveform
    except (RuntimeError, sf.LibsndfileError) as exc:
        raise ValueError("Could not decode the audio file.") from exc


def create_app(engine, *, max_audio_bytes=64 * 1024 * 1024,
               max_vocabulary_bytes=64 * 1024 * 1024,
               max_batch_size=8, max_batch_wait_ms=20, max_queue_size=128):
    """Create an app around one already-loaded engine."""
    batcher = DynamicBatcher(engine, max_batch_size=max_batch_size,
                             max_wait_ms=max_batch_wait_ms, max_queue_size=max_queue_size)

    @asynccontextmanager
    async def lifespan(app):
        batcher.start()
        try:
            yield
        finally:
            await batcher.close()

    app = FastAPI(title="Qwen3-ASR CTC+RAG", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/batching")
    async def batching_info():
        return batcher.info()

    @app.get("/v1/hotwords")
    async def hotword_info():
        return await run_in_threadpool(engine.hotword_info)

    @app.put("/v1/hotwords")
    async def replace_hotwords(request: Request):
        chunks = []
        total = 0
        async for chunk in request.stream():
            total += len(chunk)
            if total > max_vocabulary_bytes:
                raise HTTPException(413, "Vocabulary exceeds the configured size limit.")
            chunks.append(chunk)
        try:
            text = b"".join(chunks).decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise HTTPException(400, "Vocabulary must be UTF-8 text.") from exc
        return await run_in_threadpool(engine.update_hotwords, text)

    @app.post("/v1/audio/transcriptions")
    async def transcribe(
        file: UploadFile = File(...),
        hotwords: str | None = Form(None),
        max_ctx_hotwords: int | None = Form(None, ge=0),
        max_new_tokens: int | None = Form(None, ge=1),
    ):
        try:
            words = json.loads(hotwords) if hotwords is not None else None
            # Validate before acquiring the model or decoding uploaded audio.
            normalize_options(1, hotwords=words,
                              request_options=[{"max_ctx_hotwords": max_ctx_hotwords,
                                                "max_new_tokens": max_new_tokens}])
            waveform = await run_in_threadpool(_decode_audio, file, engine.max_audio_samples, max_audio_bytes)
            options = {"hotwords": words, "max_ctx_hotwords": max_ctx_hotwords,
                       "max_new_tokens": max_new_tokens}
            return await batcher.submit(waveform, options)
        except BatchQueueFull as exc:
            raise HTTPException(503, "Inference queue is full; retry later.") from exc
        except (ValueError, json.JSONDecodeError) as exc:
            raise HTTPException(400, str(exc)) from exc
        finally:
            await file.close()

    return app


def main():
    parser = argparse.ArgumentParser(description="Serve Qwen3-ASR CTC+RAG over HTTP (one GPU).")
    parser.add_argument("--model", required=True, help="Merged checkpoint with a trained CTC branch")
    parser.add_argument("--hot-txt", required=True, help="Initial UTF-8 hotword file")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8900)
    parser.add_argument("--max-ctx-hotwords", type=int, default=10)
    parser.add_argument("--rag-workers", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-audio-seconds", type=int, default=30)
    parser.add_argument("--max-audio-mib", type=int, default=64)
    parser.add_argument("--max-vocabulary-mib", type=int, default=64)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--max-batch-wait-ms", type=int, default=20)
    parser.add_argument("--max-queue-size", type=int, default=128)
    args = parser.parse_args()
    if args.max_audio_mib < 1 or args.max_vocabulary_mib < 1:
        parser.error("Upload limits must be positive.")
    if args.max_batch_size < 1 or args.max_batch_wait_ms < 0 or args.max_queue_size < 1:
        parser.error("Batch and queue sizes must be positive; batch wait must be nonnegative.")

    engine = VLLMContextASR(args.model, mode="ctc_rag", hot_txt=args.hot_txt,
                            max_ctx_hotwords=args.max_ctx_hotwords, rag_workers=args.rag_workers,
                            max_new_tokens=args.max_new_tokens, max_audio_seconds=args.max_audio_seconds,
                            gpu_memory_utilization=args.gpu_memory_utilization,
                            max_model_len=args.max_model_len, max_num_seqs=args.max_batch_size)
    app = create_app(engine, max_audio_bytes=args.max_audio_mib * 1024 * 1024,
                     max_vocabulary_bytes=args.max_vocabulary_mib * 1024 * 1024,
                     max_batch_size=args.max_batch_size,
                     max_batch_wait_ms=args.max_batch_wait_ms, max_queue_size=args.max_queue_size)
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()

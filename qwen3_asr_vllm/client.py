"""HTTP client for the Qwen3-ASR CTC+RAG service."""

import argparse
import io
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path

import httpx

# Defaults used when running this file without command-line arguments.
DEFAULT_URL = "http://127.0.0.1:8900"
DEFAULT_AUDIO = Path(__file__).resolve().parent.parent / "sample/项目例会 (2).wav"
DEFAULT_HOTWORDS = None  # e.g. ["项目名称", "人名"]; None uses the server vocabulary.
DEFAULT_CHUNK_SECONDS = 30
DEFAULT_CONCURRENCY = 4
DEFAULT_MAX_NEW_TOKENS = None  # None uses the server setting.


class ASRClient:
    def __init__(self, url=DEFAULT_URL, *, timeout=120, trust_env=False):
        self._client = httpx.Client(base_url=url.rstrip("/"), timeout=timeout, trust_env=trust_env)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self._client.close()

    def transcribe(self, audio, *, hotwords=None, max_ctx_hotwords=None, max_new_tokens=None):
        path = Path(audio)
        with path.open("rb") as stream:
            return self._transcribe_stream(path.name, stream, hotwords=hotwords,
                                           max_ctx_hotwords=max_ctx_hotwords,
                                           max_new_tokens=max_new_tokens)

    def _transcribe_stream(self, filename, stream, *, hotwords, max_ctx_hotwords, max_new_tokens):
        fields = {}
        if hotwords is not None:
            fields["hotwords"] = json.dumps(hotwords, ensure_ascii=False)
        if max_ctx_hotwords is not None:
            fields["max_ctx_hotwords"] = str(max_ctx_hotwords)
        if max_new_tokens is not None:
            fields["max_new_tokens"] = str(max_new_tokens)
        response = self._client.post("/v1/audio/transcriptions", data=fields,
                                     files={"file": (filename, stream, "application/octet-stream")})
        response.raise_for_status()
        return response.json()

    def transcribe_audio(self, audio, *, hotwords=None, max_ctx_hotwords=None, max_new_tokens=None,
                         chunk_seconds=DEFAULT_CHUNK_SECONDS, concurrency=DEFAULT_CONCURRENCY,
                         on_progress=None):
        """Split long audio into bounded requests and join results in time order."""
        import soundfile as sf

        if chunk_seconds <= 0 or concurrency < 1:
            raise ValueError("chunk_seconds and concurrency must be positive.")
        options = {"hotwords": hotwords, "max_ctx_hotwords": max_ctx_hotwords,
                   "max_new_tokens": max_new_tokens}
        with sf.SoundFile(str(audio)) as source:
            frames_per_chunk = int(source.samplerate * chunk_seconds)
            if frames_per_chunk < 1:
                raise ValueError("chunk_seconds is shorter than one audio frame.")
            if len(source) <= frames_per_chunk:
                return self.transcribe(audio, **options)

            def chunks():
                while source.tell() < len(source):
                    start = source.tell()
                    frames = source.read(frames_per_chunk, dtype="float32", always_2d=True)
                    encoded = io.BytesIO()
                    sf.write(encoded, frames, source.samplerate, format="WAV", subtype="PCM_16")
                    encoded.seek(0)
                    yield start / source.samplerate, source.tell() / source.samplerate, encoded

            def send(chunk):
                start, end, encoded = chunk
                result = self._transcribe_stream("segment.wav", encoded, **options)
                return {"start": start, "end": end, **result}

            # Limit memory and outstanding requests to one group of segments.
            segments = []
            total = (len(source) + frames_per_chunk - 1) // frames_per_chunk
            iterator = chunks()
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                while group := list(islice(iterator, concurrency)):
                    segments.extend(pool.map(send, group))
                    if on_progress is not None:
                        on_progress(len(segments), total)
        return {"mode": segments[0]["mode"],
                "text": "".join(segment["text"] for segment in segments),
                "rough": "".join(segment.get("rough", "") for segment in segments),
                "hotwords": list(dict.fromkeys(word for segment in segments for word in segment.get("hotwords", []))),
                "segments": segments}

    def transcribe_many(self, audios, *, hotwords=None, max_ctx_hotwords=None,
                        max_new_tokens=None, concurrency=8):
        """Send independent requests concurrently and return results in input order."""
        if concurrency < 1:
            raise ValueError("concurrency must be positive.")

        def send(audio):
            return self.transcribe(audio, hotwords=hotwords, max_ctx_hotwords=max_ctx_hotwords,
                                   max_new_tokens=max_new_tokens)

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            return list(pool.map(send, audios))

    def replace_hotwords(self, path):
        with Path(path).open("rb") as stream:
            response = self._client.put("/v1/hotwords", content=stream,
                                        headers={"Content-Type": "text/plain; charset=utf-8"})
        response.raise_for_status()
        return response.json()

    def hotword_info(self):
        response = self._client.get("/v1/hotwords")
        response.raise_for_status()
        return response.json()

    def batching_info(self):
        response = self._client.get("/v1/batching")
        response.raise_for_status()
        return response.json()


def main():
    parser = argparse.ArgumentParser(description="Client for Qwen3-ASR CTC+RAG service.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--use-env-proxy", action="store_true", help="Use proxy settings from the environment")
    commands = parser.add_subparsers(dest="command", required=True)
    transcribe = commands.add_parser("transcribe", help="Transcribe an audio file")
    transcribe.add_argument("--audio", default=DEFAULT_AUDIO)
    transcribe.add_argument("--chunk-seconds", type=int, default=DEFAULT_CHUNK_SECONDS)
    transcribe.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    many = commands.add_parser("transcribe-many", help="Send multiple audio requests concurrently")
    many.add_argument("--audio", action="append", required=True, help="Audio path; may be repeated")
    many.add_argument("--concurrency", type=int, default=8)
    for command in (transcribe, many):
        hotwords = command.add_mutually_exclusive_group()
        hotwords.add_argument("--hotword", action="append", help="Request word; may be repeated")
        hotwords.add_argument("--hotwords-file", help="Request vocabulary, one word per line")
        hotwords.add_argument("--no-hotwords", action="store_true", help="Disable hotwords for this request")
        command.add_argument("--max-ctx-hotwords", type=int)
        command.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    update = commands.add_parser("update-hotwords", help="Replace the global vocabulary")
    update.add_argument("--file", required=True)
    commands.add_parser("hotword-info", help="Show global vocabulary version and size")
    commands.add_parser("batching-info", help="Show current batching statistics")
    args = parser.parse_args(sys.argv[1:] or ["transcribe"])
    with ASRClient(args.url, timeout=args.timeout, trust_env=args.use_env_proxy) as client:
        if args.command in {"transcribe", "transcribe-many"}:
            if args.no_hotwords:
                words = []
            elif args.hotwords_file:
                words = [line.strip() for line in Path(args.hotwords_file).read_text(encoding="utf-8-sig").splitlines()
                         if line.strip() and not line.strip().startswith("#")]
            else:
                words = args.hotword if args.hotword is not None else DEFAULT_HOTWORDS
            options = {"hotwords": words, "max_ctx_hotwords": args.max_ctx_hotwords,
                       "max_new_tokens": args.max_new_tokens}
            if args.command == "transcribe":
                result = client.transcribe_audio(args.audio, chunk_seconds=args.chunk_seconds,
                                                 concurrency=args.concurrency,
                                                 on_progress=lambda done, total: print(
                                                     f"已完成 {done}/{total} 段", file=sys.stderr, flush=True),
                                                 **options)
            else:
                result = client.transcribe_many(args.audio, concurrency=args.concurrency, **options)
        elif args.command == "update-hotwords":
            result = client.replace_hotwords(args.file)
        elif args.command == "hotword-info":
            result = client.hotword_info()
        else:
            result = client.batching_info()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""HTTP client for the Qwen3-ASR CTC+RAG service."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx


class ASRClient:
    def __init__(self, url="http://127.0.0.1:8900", *, timeout=120, trust_env=False):
        self._client = httpx.Client(base_url=url.rstrip("/"), timeout=timeout, trust_env=trust_env)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self._client.close()

    def transcribe(self, audio, *, hotwords=None, max_ctx_hotwords=None, max_new_tokens=None):
        fields = {}
        if hotwords is not None:
            fields["hotwords"] = json.dumps(hotwords, ensure_ascii=False)
        if max_ctx_hotwords is not None:
            fields["max_ctx_hotwords"] = str(max_ctx_hotwords)
        if max_new_tokens is not None:
            fields["max_new_tokens"] = str(max_new_tokens)
        path = Path(audio)
        with path.open("rb") as stream:
            response = self._client.post("/v1/audio/transcriptions", data=fields,
                                         files={"file": (path.name, stream, "application/octet-stream")})
        response.raise_for_status()
        return response.json()

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
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--use-env-proxy", action="store_true", help="Use proxy settings from the environment")
    commands = parser.add_subparsers(dest="command", required=True)
    transcribe = commands.add_parser("transcribe", help="Transcribe an audio file")
    transcribe.add_argument("--audio", required=True)
    many = commands.add_parser("transcribe-many", help="Send multiple audio requests concurrently")
    many.add_argument("--audio", action="append", required=True, help="Audio path; may be repeated")
    many.add_argument("--concurrency", type=int, default=8)
    for command in (transcribe, many):
        hotwords = command.add_mutually_exclusive_group()
        hotwords.add_argument("--hotword", action="append", help="Request word; may be repeated")
        hotwords.add_argument("--hotwords-file", help="Request vocabulary, one word per line")
        hotwords.add_argument("--no-hotwords", action="store_true", help="Disable hotwords for this request")
        command.add_argument("--max-ctx-hotwords", type=int)
        command.add_argument("--max-new-tokens", type=int)
    update = commands.add_parser("update-hotwords", help="Replace the global vocabulary")
    update.add_argument("--file", required=True)
    commands.add_parser("hotword-info", help="Show global vocabulary version and size")
    commands.add_parser("batching-info", help="Show current batching statistics")
    args = parser.parse_args()
    with ASRClient(args.url, timeout=args.timeout, trust_env=args.use_env_proxy) as client:
        if args.command in {"transcribe", "transcribe-many"}:
            if args.no_hotwords:
                words = []
            elif args.hotwords_file:
                words = [line.strip() for line in Path(args.hotwords_file).read_text(encoding="utf-8-sig").splitlines()
                         if line.strip() and not line.strip().startswith("#")]
            else:
                words = args.hotword
            options = {"hotwords": words, "max_ctx_hotwords": args.max_ctx_hotwords,
                       "max_new_tokens": args.max_new_tokens}
            if args.command == "transcribe":
                result = client.transcribe(args.audio, **options)
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

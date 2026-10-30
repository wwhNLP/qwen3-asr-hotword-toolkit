import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from qwen3_asr_vllm.client import ASRClient


class ClientTest(unittest.TestCase):
    def setUp(self):
        self.requests = []

        def handler(request):
            self.requests.append((request, request.read()))
            if request.url.path.endswith("transcriptions"):
                return httpx.Response(200, json={"text": "广州", "hotwords": ["广州"]})
            return httpx.Response(200, json={"version": 2, "size": 1})

        self.client = ASRClient("http://test")
        self.client.close()
        self.client._client = httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler))
        self.addCleanup(self.client.close)

    def test_transcription_passes_request_words_and_empty_list(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audio.wav"
            path.write_bytes(b"fake wav")
            result = self.client.transcribe(path, hotwords=["广州"], max_ctx_hotwords=1, max_new_tokens=8)
            self.assertEqual(result["text"], "广州")
            body = self.requests[-1][1]
            self.assertIn(json.dumps(["广州"], ensure_ascii=False).encode(), body)
            self.assertIn(b"fake wav", body)
            self.assertIn(b"max_ctx_hotwords", body)
            self.client.transcribe(path, hotwords=[])
            self.assertIn(b"[]", self.requests[-1][1])
            self.client.transcribe(path)
            self.assertNotIn(b'name="hotwords"', self.requests[-1][1])

    def test_update_streams_file_and_info(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hotwords.txt"
            path.write_text("广州\n", encoding="utf-8")
            self.assertEqual(self.client.replace_hotwords(path), {"version": 2, "size": 1})
            self.assertEqual(self.requests[-1][1], "广州\n".encode())
            self.assertEqual(self.client.hotword_info()["size"], 1)
            self.assertEqual(self.client.batching_info()["size"], 1)

    def test_transcribe_many_preserves_order(self):
        with patch.object(self.client, "transcribe", side_effect=lambda audio, **kwargs: {"audio": audio}):
            result = self.client.transcribe_many(["a", "b", "c"], hotwords=["广州"], concurrency=2)
        self.assertEqual([item["audio"] for item in result], ["a", "b", "c"])
        with self.assertRaises(ValueError):
            self.client.transcribe_many(["a"], concurrency=0)

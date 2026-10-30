import json
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
import numpy as np
import soundfile as sf

from qwen3_asr_vllm.client import ASRClient, DEFAULT_AUDIO, DEFAULT_URL, main


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

    def test_long_audio_splits_and_joins_in_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "long.wav"
            samples = np.concatenate([np.full(1000, .1), np.full(1000, .2), np.full(500, .3)])
            sf.write(path, samples, 1000)

            def recognize(name, stream, **options):
                waveform, _ = sf.read(stream)
                index = round(waveform[0] * 10)
                self.assertEqual(options["hotwords"], ["词"])
                return {"mode": "ctc_rag", "text": str(index), "rough": str(index), "hotwords": ["词"]}

            progress = []
            with patch.object(self.client, "_transcribe_stream", side_effect=recognize):
                result = self.client.transcribe_audio(path, chunk_seconds=1, concurrency=2,
                                                       hotwords=["词"], on_progress=lambda *x: progress.append(x))
            self.assertEqual(result["text"], "123")
            self.assertEqual([(s["start"], s["end"]) for s in result["segments"]],
                             [(0, 1), (1, 2), (2, 2.5)])
            self.assertEqual(result["hotwords"], ["词"])
            self.assertEqual(progress, [(2, 3), (3, 3)])

    def test_long_audio_prefers_quiet_boundary_and_propagates_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "long.wav"
            samples = np.full(4500, 0.2, dtype=np.float32)
            samples[1600:1900] = 0
            sf.write(path, samples, 1000)
            with patch.object(self.client, "_transcribe_stream", return_value={
                "mode": "ctc_rag", "text": "x", "truncated": True
            }):
                result = self.client.transcribe_audio(path, chunk_seconds=2)
            segments = result["segments"]
            self.assertGreaterEqual(segments[0]["end"], 1.6)
            self.assertLessEqual(segments[0]["end"], 1.9)
            self.assertEqual(segments[0]["start"], 0)
            self.assertEqual(segments[-1]["end"], 4.5)
            for left, right in zip(segments, segments[1:]):
                self.assertEqual(left["end"], right["start"])
            self.assertTrue(all(s["end"] - s["start"] <= 2 for s in segments))
            self.assertTrue(result["truncated"])

    def test_no_arguments_uses_configured_defaults(self):
        with patch("sys.argv", ["client.py"]), patch("sys.stdout", new_callable=io.StringIO), \
             patch("qwen3_asr_vllm.client.ASRClient") as client_class:
            client = client_class.return_value.__enter__.return_value
            client.transcribe_audio.return_value = {"text": "结果"}
            main()
        client_class.assert_called_once_with(DEFAULT_URL, timeout=120, trust_env=False)
        self.assertEqual(client.transcribe_audio.call_args.args[0], DEFAULT_AUDIO)
        self.assertIsNone(client.transcribe_audio.call_args.kwargs["hotwords"])

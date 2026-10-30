import io
import json
import threading
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from qwen3_asr_vllm.runtime import VLLMContextASR
from qwen3_asr_vllm.server import create_app


def wav_bytes(seconds=0.1):
    output = io.BytesIO()
    sf.write(output, np.zeros(int(16000 * seconds), dtype=np.float32), 16000, format="WAV")
    return output.getvalue()


class FakeEngine:
    max_audio_samples = 16000

    def __init__(self):
        self.words = ["默认词"]
        self.version = 1

    def hotword_info(self):
        return {"version": self.version, "size": len(self.words)}

    def update_hotwords(self, text):
        self.words = text.splitlines()
        self.version += 1
        return self.hotword_info()

    def transcribe_batch(self, waveforms, *, request_options):
        return [{"text": "测试", "rough": "测试",
                 "hotwords": self.words if option["hotwords"] is None else option["hotwords"]}
                for option in request_options]


class ServerTest(unittest.TestCase):
    def setUp(self):
        self.engine = FakeEngine()
        self.client = TestClient(create_app(self.engine, max_audio_bytes=4096, max_vocabulary_bytes=64))
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.audio = {"file": ("sample.wav", wav_bytes(), "audio/wav")}

    def test_request_vocabulary_overrides_global_and_empty_disables(self):
        response = self.client.post("/v1/audio/transcriptions", files=self.audio)
        self.assertEqual(response.json()["hotwords"], ["默认词"])
        response = self.client.post("/v1/audio/transcriptions", files=self.audio,
                                    data={"hotwords": json.dumps(["请求词"], ensure_ascii=False)})
        self.assertEqual(response.json()["hotwords"], ["请求词"])
        response = self.client.post("/v1/audio/transcriptions", files=self.audio, data={"hotwords": "[]"})
        self.assertEqual(response.json()["hotwords"], [])
        self.assertEqual(self.engine.words, ["默认词"])

    def test_replace_and_invalid_update(self):
        response = self.client.put("/v1/hotwords", content="广州\n上海".encode())
        self.assertEqual(response.json(), {"version": 2, "size": 2})
        response = self.client.post("/v1/audio/transcriptions", files=self.audio)
        self.assertEqual(response.json()["hotwords"], ["广州", "上海"])
        self.assertEqual(self.client.put("/v1/hotwords", content=b"\xff").status_code, 400)
        self.assertEqual(self.client.put("/v1/hotwords", content=b"a" * 65).status_code, 413)
        self.assertEqual(self.client.get("/v1/hotwords").json(), {"version": 2, "size": 2})

    def test_bad_input_is_rejected(self):
        for value in ('{', '"广州"', '[""]', '["one\\ntwo"]'):
            self.assertEqual(self.client.post("/v1/audio/transcriptions", files=self.audio,
                                              data={"hotwords": value}).status_code, 400)
        self.assertEqual(self.client.post("/v1/audio/transcriptions", files={
            "file": ("bad.wav", b"not audio", "audio/wav")}).status_code, 400)
        self.assertEqual(self.client.post("/v1/audio/transcriptions", files={
            "file": ("big.wav", b"a" * 4097, "audio/wav")}).status_code, 400)
        with TestClient(create_app(self.engine, max_audio_bytes=100000)) as duration_client:
            self.assertEqual(duration_client.post("/v1/audio/transcriptions", files={
                "file": ("long.wav", wav_bytes(1.1), "audio/wav")}).status_code, 400)


class HotwordUpdateTest(unittest.TestCase):
    def test_update_builds_without_inference_lock_and_swaps_after_inference(self):
        engine = VLLMContextASR.__new__(VLLMContextASR)
        engine.mode = "ctc_rag"
        engine.rag_threshold = 0.85
        engine.rag_refiner = "auto"
        engine._lock = threading.Lock()
        engine._hotword_update_lock = threading.Lock()
        engine._hotword_version = 1
        old = object()
        engine._corrector = old
        built = threading.Event()

        class Corrector:
            def __init__(self, **kwargs):
                self.hotwords = []

            def update_hotwords(self, text):
                self.hotwords = text.splitlines()
                built.set()

        results = []
        with patch("hotword.PhonemeCorrector", Corrector):
            with engine._lock:
                thread = threading.Thread(target=lambda: results.append(engine.update_hotwords("新词")))
                thread.start()
                self.assertTrue(built.wait(5), "building should not wait for inference")
                self.assertIs(engine._corrector, old)
            thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(results, [{"version": 2, "size": 1}])
        self.assertEqual(engine.hotword_info(), {"version": 2, "size": 1})

    def test_failed_build_preserves_old_vocabulary(self):
        engine = VLLMContextASR.__new__(VLLMContextASR)
        engine.mode = "ctc_rag"
        engine.rag_threshold = 0.85
        engine.rag_refiner = "auto"
        engine._lock = threading.Lock()
        engine._hotword_update_lock = threading.Lock()
        engine._hotword_version = 1
        old = object()
        engine._corrector = old
        with patch("hotword.PhonemeCorrector", side_effect=RuntimeError("build failed")):
            with self.assertRaises(RuntimeError):
                engine.update_hotwords("新词")
        self.assertIs(engine._corrector, old)
        self.assertEqual(engine._hotword_version, 1)


if __name__ == "__main__":
    unittest.main()

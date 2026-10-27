import unittest
from threading import Lock
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch

from qwen3_asr_vllm.runtime import VLLMContextASR, select_hotwords
from qwen3_asr_vllm.transport import pack_embeddings, unpack_embeddings


class HotwordSelectionTest(unittest.TestCase):
    def test_matches_precede_similars_and_are_deduplicated(self):
        result = SimpleNamespace(
            matches=[("甲", "热词甲", 1.0), ("乙", "热词乙", 0.9)],
            similars=[("甲", "热词甲", 0.8), ("丙", "热词丙", 0.8)],
        )
        self.assertEqual(select_hotwords(result, 2), ["热词甲", "热词乙"])
        self.assertEqual(select_hotwords(result, 0), [])


class VLLMRuntimeTest(unittest.TestCase):
    def make_runtime(self, mode):
        runtime = VLLMContextASR.__new__(VLLMContextASR)
        runtime.mode = mode
        runtime.max_audio_samples = 16000
        runtime.sampling_params = object()
        runtime._lock = Lock()
        features = torch.randn(13, 8)
        prepared = {
            "prompt": "system hotwords; audio placeholder",
            "rough": "粗转写",
            "hotwords": ["热词"],
            "audio_embeds": pack_embeddings(features),
            "audio_output_lengths": [13],
        }
        worker_model = SimpleNamespace(context_prepare=Mock(return_value=prepared))
        runtime.llm = SimpleNamespace(
            collective_rpc=Mock(side_effect=lambda method, args=(): [getattr(worker_model, method)(*args)]),
            generate=Mock(return_value=[SimpleNamespace(outputs=[SimpleNamespace(text="最终转写")])]),
        )
        return runtime, worker_model, features

    def test_rag_generation_reuses_worker_embeddings_without_raw_audio(self):
        runtime, worker_model, features = self.make_runtime("ctc_rag")
        waveform = np.zeros(1600, dtype=np.float32)
        output = runtime.transcribe(waveform)
        self.assertEqual(output, {"mode": "ctc_rag", "text": "最终转写", "rough": "粗转写", "hotwords": ["热词"]})
        worker_model.context_prepare.assert_called_once()
        requests = runtime.llm.generate.call_args.args[0]
        payload = requests[0]["multi_modal_data"]["audio"]
        self.assertEqual(set(payload), {"audio_embeds", "audio_output_lengths"})
        torch.testing.assert_close(payload["audio_embeds"], features, atol=0, rtol=0)
        sent_audio = worker_model.context_prepare.call_args.args[0]
        np.testing.assert_array_equal(np.frombuffer(sent_audio, dtype=np.float32), waveform)

    def test_ctc_mode_does_not_run_text_decoder(self):
        runtime, worker_model, _ = self.make_runtime("ctc")
        self.assertEqual(runtime.transcribe(np.zeros(1600)), {"mode": "ctc", "text": "粗转写"})
        worker_model.context_prepare.assert_called_once()
        runtime.llm.generate.assert_not_called()

    def test_asr_mode_passes_manual_hotwords_to_worker(self):
        runtime, worker_model, _ = self.make_runtime("asr")
        output = runtime.transcribe(np.zeros(1600), hotwords=["手工热词"])
        self.assertEqual(output, {"mode": "asr", "text": "最终转写"})
        self.assertEqual(worker_model.context_prepare.call_args.args[1:], ("asr", ["手工热词"]))

    def test_invalid_audio_is_rejected_before_worker_execution(self):
        runtime, _, _ = self.make_runtime("ctc_rag")
        for waveform in (np.array([]), np.zeros((2, 2)), np.array([np.nan]), np.zeros(16001)):
            with self.subTest(shape=waveform.shape):
                with self.assertRaises(ValueError):
                    runtime.transcribe(waveform)
        runtime.llm.collective_rpc.assert_not_called()

    def test_empty_batch_does_not_touch_worker(self):
        runtime, _, _ = self.make_runtime("ctc_rag")
        self.assertEqual(runtime.transcribe_batch([]), [])
        runtime.llm.collective_rpc.assert_not_called()


class EmbeddingTransportTest(unittest.TestCase):
    def test_floating_point_embeddings_round_trip_without_precision_loss(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                tensor = torch.randn(3, 8).to(dtype=dtype).T
                restored = unpack_embeddings(pack_embeddings(tensor))
                torch.testing.assert_close(restored, tensor, atol=0, rtol=0)

    def test_malformed_embedding_payloads_are_rejected(self):
        payload = pack_embeddings(torch.ones(3, 8))
        for invalid in (
            {**payload, "dtype": "int64"},
            {**payload, "shape": [0, 8]},
            {**payload, "data": payload["data"][:-1]},
        ):
            with self.assertRaises(ValueError):
                unpack_embeddings(invalid)


if __name__ == "__main__":
    unittest.main()

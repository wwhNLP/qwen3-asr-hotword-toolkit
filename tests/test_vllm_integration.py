"""Opt-in GPU tests using a real CTC checkpoint and a Mandarin audio sample.

VLLM_TEST_MODEL=/path/to/checkpoint VLLM_TEST_AUDIO=/path/to/audio.wav \
    python -m unittest discover -s tests -p test_vllm_integration.py -v
"""

import os
import re
import unittest

from qwen3_asr_vllm.runtime import VLLMContextASR, read_audio


@unittest.skipUnless(
    os.environ.get("VLLM_TEST_MODEL") and os.environ.get("VLLM_TEST_AUDIO"),
    "Set VLLM_TEST_MODEL and VLLM_TEST_AUDIO to run real-weight vLLM GPU tests.",
)
class VLLMIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = VLLMContextASR(
            os.environ["VLLM_TEST_MODEL"],
            mode="ctc_rag",
            enforce_eager=os.environ.get("VLLM_TEST_EAGER", "1") != "0",
            gpu_memory_utilization=0.3,
            max_model_len=2048,
            max_num_seqs=2,
            max_new_tokens=128,
        )
        cls.waveform = read_audio(os.environ["VLLM_TEST_AUDIO"])
        cls.engine.mode = "ctc"
        rough = cls.engine.transcribe(cls.waveform)["text"]
        phrases = re.findall(r"[\u4e00-\u9fff]{2,}", rough)
        if not phrases:
            raise AssertionError(f"Expected Mandarin CTC text, got {rough!r}.")
        cls.hotword = phrases[0][:4]
        cls.engine._corrector.update_hotwords(cls.hotword)
        cls.engine.mode = "ctc_rag"

    def test_ctc_checkpoint_weights_are_loaded_including_qkv_shards(self):
        import hashlib
        import json
        from contextlib import ExitStack
        from pathlib import Path

        import torch
        from safetensors import safe_open

        (digests,) = self.engine.llm.collective_rpc("context_ctc_weight_digests")
        root = Path(os.environ["VLLM_TEST_MODEL"])
        index_path = root / "model.safetensors.index.json"
        if index_path.exists():
            weight_map = json.loads(index_path.read_text())["weight_map"]
        else:
            with safe_open(root / "model.safetensors", framework="pt") as file:
                weight_map = {key: "model.safetensors" for key in file.keys()}
        checked = set()
        with ExitStack() as stack:
            files = {}
            for key, filename in weight_map.items():
                if not key.startswith("thinker.ctc_branch."):
                    continue
                if filename not in files:
                    files[filename] = stack.enter_context(safe_open(root / filename, framework="pt"))
                name = key.removeprefix("thinker.ctc_branch.")
                actual = digests[name]
                expected = files[filename].get_tensor(key).to(dtype=getattr(torch, actual["dtype"]))
                digest = hashlib.sha256(expected.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                self.assertEqual(digest, actual["sha256"], key)
                checked.add(name)
        self.assertTrue(checked)
        self.assertEqual(checked, set(digests))

    def test_rag_hit_and_final_transcription_encode_audio_once(self):
        before = self.engine.runtime_stats()["encoder_forward_calls"]
        result = self.engine.transcribe(self.waveform)
        after = self.engine.runtime_stats()["encoder_forward_calls"]
        self.assertEqual(after - before, 1)
        self.assertIn(self.hotword, result["hotwords"])
        self.assertTrue(result["rough"])
        self.assertTrue(result["text"])

    def test_variable_length_batch_reuses_each_audio_encoding(self):
        before = self.engine.runtime_stats()["encoder_forward_calls"]
        results = self.engine.transcribe_batch([self.waveform[:16000], self.waveform])
        after = self.engine.runtime_stats()["encoder_forward_calls"]
        self.assertEqual(after - before, 2)
        self.assertEqual(self.engine.runtime_stats()["context_cache_items"], 0)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(isinstance(result["text"], str) for result in results))

    def test_packed_encoding_and_ctc_match_individual_utterances(self):
        from qwen3_asr_vllm.transport import unpack_embeddings
        import torch

        waves = [self.waveform[:16000], self.waveform]
        (batched,) = self.engine.llm.collective_rpc(
            "context_prepare_batch", args=([wave.tobytes() for wave in waves], "ctc_rag", False)
        )
        for wave, item in zip(waves, batched):
            (individual,) = self.engine.llm.collective_rpc(
                "context_prepare_batch", args=([wave.tobytes()], "ctc_rag", False)
            )
            self.assertEqual(item["rough"], individual[0]["rough"])
            torch.testing.assert_close(
                unpack_embeddings(item["audio_embeds"]),
                unpack_embeddings(individual[0]["audio_embeds"]), rtol=0.02, atol=0.03,
            )

    def test_device_and_cpu_transport_produce_same_transcriptions(self):
        waves = [self.waveform[:16000], self.waveform]
        device = self.engine.transcribe_batch(waves)
        self.engine.embedding_transport = "cpu"
        try:
            cpu = self.engine.transcribe_batch(waves)
        finally:
            self.engine.embedding_transport = "device"
        self.assertEqual(device, cpu)

    def test_rejected_prompt_releases_context_and_engine_remains_usable(self):
        self.engine.mode = "asr"
        try:
            with self.assertRaises(ValueError):
                self.engine.transcribe(self.waveform, hotwords=["北京大学" * 2048])
        finally:
            self.engine.mode = "ctc_rag"
        self.assertEqual(self.engine.runtime_stats()["context_cache_items"], 0)
        self.assertTrue(self.engine.transcribe(self.waveform)["text"])


if __name__ == "__main__":
    unittest.main()

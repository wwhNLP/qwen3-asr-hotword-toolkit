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
            enforce_eager=True,
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
        cls.engine.llm.collective_rpc(
            "context_configure",
            args=(
                {
                    "mode": "ctc_rag",
                    "hotword_text": cls.hotword,
                    "threshold": 0.85,
                    "max_hotwords": 10,
                },
            )
        )
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
        self.assertEqual(len(results), 2)
        self.assertTrue(all(isinstance(result["text"], str) for result in results))


if __name__ == "__main__":
    unittest.main()

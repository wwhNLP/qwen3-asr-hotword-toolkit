"""Request vocabulary and output-path handling in the benchmark CLI."""

import contextlib
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from scripts import benchmark_vllm


class RequestBenchmarkTest(unittest.TestCase):
    def test_request_vocabulary_is_used_for_warmup_and_timing(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            audio, vocabulary, output = root / "audio.wav", root / "words.txt", root / "results.json"
            audio.touch()
            vocabulary.write_text("# comment\n广州\n\n康辉\n", encoding="utf-8")
            engine = SimpleNamespace(
                prepare_batch_size=4, embedding_transport="device", rag_workers=4,
                _corrector=SimpleNamespace(refiner_backend="python"), mode="ctc_rag", max_ctx_hotwords=10,
                transcribe_batch=Mock(return_value=[{"mode": "ctc_rag", "text": "广州", "hotwords": ["广州"]}]),
                runtime_stats=Mock(return_value={}), request_cache_info=Mock(return_value={"entries": 0}),
            )
            argv = ["benchmark_vllm.py", "--model", "unused", "--audio", str(audio),
                    "--request-hot-txt", str(vocabulary), "--request-cache-size", "0",
                    "--batch-sizes", "1", "--warmup", "1", "--repeats", "2", "--output", str(output)]
            with patch.object(sys, "argv", argv), patch.object(benchmark_vllm, "VLLMContextASR", return_value=engine) as create, \
                 patch.object(benchmark_vllm, "read_audio", return_value=np.zeros(16000, dtype=np.float32)), \
                 patch.object(benchmark_vllm, "version", return_value="0.22.1"), contextlib.redirect_stdout(io.StringIO()):
                benchmark_vllm.main()
            self.assertEqual(create.call_args.kwargs["request_cache_size"], 0)
            self.assertEqual(engine.transcribe_batch.call_count, 3)
            for call in engine.transcribe_batch.call_args_list:
                self.assertEqual(call.kwargs["hotwords"], ["广州", "康辉"])
            report = json.loads(output.read_text())
            self.assertEqual(report["request_hot_txt"], str(vocabulary))
            self.assertEqual(report["request_cache_size"], 0)

    def test_benchmark_output_cannot_overwrite_request_vocabulary(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            audio, vocabulary = root / "audio.wav", root / "words.txt"
            audio.touch()
            vocabulary.write_text("广州\n", encoding="utf-8")
            argv = ["benchmark_vllm.py", "--model", "unused", "--audio", str(audio),
                    "--request-hot-txt", str(vocabulary), "--output", str(vocabulary)]
            with patch.object(sys, "argv", argv), patch.object(benchmark_vllm, "VLLMContextASR") as create:
                with self.assertRaises(ValueError):
                    benchmark_vllm.main()
            create.assert_not_called()
            self.assertEqual(vocabulary.read_text(), "广州\n")


if __name__ == "__main__":
    unittest.main()

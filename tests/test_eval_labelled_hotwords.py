import contextlib
import csv
import io
import json
import os
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch


class FakeEngine:
    def __init__(self, *args, **kwargs):
        self.mode = "ctc_rag"

    def transcribe_batch(self, audios, request_options=None):
        return [{"text": "测试", "rough": "测试", "hotwords": option["hotwords"]}
                for option in request_options]


class LabelledEvaluationTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.labels = self.root / "labels.csv"
        self.audio = self.root / "audio.wav"
        self.audio.write_bytes(b"audio")
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text("model config", encoding="utf-8")

    def write_labels(self, hotwords):
        with self.labels.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("file_name", "transcript", "hotwords"))
            writer.writeheader()
            writer.writerow({"file_name": self.audio.name, "transcript": "测试", "hotwords": hotwords})

    def run_eval(self, output):
        script = Path(__file__).resolve().parents[1] / "scripts/eval_labelled_hotwords.py"
        argv = [str(script), "--labels", str(self.labels), "--encoding", "utf-8", "--audio-dir", str(self.root),
                "--model", str(self.model), "--output", str(output), "--rag-refiner", "python"]
        with patch("sys.argv", argv), patch("qwen3_asr_vllm.runtime.VLLMContextASR", FakeEngine), \
             patch("qwen3_asr_vllm.runtime.read_audio", return_value=[0]), \
             contextlib.redirect_stdout(io.StringIO()):
            runpy.run_path(str(script), run_name="__main__")

    def test_empty_hotwords_complete_all_comparisons(self):
        self.write_labels("")
        output = self.root / "report.json"
        self.run_eval(output)
        result = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(result["shared_vocabulary"], [])
        self.assertEqual(len(result["cases"]), 7)
        self.assertTrue(all(case["gold_hotwords"] == 0 for case in result["cases"].values()))

    def test_output_cannot_overwrite_labels_audio_or_model(self):
        self.write_labels("热词")
        label_copy = self.labels.read_bytes()
        alias = self.root / "label-hardlink"
        os.link(self.labels, alias)
        for output in (self.labels, alias, self.audio, self.model / "config.json"):
            with self.subTest(output=output), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.run_eval(output)
        self.assertEqual(self.labels.read_bytes(), label_copy)
        self.assertEqual(self.audio.read_bytes(), b"audio")
        self.assertEqual((self.model / "config.json").read_text(encoding="utf-8"), "model config")


if __name__ == "__main__":
    unittest.main()

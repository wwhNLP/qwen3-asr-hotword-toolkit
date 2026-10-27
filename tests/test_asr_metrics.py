import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from qwen3_asr_vllm.cli import main
from utils.asr_metrics import cer_counts, transcript_text


class ASRMetricsTest(unittest.TestCase):
    def test_base_model_prefix_is_removed_and_plain_transcript_is_preserved(self):
        self.assertEqual(transcript_text("language Chinese<asr_text>你好。"), "你好。")
        self.assertEqual(transcript_text("language English<asr_text>Hello."), "Hello.")
        self.assertEqual(transcript_text("你好。"), "你好。")
        self.assertEqual(transcript_text("language is interesting"), "language is interesting")

    def test_cer_ignores_template_spaces_and_punctuation(self):
        self.assertEqual(cer_counts("你好 世界", "language Chinese<asr_text>你好，世界！"), (0, 4))
        self.assertEqual(cer_counts("你好", "你"), (1, 2))
        self.assertEqual(cer_counts("", "你好"), (2, 0))

    def test_cli_reports_corpus_cer_instead_of_average_utterance_cer(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "test.jsonl"
            rows = [{"audios": [f"/{i}.wav"], "messages": [{"role": "assistant", "content": ref}]}
                    for i, ref in enumerate(["甲乙丙", "甲"])]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            engine = Mock()
            engine.transcribe_batch.return_value = [{"text": "甲乙丁"}, {"text": "乙"}]
            stream = io.StringIO()
            with patch("qwen3_asr_vllm.cli.VLLMContextASR", return_value=engine), contextlib.redirect_stdout(stream):
                main(["--model", "dummy", "--test-json", str(path)])
            summary = json.loads(stream.getvalue().splitlines()[-1])
            self.assertEqual(summary["character_errors"], 2)
            self.assertEqual(summary["reference_characters"], 4)
            self.assertEqual(summary["cer"], 0.5)


if __name__ == "__main__":
    unittest.main()

import contextlib
import io
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from qwen3_asr_vllm.cli import main


class VLLMInputOutputPathsTest(unittest.TestCase):
    def test_input_file_cannot_be_used_as_output_via_alias(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            source.write_text('{"audios":["audio.wav"]}\n', encoding="utf-8")
            symlink = root / "symlink.jsonl"
            symlink.symlink_to(source)
            hardlink = root / "hardlink.jsonl"
            os.link(source, hardlink)
            original = source.read_bytes()
            for output in (source, symlink, hardlink):
                with self.subTest(output=output), patch(
                    "qwen3_asr_vllm.cli.VLLMContextASR"
                ) as engine, contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        main(["--model", "unused", "--test-json", str(source),
                              "--out-json", str(output)])
                    self.assertEqual(raised.exception.code, 2)
                    engine.assert_not_called()
                    self.assertEqual(source.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()

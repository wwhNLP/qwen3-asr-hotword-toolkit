import contextlib
import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from qwen3_asr_vllm.cli import main


class VLLMInputOutputPathsTest(unittest.TestCase):
    def test_audio_cannot_be_used_as_output_via_alias(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "input.wav"
            original = b"audio must remain intact"
            audio.write_bytes(original)
            symlink = root / "symbolic.jsonl"
            symlink.symlink_to(audio)
            hardlink = root / "hard.jsonl"
            os.link(audio, hardlink)
            manifest = root / "input.jsonl"
            # The colliding audio is beyond the first batch (batch size = 1).
            manifest.write_text("".join(json.dumps({"audios": [path]}) + "\n"
                                        for path in ("other.wav", str(audio))))
            for source in (["--audio", str(audio)], ["--test-json", str(manifest)]):
                for output in (audio, symlink, hardlink):
                    with self.subTest(source=source, output=output), patch(
                        "qwen3_asr_vllm.cli.VLLMContextASR"
                    ) as engine, contextlib.redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit) as raised:
                            main(["--model", "unused", *source, "--batch-size", "1",
                                  "--out-json", str(output)])
                        self.assertEqual(raised.exception.code, 2)
                        engine.assert_not_called()
                        self.assertEqual(audio.read_bytes(), original)

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

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
    def test_ctc_rejects_request_options_before_loading_or_truncating_output(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, output = root / "input.jsonl", root / "output.jsonl"
            original = "previous results\n"
            output.write_text(original, encoding="utf-8")
            for field, value in (("hotwords", ["广州"]), ("max_ctx_hotwords", 0),
                                 ("max_new_tokens", 8)):
                manifest.write_text(json.dumps({"audios": ["first.wav"]}) + "\n"
                                    + json.dumps({"audios": ["second.wav"], field: value}) + "\n")
                with self.subTest(field=field), patch("qwen3_asr_vllm.cli.VLLMContextASR") as engine, \
                     contextlib.redirect_stderr(io.StringIO()) as error:
                    with self.assertRaises(SystemExit):
                        main(["--model", "unused", "--test-json", str(manifest), "--mode", "ctc",
                              "--batch-size", "1", "--out-json", str(output)])
                    self.assertIn("input.jsonl:2:", error.getvalue())
                    engine.assert_not_called()
                    self.assertEqual(output.read_text(encoding="utf-8"), original)
            with patch("qwen3_asr_vllm.cli.VLLMContextASR") as engine, \
                 contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    main(["--model", "unused", "--audio", "x.wav", "--mode", "ctc", "--hotword", "广州"])
                engine.assert_not_called()

    def test_invalid_later_request_is_rejected_before_engine_or_output(self):
        with TemporaryDirectory() as directory:
            manifest = Path(directory) / "input.jsonl"
            manifest.write_text('{"audios":["a.wav"]}\n'
                                '{"audios":["b.wav"],"hotwords":"bad"}\n', encoding="utf-8")
            error = io.StringIO()
            with patch("qwen3_asr_vllm.cli.VLLMContextASR") as engine, \
                 contextlib.redirect_stderr(error):
                with self.assertRaises(SystemExit) as raised:
                    main(["--model", "unused", "--test-json", str(manifest), "--batch-size", "1"])
                self.assertEqual(raised.exception.code, 2)
                self.assertIn("input.jsonl:2: hotwords", error.getvalue())
                engine.assert_not_called()
            self.assertFalse(manifest.with_name("input_ctc_rag_vllm.jsonl").exists())

    def test_jsonl_passes_distinct_request_options_and_shared_cli_hotwords(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "input.jsonl"
            samples = [
                {"audios": ["one.wav"], "hotwords": ["上海"],
                 "max_ctx_hotwords": 1, "max_new_tokens": 8},
                {"audios": ["two.wav"], "hotwords": []},
                {"audios": ["three.wav"]},
            ]
            manifest.write_text("".join(json.dumps(sample, ensure_ascii=False) + "\n"
                                        for sample in samples), encoding="utf-8")
            with patch("qwen3_asr_vllm.cli.VLLMContextASR") as engine, \
                 contextlib.redirect_stdout(io.StringIO()):
                engine.return_value.transcribe_batch.return_value = [
                    {"mode": "ctc_rag", "text": "转写", "rough": "粗", "hotwords": []}
                    for _ in samples
                ]
                main(["--model", "unused", "--test-json", str(manifest),
                      "--hotword", "公共", "--batch-size", "3"])
            engine.return_value.transcribe_batch.assert_called_once_with(
                ["one.wav", "two.wav", "three.wav"], hotwords=["公共"],
                request_options=[{"hotwords": ["上海"], "max_ctx_hotwords": 1,
                                  "max_new_tokens": 8}, {"hotwords": []}, None],
            )
            output = manifest.with_name("input_ctc_rag_vllm.jsonl")
            records = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual([row["uttid"] for row in records], ["one", "two", "three"])
            self.assertEqual(records[0]["request_options"]["max_new_tokens"], 8)

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

import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

# The Transformers entry point imports this optional audio helper; the tests
# exercise the surrounding logic without decoding real audio or loading weights.
qwen_omni_stub = ModuleType("qwen_omni_utils")
qwen_omni_stub.process_mm_info = Mock()
previous_omni = sys.modules.get("qwen_omni_utils")
sys.modules["qwen_omni_utils"] = qwen_omni_stub
try:
    import infer_ctc
finally:
    if previous_omni is None:
        sys.modules.pop("qwen_omni_utils", None)
    else:
        sys.modules["qwen_omni_utils"] = previous_omni


class TransformersInferenceTest(unittest.TestCase):
    def test_import_preserves_device_selection(self):
        code = (
            "import os, sys, types; "
            "m=types.ModuleType('qwen_omni_utils'); "
            "m.process_mm_info=lambda *a, **k: None; "
            "sys.modules['qwen_omni_utils']=m; "
            "import infer_ctc; "
            "assert 'CUDA_VISIBLE_DEVICES' not in os.environ"
        )
        env = os.environ.copy()
        env.pop("CUDA_VISIBLE_DEVICES", None)
        subprocess.run([sys.executable, "-c", code], env=env, check=True)

    def test_decoding_removes_model_language_marker(self):
        processor = SimpleNamespace(batch_decode=Mock(return_value=["language Chinese<asr_text>你好，世界！"]))
        inputs = {"input_ids": torch.tensor([[1]]), "attention_mask": torch.tensor([[1]])}
        thinker = SimpleNamespace(
            get_input_embeddings=lambda: lambda _: torch.zeros(1, 1, 2),
            get_placeholder_mask=lambda *a, **k: torch.ones(1, 1, 2, dtype=torch.bool),
            generate=Mock(return_value=SimpleNamespace(sequences=torch.tensor([[1]]))),
        )
        text = infer_ctc.asr_from_audio(
            SimpleNamespace(thinker=thinker), processor, "sample.wav", torch.zeros(1, 2),
            base_inputs=inputs,
        )
        self.assertEqual(text, "你好，世界！")
        self.assertEqual(infer_ctc._edit_distance(infer_ctc._norm_cer("你好世界"), infer_ctc._norm_cer(text)), 0)
        self.assertEqual(infer_ctc._norm_cer("language Chinese<asr_text>你好，世界！"), "你好世界")

    def test_single_pass_decoding_removes_model_language_marker(self):
        class Inputs(dict):
            def to(self, *_):
                return self

        inputs = Inputs(input_ids=torch.tensor([[1]]))
        model = SimpleNamespace(device="cpu", dtype=torch.float32,
                                generate=Mock(return_value=SimpleNamespace(sequences=torch.tensor([[1, 2]]))))
        processor = SimpleNamespace(batch_decode=Mock(return_value=["language Chinese<asr_text>你好。 "]))
        with patch.object(infer_ctc, "_prepare_inputs", return_value=inputs):
            self.assertEqual(infer_ctc.generate_asr(model, processor, "sample.wav"), "你好。")

    def test_hotword_limit_is_passed_to_corrector(self):
        corrector = Mock()
        corrector.correct.return_value = SimpleNamespace(
            matches=[], similars=[("source", f"word{i}", 1.0) for i in range(15)]
        )
        self.assertEqual(len(infer_ctc.retrieve_hotwords(corrector, "rough", max_n=15)), 15)
        corrector.correct.assert_called_once_with("rough", k=15)
        corrector.reset_mock()
        self.assertEqual(infer_ctc.retrieve_hotwords(corrector, "rough", max_n=0), [])
        corrector.correct.assert_not_called()

    def test_file_rejects_input_output_aliases_without_truncation(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "input.jsonl"
            original = '{"audios":["audio.wav"]}\n'
            source.write_text(original, encoding="utf-8")
            symlink = Path(folder) / "symbolic.jsonl"
            symlink.symlink_to(source)
            hardlink = Path(folder) / "hard.jsonl"
            os.link(source, hardlink)
            for output in (source, symlink, hardlink):
                with self.subTest(output=output):
                    with self.assertRaisesRegex(ValueError, "different files"):
                        infer_ctc.run_file(None, None, str(source), mode="asr", out_json=str(output))
                    self.assertEqual(source.read_text(encoding="utf-8"), original)

    def test_file_reports_cer_for_base_model_output(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "input.jsonl"
            output = Path(folder) / "output.jsonl"
            source.write_text(json.dumps({"audios": ["audio.wav"], "messages": [{"role": "assistant", "content": "你好世界"}]}, ensure_ascii=False) + "\n", encoding="utf-8")
            with patch.object(infer_ctc, "transcribe", return_value={"mode": "asr", "text": "language Chinese<asr_text>你好，世界！"}), contextlib.redirect_stdout(io.StringIO()):
                cer = infer_ctc.run_file(None, None, str(source), mode="asr", out_json=str(output))
            self.assertEqual(cer, 0)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["err"], 0)

    def test_file_rejects_output_aliasing_a_later_audio(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            audio = root / "input.wav"
            original = b"audio must remain intact"
            audio.write_bytes(original)
            symlink = root / "symbolic.jsonl"
            symlink.symlink_to(audio)
            hardlink = root / "hard.jsonl"
            os.link(audio, hardlink)
            source = root / "input.jsonl"
            source.write_text("".join(json.dumps({"audios": [path]}) + "\n"
                                      for path in ("other.wav", str(audio))))
            for output in (audio, symlink, hardlink):
                with self.subTest(output=output), patch.object(infer_ctc, "transcribe") as transcribe:
                    with self.assertRaisesRegex(ValueError, "different files"):
                        infer_ctc.run_file(None, None, str(source), mode="asr", out_json=str(output))
                    transcribe.assert_not_called()
                    self.assertEqual(audio.read_bytes(), original)

    def run_file_case(self, references, outcomes, mode="asr"):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output = root / "input.jsonl", root / "output.jsonl"
            rows = []
            for index, reference in enumerate(references):
                row = {"audios": [f"{index}.wav"]}
                if reference is not None:
                    row["messages"] = [{"role": "assistant", "content": reference}]
                rows.append(row)
            source.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                              encoding="utf-8")
            log = io.StringIO()
            with patch.object(infer_ctc, "transcribe", side_effect=outcomes), contextlib.redirect_stdout(log):
                cer = infer_ctc.run_file(None, None, str(source), mode=mode, out_json=str(output))
            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            return cer, records, log.getvalue()

    def test_file_excludes_missing_references_from_cer(self):
        cer, records, _ = self.run_file_case(
            ["你好", None], [{"text": "你好"}, {"text": "你好"}],
        )
        self.assertEqual(cer, 0)
        self.assertEqual(records[0]["ref_len"], 2)
        self.assertIsNone(records[1]["ref"])
        self.assertIsNone(records[1]["err"])
        self.assertIsNone(records[1]["ref_len"])
        self.assertEqual(records[1]["status"], "ok")

    def test_file_preserves_explicit_empty_reference_as_scored(self):
        cer, records, _ = self.run_file_case(
            ["你好", ""], [{"text": "你好"}, {"text": "你好"}],
        )
        self.assertEqual(cer, 1)
        self.assertEqual(records[1]["err"], 2)
        self.assertEqual(records[1]["ref_len"], 0)

    def test_file_records_failure_separately_and_continues_scoring(self):
        cer, records, log = self.run_file_case(
            ["你好", "你好", "你好"],
            [{"text": ""}, FileNotFoundError("missing.wav"), {"text": "你好"}],
            mode="ctc_rag",
        )
        self.assertEqual(cer, 0.5)
        self.assertEqual(records[0]["status"], "ok")
        self.assertEqual(records[0]["hyp"], "")
        self.assertEqual(records[0]["err"], 2)
        self.assertNotIn("error", records[0])
        self.assertEqual(records[1]["status"], "error")
        self.assertEqual(records[1]["error"], {"type": "FileNotFoundError", "message": "missing.wav"})
        self.assertIsNone(records[1]["hyp"])
        self.assertIsNone(records[1]["err"])
        self.assertIsNone(records[1]["ref_len"])
        self.assertEqual(records[2]["err"], 0)
        self.assertIn("失败样本      : 1", log)
        self.assertIn("参与 CER 样本 : 2", log)

    def test_file_reports_no_cer_when_no_reference_characters_can_be_scored(self):
        for references, outcomes in (
            ([None], [{"text": "你好"}]),
            (["你好"], [FileNotFoundError("missing.wav")]),
            ([None], [FileNotFoundError("missing.wav")]),
            ([""], [{"text": "你好"}]),
            ([], []),
            ([None] * 50, [{"text": "你好"}] * 50),
        ):
            with self.subTest(references=references):
                cer, _, log = self.run_file_case(references, outcomes)
                self.assertIsNone(cer)
                self.assertIn("CER           : N/A", log)


if __name__ == "__main__":
    unittest.main()

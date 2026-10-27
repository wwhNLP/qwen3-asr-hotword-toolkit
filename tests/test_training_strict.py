import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest


@unittest.skipUnless(importlib.util.find_spec("swift"), "ms-swift is required")
class TrainingStrictTest(unittest.TestCase):
    def test_training_stages_reject_bad_samples_instead_of_replacing_them(self):
        from swift.dataset.utils import LazyLLMDataset
        from swift.arguments.base_args.data_args import DataArguments
        from swift.utils import parse_args
        from training.swift_plugin import ContextCTCTemplate

        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            swift = Path(directory) / "swift"
            swift.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n', encoding="utf-8")
            swift.chmod(0o755)
            env = {**os.environ, "PATH": directory + os.pathsep + os.environ["PATH"],
                   "MODEL_PATH": "unused"}
            for stage in ("sft", "ctc"):
                with self.subTest(stage=stage):
                    result = subprocess.run(
                        ["bash", str(root / "scripts/train_swift.sh"), stage],
                        env=env, check=True, text=True, capture_output=True,
                    )
                    args = result.stdout.splitlines()
                    strict_args, remaining = parse_args(
                        DataArguments, ["--strict", args[args.index("--strict") + 1]]
                    )
                    self.assertEqual(remaining, [])

                    def encode(row, **kwargs):
                        if row == "bad":
                            inputs = SimpleNamespace(audios=["x.wav"], messages=[
                                {"role": "user", "content": "<audio>"},
                                {"role": "assistant", "content": "龘"},
                            ])
                            return ContextCTCTemplate._encode(None, inputs)
                        return {"text": "different sample"}

                    dataset = LazyLLMDataset(
                        ["bad", "good"], encode, strict=strict_args.strict, random_state=0
                    )
                    with self.assertRaisesRegex(ValueError, "unknown CTC target"):
                        dataset[0]


if __name__ == "__main__":
    unittest.main()

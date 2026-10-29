import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class TrainScriptPathsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "project with spaces"
        self.root.mkdir()
        scripts = self.root / "scripts"
        scripts.mkdir()
        shutil.copy(Path(__file__).resolve().parents[1] / "scripts/train_swift.sh", scripts)
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        swift = bin_dir / "swift"
        swift.write_text(
            '#!/bin/sh\n'
            'printf "%s\\n%s\\n%s\\n" "$CUDA_VISIBLE_DEVICES" "$NPROC_PER_NODE" "$MASTER_PORT" > "$SWIFT_ENV_FILE"\n'
            'printf "%s\\n" "$@"\n', encoding="utf-8"
        )
        swift.chmod(0o755)
        self.env = {**os.environ, "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
                    "SWIFT_ENV_FILE": str(self.root / "swift-env.txt")}
        for name in ("MODEL_PATH", "DATA_DIR", "OUTPUT_DIR", "CUDA_VISIBLE_DEVICES",
                     "NPROC_PER_NODE", "MASTER_PORT"):
            self.env.pop(name, None)

    def run_script(self, *args, env=None):
        return subprocess.run(
            ["bash", str(self.root / "scripts/train_swift.sh"), *args],
            cwd="/", env=env or self.env, text=True, capture_output=True,
        )

    def test_sft_defaults_use_project_paths_from_any_directory(self):
        result = self.run_script("sft")
        self.assertEqual(result.returncode, 0, result.stderr)
        args = result.stdout.splitlines()
        self.assertEqual(args[args.index("--model") + 1], str(self.root / "model/Qwen3-ASR-0.6B"))
        self.assertEqual(args[args.index("--dataset") + 1], str(self.root / "artifacts/aishell/train_sft.jsonl"))
        self.assertEqual(args[args.index("--output_dir") + 1], str(self.root / "artifacts/sft-0.6b"))

    def test_ctc_selects_highest_complete_sft_checkpoint(self):
        checkpoints = self.root / "artifacts/sft-0.6b"
        for step in (2, 10, 100):
            checkpoint = checkpoints / f"checkpoint-{step}"
            checkpoint.mkdir(parents=True)
            (checkpoint / "config.json").touch()
            if step != 100:
                (checkpoint / "model.safetensors").touch()
        result = self.run_script("ctc")
        self.assertEqual(result.returncode, 0, result.stderr)
        args = result.stdout.splitlines()
        self.assertEqual(args[args.index("--model") + 1], str(checkpoints / "checkpoint-10"))
        self.assertIn("checkpoint-10", result.stderr)

    def test_ctc_requires_sft_checkpoint_but_allows_override(self):
        result = self.run_script("ctc")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("No SFT checkpoint with config and weights", result.stderr)
        env = {**self.env, "MODEL_PATH": "/chosen/checkpoint"}
        result = self.run_script("ctc", env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        args = result.stdout.splitlines()
        self.assertEqual(args[args.index("--model") + 1], "/chosen/checkpoint")

    def test_default_stage_accepts_extra_arguments_without_stage(self):
        result = self.run_script("--max_steps", "2", env={**self.env, "MODEL_PATH": "/chosen/model"})
        self.assertEqual(result.returncode, 0, result.stderr)
        args = result.stdout.splitlines()
        self.assertEqual(args[args.index("--model_type") + 1], "qwen3_asr_context_ctc")
        self.assertEqual(args[args.index("--dataset") + 1], str(self.root / "artifacts/aishell/train_ctc.jsonl"))
        self.assertEqual(args[-2:], ["--max_steps", "2"])

    def test_default_two_gpu_data_parallel_batch(self):
        result = self.run_script("sft")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.root / "swift-env.txt").read_text().splitlines(), ["3,6", "2", "29531"])
        args = result.stdout.splitlines()
        self.assertEqual(args[args.index("--per_device_train_batch_size") + 1], "8")
        self.assertEqual(args[args.index("--gradient_accumulation_steps") + 1], "8")

    def test_gpu_selection_can_be_overridden(self):
        env = {**self.env, "CUDA_VISIBLE_DEVICES": "1", "MASTER_PORT": "29532"}
        result = self.run_script("sft", env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.root / "swift-env.txt").read_text().splitlines(), ["1", "1", "29532"])
        args = result.stdout.splitlines()
        self.assertEqual(args[args.index("--gradient_accumulation_steps") + 1], "8")

    def test_rejects_more_workers_than_visible_gpus(self):
        env = {**self.env, "CUDA_VISIBLE_DEVICES": "2,6", "NPROC_PER_NODE": "3"}
        result = self.run_script(env=env)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("NPROC_PER_NODE must be between", result.stderr)


if __name__ == "__main__":
    unittest.main()

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class UVLauncherTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'project'
        (self.root / 'scripts').mkdir(parents=True)
        shutil.copy(Path(__file__).resolve().parents[1] / 'scripts/serve.sh', self.root / 'scripts/serve.sh')
        self.tools = Path(self.tmp.name) / 'bin'
        self.tools.mkdir()
        fake_uv = self.tools / 'uv'
        fake_uv.write_text(
            '#!/usr/bin/env python3\n'
            'import json, os, sys\n'
            'with open(os.environ["TEST_UV_CAPTURE"], "w") as f:\n'
            ' json.dump({"args":sys.argv[1:], "device":os.environ.get("CUDA_VISIBLE_DEVICES"), '
            '"offline":os.environ.get("HF_HUB_OFFLINE"), "env":os.environ.get("UV_PROJECT_ENVIRONMENT")}, f)\n'
        )
        fake_uv.chmod(0o755)
        self.capture = Path(self.tmp.name) / 'capture.json'
        self.env = {**os.environ, 'PATH': str(self.tools) + os.pathsep + os.environ['PATH'],
                    'TEST_UV_CAPTURE': str(self.capture)}

    def run_launcher(self):
        return subprocess.run(['bash', str(self.root / 'scripts/serve.sh')], cwd='/',
                              env=self.env, text=True, capture_output=True)

    def test_missing_checkpoint_fails_before_uv_or_network(self):
        result = self.run_launcher()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(str(self.root / 'checkpoint' / 'config.json'), result.stderr)
        self.assertFalse(self.capture.exists())

    def test_sharded_checkpoint_uses_project_relative_config(self):
        checkpoint = self.root / 'checkpoint'
        checkpoint.mkdir()
        for name in ('config.json', 'preprocessor_config.json', 'tokenizer.json',
                     'model.safetensors.index.json'):
            (checkpoint / name).write_text('{}')
        (self.root / 'train_hotwords.txt').write_text('广州\n')
        python = self.root / '.venv-uv/bin/python'
        python.parent.mkdir(parents=True)
        python.write_text('#!/bin/sh\nexit 0\n')
        python.chmod(0o755)
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        captured = json.loads(self.capture.read_text())
        self.assertEqual(captured['device'], '0')
        self.assertEqual(captured['offline'], '1')
        self.assertEqual(captured['env'], str(self.root / '.venv-uv'))
        self.assertIn(str(checkpoint), captured['args'])
        self.assertIn(str(self.root / 'train_hotwords.txt'), captured['args'])
        self.assertIn('--kv-cache-mib', captured['args'])
        self.assertIn('1024', captured['args'])

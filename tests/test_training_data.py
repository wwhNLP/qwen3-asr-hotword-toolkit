import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import wave

from training.prepare_aishell import (
    LexiconMatcher, index_audio, prepare, read_transcripts, sample_hotwords,
)


class AISHELLPreparationTest(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.aishell = self.root / "data_aishell"
        texts = {
            "train": {"train_a": "北京大学欢迎你", "train_b": "清华大学欢迎你"},
            "dev": {"dev_a": "深圳欢迎你"},
            "test": {"test_a": "上海欢迎你"},
        }
        transcript = []
        for split, items in texts.items():
            folder = self.aishell / "wav" / split / f"speaker_{split}"
            folder.mkdir(parents=True)
            for uttid, text in items.items():
                with wave.open(str(folder / f"{uttid}.wav"), "wb") as wav:
                    wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                    wav.writeframes(b"\0\0" * 1600)
                transcript.append(f"{uttid} {' '.join(text)}")
        folder = self.aishell / "transcript"
        folder.mkdir()
        (folder / "aishell_transcript_v0.8.txt").write_text("\n".join(transcript), encoding="utf-8")
        self.lexicon = self.root / "terms.txt"
        self.lexicon.write_text("北京\n清华\n深圳\n上海\n", encoding="utf-8")
        self.seaco = self.root / "seaco"
        for split in ("dev", "test"):
            folder = self.seaco / "data" / split
            folder.mkdir(parents=True)
            (folder / "uttid").write_text(f"{split}_a\n")
            (folder / "hotword.txt").write_text("仅用于评测\n", encoding="utf-8")
            (folder / "r1-hotword.txt").write_text("评测\n", encoding="utf-8")

    def run_prepare(self, output="prepared"):
        out = self.root / output
        args = SimpleNamespace(
            aishell_root=self.aishell, seaco_root=self.seaco, output_dir=out,
            train_hotwords=self.lexicon, min_hotword_frequency=1, empty_ratio=0.2,
            negative_ratio=0.1, max_positive=4, distractors=8, seed=42, max_audio_seconds=30,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            prepare(args)
        return out

    def test_training_does_not_include_dev_test_words_or_audio(self):
        out = self.run_prepare()
        self.assertEqual((out / "train_hotwords.txt").read_text().splitlines(), ["北京", "清华"])
        rows = [json.loads(line) for line in (out / "train_sft.jsonl").read_text().splitlines()]
        self.assertEqual({row["uttid"] for row in rows}, {"train_a", "train_b"})
        self.assertTrue(all("/wav/train/" in row["audios"][0] for row in rows))
        self.assertNotIn("仅用于评测", (out / "train_sft.jsonl").read_text())
        self.assertNotIn("深圳", (out / "train_sft.jsonl").read_text())

    def test_ne_manifests_have_references_and_no_oracle_context(self):
        out = self.run_prepare()
        row = json.loads((out / "test_ne.jsonl").read_text())
        self.assertEqual(row["uttid"], "test_a")
        self.assertEqual(row["messages"][0]["content"], "")
        self.assertEqual(row["messages"][-1]["content"], "上海欢迎你")
        self.assertEqual((out / "test_hotword.txt").read_text(), "仅用于评测\n")

    def test_sampling_is_reproducible(self):
        first, second = self.run_prepare("first"), self.run_prepare("second")
        self.assertEqual((first / "train_sft.jsonl").read_bytes(), (second / "train_sft.jsonl").read_bytes())

    def test_duplicate_audio_across_splits_is_rejected(self):
        original = next((self.aishell / "wav" / "dev").rglob("*.wav"))
        original.rename(original.with_name("train_a.wav"))
        transcript = read_transcripts(self.aishell / "transcript" / "aishell_transcript_v0.8.txt")
        with self.assertRaisesRegex(ValueError, "Duplicate utterance"):
            index_audio(self.aishell, transcript, 30)

    def test_missing_seaco_id_is_not_silently_dropped(self):
        (self.seaco / "data" / "test" / "uttid").write_text("missing\n")
        with self.assertRaisesRegex(ValueError, "Missing test SeACo IDs"):
            self.run_prepare()

    def test_negative_only_sampling_excludes_words_in_reference(self):
        lexicon = ["北京", "清华", "上海"]
        words = sample_hotwords(
            {"uttid": "a", "text": "北京欢迎你"}, lexicon, LexiconMatcher(lexicon),
            empty_ratio=0, negative_ratio=1, distractors=2,
        )
        self.assertEqual(set(words), {"清华", "上海"})


if __name__ == "__main__":
    unittest.main()

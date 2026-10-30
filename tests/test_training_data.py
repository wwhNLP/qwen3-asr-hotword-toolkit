import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
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

    def test_empty_seaco_id_list_is_rejected(self):
        (self.seaco / "data" / "dev" / "uttid").write_text("\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Empty dev SeACo ID list"):
            self.run_prepare()

    def test_zero_distractors_does_not_sample_negative_words(self):
        lexicon = ["北京", "清华", "上海"]
        with patch("random.Random.sample", side_effect=AssertionError("unexpected negative sampling")):
            words = sample_hotwords({"uttid": "a", "text": "北京欢迎你"}, lexicon,
                                    LexiconMatcher(lexicon), empty_ratio=0,
                                    negative_ratio=1, distractors=0)
        self.assertEqual(words, [])

    def test_export_rejects_an_existing_output_file(self):
        from training.export_checkpoint import export

        output = self.root / "existing-file"
        output.write_text("untouched", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "empty directory"):
            export(str(self.root / "source"), str(output))
        self.assertEqual(output.read_text(encoding="utf-8"), "untouched")

    def test_negative_only_sampling_excludes_words_in_reference(self):
        lexicon = ["北京", "清华", "上海"]
        words = sample_hotwords(
            {"uttid": "a", "text": "北京欢迎你"}, lexicon, LexiconMatcher(lexicon),
            empty_ratio=0, negative_ratio=1, distractors=2,
        )
        self.assertEqual(set(words), {"清华", "上海"})

    def test_ctc_normalizes_or_excludes_unknown_labels_without_changing_references(self):
        examples = {
            "dev_width": "蔡卓妍穿着黑色ｔ恤",
            "dev_unknown": "约翰逊将扮演草薙素子这个角色",
        }
        folder = self.aishell / "wav/dev/speaker_dev"
        transcript = self.aishell / "transcript/aishell_transcript_v0.8.txt"
        with transcript.open("a", encoding="utf-8") as out:
            for uttid, text in examples.items():
                with wave.open(str(folder / f"{uttid}.wav"), "wb") as wav:
                    wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                    wav.writeframes(b"\0\0" * 1600)
                out.write(f"\n{uttid} {' '.join(text)}")

        output = self.run_prepare()
        def load(name):
            return {row["uttid"]: row for row in map(json.loads, (output / name).read_text().splitlines())}
        ctc = load("dev_ctc.jsonl")
        self.assertEqual(ctc["dev_width"]["messages"][-1]["content"], "蔡卓妍穿着黑色t恤")
        self.assertNotIn("dev_unknown", ctc)
        for name in ("dev_sft.jsonl", "dev_asr.jsonl"):
            rows = load(name)
            for uttid, text in examples.items():
                self.assertEqual(rows[uttid]["messages"][-1]["content"], text)
        self.assertEqual(set(load("dev_ne.jsonl")), {"dev_a"})
        report = json.loads((output / "summary.json").read_text())["ctc_preparation"]
        self.assertEqual(report["dev"]["normalized_ids"], ["dev_width"])
        self.assertEqual(report["dev"]["skipped_ids"], ["dev_unknown"])
        self.assertEqual(report["dev"]["normalized_count"], 1)
        self.assertEqual(report["dev"]["skipped_count"], 1)
        self.assertEqual(report["train"]["normalized_count"], 0)
        self.assertEqual(report["train"]["skipped_count"], 0)


if __name__ == "__main__":
    unittest.main()

"""Build ms-swift manifests from the official AISHELL-1 train/dev/test split.

SeACo supplies evaluation IDs, not a training split. No dev/test transcript or
hotword list is used to build the training lexicon.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import wave

HOTWORD_PREFIX = "可能出现的热词："


def read_transcripts(path: Path) -> dict[str, str]:
    result = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or fields[0] in result:
            raise ValueError(f"Malformed/duplicate transcript: {path}:{number}")
        # AISHELL transcriptions separate Chinese characters with spaces.
        text = "".join(fields[1].split())
        if not text:
            raise ValueError(f"Empty transcript: {fields[0]}")
        result[fields[0]] = text
    return result


def index_audio(root: Path, transcripts: dict[str, str], max_seconds: float):
    splits = {}
    seen = set()
    skipped = Counter()
    for split in ("train", "dev", "test"):
        folder = root / "wav" / split
        files = sorted(folder.rglob("*.wav"))
        if not files:
            raise ValueError(f"No WAV files in {folder}; unpack the speaker *.tar.gz files first.")
        rows = []
        for path in files:
            uttid = path.stem
            if uttid in seen:
                raise ValueError(f"Duplicate utterance across splits/files: {uttid}")
            seen.add(uttid)
            if uttid not in transcripts:
                skipped[f"{split}_missing_transcript"] += 1
                continue
            with wave.open(str(path), "rb") as wav:
                if wav.getframerate() != 16000 or wav.getnchannels() != 1 or wav.getsampwidth() != 2:
                    raise ValueError(f"Expected 16 kHz mono PCM16: {path}")
                duration = wav.getnframes() / wav.getframerate()
            if duration <= 0 or duration > max_seconds:
                skipped[f"{split}_duration"] += 1
                continue
            rows.append({"uttid": uttid, "audio": str(path.resolve()), "text": transcripts[uttid]})
        if not rows:
            raise ValueError(f"No usable audio in {folder}")
        splits[split] = rows
    return splits, dict(skipped)


def training_lexicon(rows, supplied: Path | None = None, min_frequency: int = 1):
    if supplied is not None:
        candidates = supplied.read_text(encoding="utf-8").splitlines()
    else:
        import jieba
        import jieba.posseg

        cache = Path.home() / ".cache" / "qwen3-asr-context-deploy" / "jieba"
        cache.mkdir(parents=True, exist_ok=True)
        jieba.dt.tmp_dir = str(cache)
        counts = Counter()
        for row in rows:
            for word, tag in jieba.posseg.cut(row["text"]):
                # A lightweight heuristic, not gold named-entity annotation.
                if tag.startswith(("nr", "ns", "nt", "nz")) and re.fullmatch(r"[\u4e00-\u9fff]{2,8}", word):
                    counts[word] += 1
        candidates = [word for word, count in counts.items() if count >= min_frequency]
    candidates = sorted({word.strip() for word in candidates if word.strip()})
    matcher = LexiconMatcher(candidates)
    observed = set()
    for row in rows:
        observed.update(matcher.find(row["text"]))
    result = sorted(observed)
    if not result:
        raise ValueError("No training hotwords found. Supply --train-hotwords with words occurring in train.")
    return result


class LexiconMatcher:
    def __init__(self, words):
        self.trie = {}
        for word in words:
            node = self.trie
            for char in word:
                node = node.setdefault(char, {})
            node[None] = word

    def find(self, text):
        found = set()
        for start in range(len(text)):
            node = self.trie
            for char in text[start:]:
                node = node.get(char)
                if node is None:
                    break
                if None in node:
                    found.add(node[None])
        return sorted(found)


def swift_row(row, hotwords=()):
    return {
        "uttid": row["uttid"],
        "messages": [
            {"role": "system", "content": HOTWORD_PREFIX + "，".join(hotwords) if hotwords else ""},
            {"role": "user", "content": "<audio>"},
            {"role": "assistant", "content": row["text"]},
        ],
        "audios": [row["audio"]],
    }


def sample_hotwords(row, lexicon, matcher, seed=42, empty_ratio=0.2, negative_ratio=0.1,
                    max_positive=4, distractors=8):
    digest = hashlib.sha256(f"{seed}:{row['uttid']}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    condition = rng.random()
    if condition < empty_ratio:
        return []
    positives = matcher.find(row["text"]) if condition >= empty_ratio + negative_ratio else []
    rng.shuffle(positives)
    selected = positives[:max_positive]
    negatives = []
    # Bounded sampling avoids scanning the entire lexicon for every utterance.
    for index in rng.sample(range(len(lexicon)), min(len(lexicon), max(64, distractors * 8))):
        word = lexicon[index]
        if word not in row["text"]:
            negatives.append(word)
        if len(negatives) >= distractors:
            break
    if distractors:
        selected.extend(negatives)
    rng.shuffle(selected)
    return selected


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare(args):
    if not (0 <= args.empty_ratio <= 1 and 0 <= args.negative_ratio <= 1
            and args.empty_ratio + args.negative_ratio <= 1):
        raise ValueError("Sampling ratios must be nonnegative and sum to at most 1.")
    if args.max_positive < 1 or args.distractors < 0 or args.max_audio_seconds <= 0:
        raise ValueError("Invalid hotword count/audio duration.")
    root = Path(args.aishell_root).resolve()
    transcripts = read_transcripts(root / "transcript" / "aishell_transcript_v0.8.txt")
    splits, skipped = index_audio(root, transcripts, args.max_audio_seconds)
    lexicon = training_lexicon(splits["train"], args.train_hotwords, args.min_hotword_frequency)
    matcher = LexiconMatcher(lexicon)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_hotwords.txt").write_text("\n".join(lexicon) + "\n", encoding="utf-8")
    counts = {}
    for split, rows in splits.items():
        # Reference-only manifests can be passed to infer_vllm.py. No oracle hotword prompt.
        write_jsonl(out / f"{split}_asr.jsonl", (swift_row(row) for row in rows))
        counts[split] = len(rows)
        if split in ("train", "dev"):
            write_jsonl(out / f"{split}_ctc.jsonl", (swift_row(row) for row in rows))
            write_jsonl(out / f"{split}_sft.jsonl", (
                swift_row(row, sample_hotwords(
                    row, lexicon, matcher, args.seed, args.empty_ratio, args.negative_ratio,
                    args.max_positive, args.distractors)) for row in rows))
    if args.seaco_root:
        for split in ("dev", "test"):
            source = Path(args.seaco_root) / "data" / split
            ids = {line.split()[0] for line in (source / "uttid").read_text().splitlines() if line.strip()}
            rows = [row for row in splits[split] if row["uttid"] in ids]
            missing = ids - {row["uttid"] for row in rows}
            if missing:
                raise ValueError(f"Missing {split} SeACo IDs after audio filtering: {sorted(missing)[:5]}")
            write_jsonl(out / f"{split}_ne.jsonl", (swift_row(row) for row in rows))
            for name in ("hotword.txt", "r1-hotword.txt"):
                (out / f"{split}_{name}").write_bytes((source / name).read_bytes())
            counts[f"{split}_ne"] = len(rows)
    summary = {
        "aishell_root": str(root), "counts": counts, "skipped": skipped,
        "training_lexicon_size": len(lexicon), "seed": args.seed,
        "hotword_sampling": {"empty_ratio": args.empty_ratio, "negative_only_ratio": args.negative_ratio,
                             "max_positive": args.max_positive, "distractors": args.distractors},
        "note": "Train lexicon uses train only; SeACo dev/test IDs and hotwords are evaluation only.",
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aishell-root", type=Path, required=True)
    parser.add_argument("--seaco-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-hotwords", type=Path, help="Optional train-only lexicon; never use SeACo eval lists.")
    parser.add_argument("--min-hotword-frequency", type=int, default=1)
    parser.add_argument("--empty-ratio", type=float, default=0.2)
    parser.add_argument("--negative-ratio", type=float, default=0.1)
    parser.add_argument("--max-positive", type=int, default=4)
    parser.add_argument("--distractors", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-audio-seconds", type=float, default=30)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()

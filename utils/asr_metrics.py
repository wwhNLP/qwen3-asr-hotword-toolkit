"""Transcript extraction and corpus CER, matching the original CLI normalization."""

import re

_PUNCT_RE = re.compile(
    r"[\s，。！？、；：,\.!\?;:\"'“”‘’（）()《》〈〉【】\[\]{}…—\-·・~`@#\$%\^&\*_\+=\|/\\<>]+"
)


def transcript_text(text):
    text = text.strip()
    prefix, marker, transcript = text.partition("<asr_text>")
    if marker and prefix.startswith("language "):
        return transcript.strip()
    return text


def cer_counts(reference, hypothesis):
    reference = _PUNCT_RE.sub("", transcript_text(reference))
    hypothesis = _PUNCT_RE.sub("", transcript_text(hypothesis))
    previous = list(range(len(hypothesis) + 1))
    for index, char in enumerate(reference, 1):
        current = [index]
        for column, other in enumerate(hypothesis, 1):
            current.append(min(previous[column] + 1, current[-1] + 1,
                               previous[column - 1] + (char != other)))
        previous = current
    return previous[-1], len(reference)

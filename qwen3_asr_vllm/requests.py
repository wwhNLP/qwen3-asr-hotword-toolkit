"""Per-utterance options and a bounded cache of request vocabularies."""

from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple


@dataclass(frozen=True)
class RequestOptions:
    # None uses the engine vocabulary; () explicitly disables hotwords.
    hotwords: Optional[Tuple[str, ...]] = None
    max_ctx_hotwords: Optional[int] = None
    max_new_tokens: Optional[int] = None


def _hotwords(value):
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise ValueError("hotwords must be a list of strings or None.")
    words = []
    for word in value:
        if not isinstance(word, str) or not word.strip() or word.splitlines() != [word]:
            raise ValueError("Each hotword must be a nonempty single-line string.")
        words.append(word.strip())
    return tuple(words)


def normalize_options(count, hotwords=None, request_options=None):
    """Validate an optional shared vocabulary and a per-audio options list."""
    shared = _hotwords(hotwords)
    if request_options is None:
        request_options = [None] * count
    if not isinstance(request_options, (list, tuple)) or len(request_options) != count:
        raise ValueError("request_options must contain one object per audio.")
    result = []
    allowed = {"hotwords", "max_ctx_hotwords", "max_new_tokens"}
    for options in request_options:
        if options is None:
            options = {}
        if not isinstance(options, dict):
            raise ValueError("Each request option must be an object or None.")
        if any(not isinstance(key, str) for key in options):
            raise ValueError("Request option names must be strings.")
        unknown = set(options) - allowed
        if unknown:
            raise ValueError(f"Unknown request options: {sorted(unknown)}.")
        words = _hotwords(options["hotwords"]) if "hotwords" in options else shared
        limit = options.get("max_ctx_hotwords")
        tokens = options.get("max_new_tokens")
        if limit is not None and (type(limit) is not int or limit < 0):
            raise ValueError("max_ctx_hotwords must be a nonnegative integer.")
        if tokens is not None and (type(tokens) is not int or tokens < 1):
            raise ValueError("max_new_tokens must be a positive integer.")
        result.append(RequestOptions(words, limit, tokens))
    return result


class RequestVocabularyCache:
    """LRU bounded by entries and vocabulary text bytes, not total object memory.

    Engine calls serialize access; live request groups retain their own
    references even if another vocabulary is evicted.
    """

    def __init__(self, max_entries=8, max_text_bytes=512 * 1024):
        if type(max_entries) is not int or max_entries < 0:
            raise ValueError("request_cache_size must be a nonnegative integer.")
        self.max_entries = max_entries
        self.max_text_bytes = max_text_bytes
        self._entries = OrderedDict()
        self._text_bytes = 0
        self.hits = self.misses = 0

    def get(self, words, factory):
        entry = self._entries.get(words)
        if entry is not None:
            self._entries.move_to_end(words)
            self.hits += 1
            return entry[0]
        self.misses += 1
        corrector = factory(words)
        size = sum(len(word.encode("utf-8")) + 1 for word in words)
        if self.max_entries and size <= self.max_text_bytes:
            while self._entries and (len(self._entries) >= self.max_entries
                                     or self._text_bytes + size > self.max_text_bytes):
                _, (_, removed_size) = self._entries.popitem(last=False)
                self._text_bytes -= removed_size
            self._entries[words] = (corrector, size)
            self._text_bytes += size
        return corrector

    def info(self):
        return {"entries": len(self._entries), "vocabulary_text_bytes": self._text_bytes,
                "hits": self.hits, "misses": self.misses}

    def clear(self):
        self._entries.clear()
        self._text_bytes = 0
        self.hits = self.misses = 0


class RequestPhonemeCache:
    """Reuse immutable per-entry phonemes across request vocabularies."""

    def __init__(self, max_entries=8192, max_entry_bytes=256):
        self.max_entry_bytes = max_entry_bytes
        self.bypassed = 0

        @lru_cache(maxsize=max_entries)
        def lookup(word):
            from hotword.algo_phoneme import get_phoneme_info

            return tuple(get_phoneme_info(word))

        self._lookup = lookup

    def get(self, word):
        if len(word.encode("utf-8")) > self.max_entry_bytes:
            from hotword.algo_phoneme import get_phoneme_info

            self.bypassed += 1
            return tuple(get_phoneme_info(word))
        return self._lookup(word)

    def info(self):
        info = self._lookup.cache_info()
        return {"entries": info.currsize, "hits": info.hits, "misses": info.misses,
                "bypassed": self.bypassed}

    def clear(self):
        self._lookup.cache_clear()
        self.bypassed = 0

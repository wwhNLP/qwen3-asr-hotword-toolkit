"""Native/Python parity for weighted phonemes, boundaries and vocabulary updates."""

import importlib.util
import random
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


HAS_NATIVE = (importlib.util.find_spec("numba") is not None
              and importlib.util.find_spec("pypinyin") is not None)


@unittest.skipUnless(HAS_NATIVE, "Numba and pypinyin are required for native refinement")
class NativeRefinementTest(unittest.TestCase):
    def test_native_match_rejects_partial_word_at_window_start(self):
        from hotword.native_refine import match_costs

        result = match_costs(np.zeros((1, 1)), np.array([False]), np.array([True]), 0.9)
        self.assertEqual(result, [])

    def test_generated_sequences_match_python_including_window_offsets(self):
        from hotword.algo_calc import fuzzy_substring_search_constrained
        from hotword.native_refine import NativeRefiner

        rng = random.Random(429)
        values = ("an", "ang", "p", "t", "b", "l", "n", "1", "2", "a", "hello", "hallo", "é", "")
        for case in range(400):
            hotword = [(rng.choice(values), rng.choice(("zh", "en", "other")),
                        bool(rng.getrandbits(1)), bool(rng.getrandbits(1)), bool(rng.getrandbits(1)))
                       for _ in range(rng.randrange(10))]
            source = [(rng.choice(values), rng.choice(("zh", "en", "other")),
                       bool(rng.getrandbits(1)), bool(rng.getrandbits(1)), False, index, index + 1)
                      for index in range(rng.randrange(15))]
            phones = [SimpleNamespace(value=item[0], lang=item[1], is_tone=item[4]) for item in hotword]
            native = NativeRefiner({"term": [phones]}).prepare(source, ["term"])
            start = rng.randrange(len(source) + 1)
            end = rng.randrange(start, len(source) + 1)
            threshold = rng.choice((0, 0.55, 0.65, 0.85, 1, rng.random()))
            with self.subTest(case=case):
                self.assertEqual(native.search("term", 0, start, end, threshold),
                                 fuzzy_substring_search_constrained(hotword, source[start:end], threshold))

    def test_aliases_blacklists_and_vocabulary_updates_match_python(self):
        from hotword import PhonemeCorrector

        native = PhonemeCorrector(refiner="numba")
        reference = PhonemeCorrector(refiner="python")
        texts = ["朋友叫康灰，在广洲用Python 3", "央视主持人康灰", "广洲和广州", "hello hallo", "", "!!!"]
        for vocabulary in ("康辉|康灰~~~央视\n广州|广洲\nPython\nhello|hallo", "上海|上嗨\n康辉~~~朋友", ""):
            native.update_hotwords(vocabulary)
            reference.update_hotwords(vocabulary)
            self.assertEqual(native.refiner_backend, "numba" if vocabulary else "python")
            for k in (0, 1, 10):
                self.assertEqual(native.correct_batch(texts, k=k), reference.correct_batch(texts, k=k))

    def test_cost_budget_falls_back_without_changing_results(self):
        from hotword import PhonemeCorrector
        from hotword import native_refine

        corrector = PhonemeCorrector(refiner="numba")
        corrector.update_hotwords("广州|广洲\n康辉")
        expected = corrector.correct("广洲和康灰")
        with patch.object(native_refine, "_MAX_COST_BYTES", 1):
            self.assertEqual(corrector.correct("广洲和康灰"), expected)

        # Repeated phonemes can fit the shared table while one expanded alias
        # would exceed the budget; that alias must use the Python matcher.
        vocabulary = "广州" * 8
        text = "广州" * 7
        corrector.update_hotwords(vocabulary)
        expected = corrector.correct(text)
        self.assertTrue(expected.matches)
        with patch.object(native_refine, "_MAX_COST_BYTES", 4096), \
             patch.object(native_refine, "match_costs", side_effect=AssertionError("Oversized native cost matrix")):
            self.assertEqual(corrector.correct(text), expected)

    def test_vocabulary_warmup_covers_runtime_signature(self):
        from hotword import PhonemeCorrector
        from hotword.native_refine import match_costs

        corrector = PhonemeCorrector(refiner="numba")
        corrector.update_hotwords("广州\n康辉")
        signatures = list(match_costs.nopython_signatures)
        self.assertTrue(signatures)
        corrector.correct("广洲和康灰")
        self.assertEqual(match_costs.nopython_signatures, signatures)
        self.assertEqual(match_costs(np.empty((0, 0)), np.zeros(0, dtype=bool),
                                    np.zeros(0, dtype=bool), 0.6), [])

    def test_optional_dependency_fallback_and_explicit_failure(self):
        from hotword import PhonemeCorrector

        automatic = PhonemeCorrector(refiner="auto")
        explicit = PhonemeCorrector(refiner="numba")
        explicit.update_hotwords("广州")
        expected = explicit.correct("广洲")
        with patch.dict(sys.modules, {"hotword.native_refine": None}):
            automatic.update_hotwords("广州")
            self.assertEqual(automatic.refiner_backend, "python")
            self.assertEqual(automatic.correct("广洲"), expected)
            with self.assertRaisesRegex(RuntimeError, "Numba refinement is unavailable"):
                explicit.update_hotwords("上海")
        # A failed update retains the preceding vocabulary/refiner snapshot.
        self.assertEqual(explicit.correct("广洲"), expected)

    def test_auto_falls_back_when_numba_initialization_fails(self):
        from hotword import PhonemeCorrector
        from hotword import native_refine

        automatic = PhonemeCorrector(refiner="auto")
        reference = PhonemeCorrector(refiner="python")
        reference.update_hotwords("广州")
        for error in (AttributeError("coverage.types.Tracer"), RuntimeError("JIT failed")):
            with self.subTest(error=error), patch.object(native_refine, "NativeRefiner", side_effect=error):
                automatic.update_hotwords("广州")
                self.assertEqual(automatic.refiner_backend, "python")
                self.assertEqual(automatic.correct("广洲"), reference.correct("广洲"))
                with self.assertRaisesRegex(RuntimeError, "Numba refinement is unavailable"):
                    PhonemeCorrector(refiner="numba").update_hotwords("广州")


if __name__ == "__main__":
    unittest.main()

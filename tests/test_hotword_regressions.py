"""Regression tests for retrieval limits and contextual blacklists."""

import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import Mock


HAS_PINYIN = importlib.util.find_spec("pypinyin") is not None
HAS_VLLM = importlib.util.find_spec("vllm") is not None


@unittest.skipUnless(HAS_PINYIN, "pypinyin is required for hotword retrieval")
class HotwordRetrievalTest(unittest.TestCase):
    def test_similar_phoneme_lookup_preserves_group_membership(self):
        from hotword.algo_calc import SIMILAR_PHONEMES, _is_similar_phoneme

        values = {value for group in SIMILAR_PHONEMES for value in group}
        values.update(("", "unknown", "a", "1", "AA"))
        for left in values:
            for right in values:
                expected = any({left, right}.issubset(group) for group in SIMILAR_PHONEMES)
                self.assertEqual(_is_similar_phoneme(left, right), expected, (left, right))
        self.assertTrue(_is_similar_phoneme("p", "b"))
        self.assertTrue(_is_similar_phoneme("p", "t"))
        self.assertFalse(_is_similar_phoneme("t", "b"))

    def test_weighted_matching_preserves_boundaries_scores_and_ties(self):
        from hotword.algo_calc import fuzzy_substring_search_constrained

        def word(*values, lang="zh"):
            return [(value, lang, index == 0, index == len(values) - 1, value.isdigit())
                    for index, value in enumerate(values)]

        cases = [
            (word("p", "an"), word("p", "an") * 2, 0.9,
             [(1.0, 0, 2), (1.0, 2, 4)]),
            (word("an"), word("p", "an"), 0.9, []),
            (word("p"), word("p", "an"), 0.9, []),
            (word("p", "an"), word("b", "ang"), 0.5, [(0.5, 0, 2)]),
            (word("p", "an"), word("b", "ang"), 0.55, []),
            (word("1"), word("2"), 0.5, [(0.5, 0, 1)]),
            (word("hello", lang="en"), word("hallo", lang="en"), 0.6, [(0.8, 0, 1)]),
            (word("a"), word("a", lang="en"), 0.6, []),
            ([], word("a"), 0.6, []),
            (word("a"), [], 0.6, []),
        ]
        for hotword, text, threshold, expected in cases:
            with self.subTest(hotword=hotword, text=text, threshold=threshold):
                self.assertEqual(fuzzy_substring_search_constrained(hotword, text, threshold), expected)

    def test_blacklisted_occurrence_is_excluded_from_prompt_candidates(self):
        from hotword import PhonemeCorrector

        corrector = PhonemeCorrector(threshold=0.85)
        corrector.update_hotwords("康辉~~~央视")
        blocked = corrector.correct("央视主持人康灰")
        self.assertEqual(blocked.matches, [])
        self.assertEqual(blocked.similars, [])

        # A second occurrence outside the blacklist window remains eligible.
        both = corrector.correct("央视主持人康灰，后来一二三四五六七八九十，朋友叫康灰")
        self.assertEqual(both.text, "央视主持人康灰，后来一二三四五六七八九十，朋友叫康辉")
        self.assertIn(("康灰", "康辉", 1.0), both.similars)

@unittest.skipUnless(HAS_VLLM, "vLLM is required for its model adapter")
class VLLMHotwordLimitTest(unittest.TestCase):
    def test_worker_passes_configured_limit_to_retrieval(self):
        import qwen3_asr_vllm.model as model_module

        candidates = [("粗", f"词{i}", 1.0) for i in range(12)]
        corrector = Mock()
        corrector.correct.side_effect = lambda text, k: SimpleNamespace(matches=[], similars=candidates[:k])
        processor = SimpleNamespace(apply_chat_template=Mock(return_value="prompt"))
        fake_model = SimpleNamespace(
            _context_processor=processor,
            prepare_context_batch=Mock(return_value=[{"rough": "粗"}]),
            _context_corrector=corrector,
            _max_ctx_hotwords=12,
        )
        waveform = [0.0] * 100
        result = model_module.Qwen3ASRContextForConditionalGeneration.prepare_context(
            fake_model, waveform, mode="ctc_rag", hotwords=[]
        )
        fake_model.prepare_context_batch.assert_called_once_with(
            [waveform], mode="ctc_rag", cache_on_device=False
        )
        corrector.correct.assert_called_once_with("粗", k=12)
        self.assertEqual(result["hotwords"], [f"词{i}" for i in range(12)])


if __name__ == "__main__":
    unittest.main()

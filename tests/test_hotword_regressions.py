"""Regression tests for retrieval limits and contextual blacklists."""

import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import Mock


HAS_PINYIN = importlib.util.find_spec("pypinyin") is not None
HAS_VLLM = importlib.util.find_spec("vllm") is not None


@unittest.skipUnless(HAS_PINYIN, "pypinyin is required for hotword retrieval")
class HotwordRetrievalTest(unittest.TestCase):
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

"""Regression tests for retrieval limits and contextual blacklists."""

import importlib.util
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch


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
        import torch
        import qwen3_asr_vllm.model as model_module

        features = {
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
            "input_features": torch.ones((1, 1, 3)),
        }
        candidates = [("粗", f"词{i}", 1.0) for i in range(12)]
        corrector = Mock()
        corrector.correct.side_effect = lambda text, k: SimpleNamespace(matches=[], similars=candidates[:k])
        processor = SimpleNamespace(feature_extractor=Mock(return_value=features),
                                    apply_chat_template=Mock(return_value="prompt"))
        fake_model = SimpleNamespace(
            _context_processor=processor,
            audio_tower=SimpleNamespace(device=torch.device("cpu"), dtype=torch.float32),
            vllm_config=object(),
            _process_audio_input=Mock(return_value=(torch.ones((3, 2)),)),
            ctc_branch=Mock(return_value=torch.ones((3, 2))),
            _context_vocab=SimpleNamespace(ctc_greedy_decode=Mock(return_value="粗")),
            _context_corrector=corrector,
            _max_ctx_hotwords=12,
        )
        with patch.object(model_module, "set_forward_context", return_value=nullcontext()):
            result = model_module.Qwen3ASRContextForConditionalGeneration.prepare_context(
                fake_model, [0.0] * 100, mode="ctc_rag", hotwords=[]
            )
        corrector.correct.assert_called_once_with("粗", k=12)
        self.assertEqual(result["hotwords"], [f"词{i}" for i in range(12)])


if __name__ == "__main__":
    unittest.main()

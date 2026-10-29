import unittest
from threading import Lock, Event
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch

from qwen3_asr_vllm.runtime import VLLMContextASR, select_hotwords
from qwen3_asr_vllm.requests import RequestPhonemeCache, RequestVocabularyCache
from qwen3_asr_vllm.transport import EmbeddingCache, pack_embeddings, unpack_embeddings


class HotwordSelectionTest(unittest.TestCase):
    def test_matches_precede_similars_and_are_deduplicated(self):
        result = SimpleNamespace(
            matches=[("甲", "热词甲", 1.0), ("乙", "热词乙", 0.9)],
            similars=[("甲", "热词甲", 0.8), ("丙", "热词丙", 0.8)],
        )
        self.assertEqual(select_hotwords(result, 2), ["热词甲", "热词乙"])
        self.assertEqual(select_hotwords(result, 0), [])


class VLLMRuntimeTest(unittest.TestCase):
    def make_runtime(self, mode, transport="device"):
        runtime = VLLMContextASR.__new__(VLLMContextASR)
        runtime.mode = mode
        runtime.max_audio_samples = 16000
        runtime.prepare_batch_size = 2
        runtime.embedding_transport = transport
        runtime.rag_workers = 4
        runtime._request_cache = RequestVocabularyCache()
        runtime._request_phonemes = RequestPhonemeCache(max_entries=0)
        runtime.max_ctx_hotwords = 8
        runtime.sampling_params = object()
        runtime._lock = Lock()
        runtime._processor = SimpleNamespace(apply_chat_template=Mock(side_effect=lambda messages, **kw: str(messages)))
        runtime._corrector = SimpleNamespace(correct=Mock(return_value=SimpleNamespace(matches=[("错词", "热词", 1)], similars=[])))
        runtime._corrector.correct_batch = Mock(side_effect=lambda texts, k, workers: [
            runtime._corrector.correct(text, k=k) for text in texts
        ])
        features = torch.randn(13, 8)
        next_id = 0

        def prepare(waveforms, mode, cache_on_device):
            nonlocal next_id
            items = []
            for _ in waveforms:
                next_id += 1
                item = {"rough": f"粗转写{next_id}", "audio_output_lengths": [13]}
                if mode != "ctc":
                    if cache_on_device:
                        item["audio_context_ids"] = [next_id]
                    else:
                        item["audio_embeds"] = pack_embeddings(features)
                items.append(item)
            return items

        worker = SimpleNamespace(context_prepare_batch=Mock(side_effect=prepare), context_release=Mock())
        runtime.llm = SimpleNamespace(
            collective_rpc=Mock(side_effect=lambda method, args=(): [getattr(worker, method)(*args)]),
            generate=Mock(side_effect=lambda requests, *a, **kw: [SimpleNamespace(outputs=[SimpleNamespace(text="最终转写")]) for _ in requests]),
        )
        return runtime, worker, features

    def test_rag_batch_passes_only_context_ids_and_releases_them(self):
        runtime, worker, _ = self.make_runtime("ctc_rag")
        waves = [np.zeros(size, dtype=np.float32) for size in (1600, 3200, 4800)]
        outputs = runtime.transcribe_batch(waves)
        self.assertEqual([item["rough"] for item in outputs], ["粗转写1", "粗转写2", "粗转写3"])
        self.assertTrue(all(item["hotwords"] == ["热词"] for item in outputs))
        self.assertEqual(worker.context_prepare_batch.call_count, 2)
        self.assertEqual(runtime._corrector.correct_batch.call_count, 2)
        self.assertEqual(runtime._corrector.correct_batch.call_args_list[0].args[0],
                         ["粗转写1", "粗转写2"])
        self.assertEqual(runtime._corrector.correct_batch.call_args_list[0].kwargs,
                         {"k": 8, "workers": 4})
        sent_audio = worker.context_prepare_batch.call_args_list[0].args[0]
        for data, wave in zip(sent_audio, waves):
            np.testing.assert_array_equal(np.frombuffer(data, dtype=np.float32), wave)
        requests = runtime.llm.generate.call_args.args[0]
        for index, request in enumerate(requests):
            audio = request["multi_modal_data"]["audio"]
            self.assertEqual(set(audio), {"audio_context_ids", "audio_output_lengths"})
            self.assertEqual(audio["audio_context_ids"].tolist(), [index + 1])
        worker.context_release.assert_called_once_with([1, 2, 3])

    def test_cpu_transport_preserves_embeddings(self):
        runtime, worker, features = self.make_runtime("ctc_rag", "cpu")
        runtime.transcribe(np.zeros(1600))
        payload = runtime.llm.generate.call_args.args[0][0]["multi_modal_data"]["audio"]
        torch.testing.assert_close(payload["audio_embeds"], features, atol=0, rtol=0)
        worker.context_release.assert_not_called()

    def test_ctc_mode_does_not_run_text_decoder_or_cache_features(self):
        runtime, worker, _ = self.make_runtime("ctc")
        self.assertEqual(runtime.transcribe(np.zeros(1600)), {"mode": "ctc", "text": "粗转写1"})
        worker.context_prepare_batch.assert_called_once()
        runtime.llm.generate.assert_not_called()
        worker.context_release.assert_not_called()

    def test_asr_mode_uses_manual_hotwords_without_retrieval(self):
        runtime, _, _ = self.make_runtime("asr")
        output = runtime.transcribe(np.zeros(1600), hotwords=["手工热词"])
        self.assertEqual(output, {"mode": "asr", "text": "最终转写"})
        messages = runtime._processor.apply_chat_template.call_args.args[0]
        self.assertEqual(messages[0]["content"], "可能出现的热词：手工热词")
        runtime._corrector.correct.assert_not_called()

    def test_retrieval_overlaps_next_gpu_batch_and_preserves_order(self):
        runtime, worker, _ = self.make_runtime("ctc_rag")
        runtime.prepare_batch_size = 1
        retrieval_started, next_gpu_started = Event(), Event()
        prepare = worker.context_prepare_batch.side_effect

        def gpu(*args):
            if worker.context_prepare_batch.call_count == 2:
                self.assertTrue(retrieval_started.wait(2))
                next_gpu_started.set()
            return prepare(*args)

        def retrieve(text, **kw):
            if text == "粗转写1":
                retrieval_started.set()
                self.assertTrue(next_gpu_started.wait(2))
            return SimpleNamespace(matches=[("", text, 1)], similars=[])

        worker.context_prepare_batch.side_effect = gpu
        runtime._corrector.correct.side_effect = retrieve
        outputs = runtime.transcribe_batch([np.zeros(1600), np.zeros(3200)])
        self.assertEqual([item["hotwords"] for item in outputs], [["粗转写1"], ["粗转写2"]])

    def test_cached_features_are_released_on_retrieval_and_generation_errors(self):
        for stage in ("retrieval", "prompt", "generation", "next_batch"):
            runtime, worker, _ = self.make_runtime("ctc_rag")
            error = RuntimeError("test failure")
            if stage == "retrieval":
                runtime._corrector.correct.side_effect = error
            elif stage == "prompt":
                runtime._processor.apply_chat_template.side_effect = error
            elif stage == "generation":
                runtime.llm.generate.side_effect = error
            else:
                runtime.prepare_batch_size = 1
                original = worker.context_prepare_batch.side_effect
                def fail_next_batch(*args):
                    if worker.context_prepare_batch.call_count > 1:
                        raise error
                    return original(*args)
                worker.context_prepare_batch.side_effect = fail_next_batch
            with self.subTest(stage=stage), self.assertRaisesRegex(RuntimeError, "test failure"):
                runtime.transcribe_batch([np.zeros(1600), np.zeros(3200)])
            self.assertEqual(worker.context_release.call_count, 1)
            self.assertEqual(worker.context_release.call_args.args[0], [1] if stage == "next_batch" else [1, 2])

    def test_invalid_audio_is_rejected_before_worker_execution(self):
        runtime, _, _ = self.make_runtime("ctc_rag")
        for waveform in (np.array([]), np.zeros((2, 2)), np.array([np.nan]), np.zeros(16001)):
            with self.subTest(shape=waveform.shape):
                with self.assertRaises(ValueError):
                    runtime.transcribe(waveform)
        runtime.llm.collective_rpc.assert_not_called()

    def test_empty_batch_does_not_touch_worker(self):
        runtime, _, _ = self.make_runtime("ctc_rag")
        self.assertEqual(runtime.transcribe_batch([]), [])
        runtime.llm.collective_rpc.assert_not_called()

    def test_wrong_retrieval_result_count_releases_contexts(self):
        runtime, worker, _ = self.make_runtime("ctc_rag")
        runtime._corrector.correct_batch.side_effect = None
        runtime._corrector.correct_batch.return_value = []
        with self.assertRaisesRegex(RuntimeError, "Retriever returned"):
            runtime.transcribe_batch([np.zeros(1600), np.zeros(3200)])
        worker.context_release.assert_called_once_with([1, 2])
        runtime.llm.generate.assert_not_called()

    def test_zero_hotword_limit_skips_retrieval(self):
        runtime, _, _ = self.make_runtime("ctc_rag")
        runtime.max_ctx_hotwords = 0
        outputs = runtime.transcribe_batch([np.zeros(1600), np.zeros(3200)])
        runtime._corrector.correct_batch.assert_not_called()
        self.assertEqual([item["hotwords"] for item in outputs], [[], []])

    def test_request_vocabularies_limits_and_cache_are_isolated(self):
        runtime, _, _ = self.make_runtime("ctc_rag")

        def make_corrector(words):
            return SimpleNamespace(correct_batch=Mock(side_effect=lambda texts, k, workers: [
                SimpleNamespace(matches=[], similars=[("", word, 1) for word in words][:k])
                for _ in texts
            ]))

        runtime._create_request_corrector = Mock(side_effect=make_corrector)
        waves = [np.zeros(1600) for _ in range(5)]
        options = [
            {"hotwords": ["甲", "乙"], "max_ctx_hotwords": 1},
            {"hotwords": ["甲", "乙"], "max_ctx_hotwords": 2},
            {"hotwords": ["丙"]},
            {"hotwords": []},
            {},
        ]
        outputs = runtime.transcribe_batch(waves, request_options=options)
        self.assertEqual([item["hotwords"] for item in outputs],
                         [["甲"], ["甲", "乙"], ["丙"], [], ["热词"]])
        self.assertEqual(runtime._create_request_corrector.call_count, 2)
        runtime.transcribe(np.zeros(1600), hotwords=["甲", "乙"])
        self.assertEqual(runtime._create_request_corrector.call_count, 2)
        self.assertEqual(runtime.request_cache_info()["hits"], 1)
        self.assertEqual(runtime.transcribe(np.zeros(1600))["hotwords"], ["热词"])

        shared = runtime.transcribe_batch(
            [np.zeros(1600), np.zeros(1600)], hotwords=["共享"],
            request_options=[{}, {"hotwords": None}],
        )
        self.assertEqual([item["hotwords"] for item in shared], [["共享"], ["热词"]])
        runtime.clear_request_cache()
        self.assertEqual(runtime.request_cache_info()["entries"], 0)
        self.assertEqual(runtime.request_cache_info()["hits"], 0)

    def test_request_generation_limits_do_not_modify_engine_defaults(self):
        runtime, _, _ = self.make_runtime("asr")

        class Sampling:
            def __init__(self, max_tokens):
                self.max_tokens = max_tokens

            def clone(self):
                return Sampling(self.max_tokens)

        runtime.sampling_params = Sampling(512)
        runtime.transcribe_batch([np.zeros(1600), np.zeros(1600)], request_options=[
            {"hotwords": ["甲", "乙"], "max_ctx_hotwords": 1, "max_new_tokens": 8},
            {"hotwords": ["丙"], "max_new_tokens": 12},
        ])
        requests, sampling = runtime.llm.generate.call_args.args
        self.assertIn("甲", requests[0]["prompt"])
        self.assertNotIn("乙", requests[0]["prompt"])
        self.assertIn("丙", requests[1]["prompt"])
        self.assertEqual([value.max_tokens for value in sampling], [8, 12])
        self.assertEqual(runtime.sampling_params.max_tokens, 512)

    def test_invalid_request_options_fail_before_gpu_work(self):
        runtime, _, _ = self.make_runtime("ctc_rag")
        bad = [[], [{"hotwords": "广州"}], [{"hotwords": ["a\nb"]}],
               [{"hotwords": ["a\u2028b"]}], [{1: True}],
               [{"max_ctx_hotwords": -1}], [{"max_new_tokens": True}], [{"unknown": 1}]]
        for options in bad:
            with self.subTest(options=options), self.assertRaises(ValueError):
                runtime.transcribe_batch([np.zeros(1600)], request_options=options)
        runtime.llm.collective_rpc.assert_not_called()

    def test_ctc_rejects_unused_request_parameters(self):
        runtime, _, _ = self.make_runtime("ctc")
        with self.assertRaisesRegex(ValueError, "require asr or ctc_rag"):
            runtime.transcribe(np.zeros(1600), hotwords=["广州"])
        runtime.llm.collective_rpc.assert_not_called()

    def test_reuses_request_vocab_across_microbatches_with_cache_disabled(self):
        runtime, _, _ = self.make_runtime("ctc_rag")
        runtime.prepare_batch_size = 1
        runtime._request_cache = RequestVocabularyCache(max_entries=0)
        corrector = SimpleNamespace(correct_batch=Mock(side_effect=lambda texts, k, workers: [
            SimpleNamespace(matches=[], similars=[("", "甲", 1)]) for _ in texts
        ]))
        runtime._create_request_corrector = Mock(return_value=corrector)
        outputs = runtime.transcribe_batch([np.zeros(1600), np.zeros(1600)], hotwords=["甲"])
        self.assertEqual([item["hotwords"] for item in outputs], [["甲"], ["甲"]])
        runtime._create_request_corrector.assert_called_once_with(("甲",))
        self.assertEqual(runtime.request_cache_info()["entries"], 0)
        runtime.transcribe(np.zeros(1600), hotwords=["甲"])
        self.assertEqual(runtime._create_request_corrector.call_count, 2)

    def test_request_vocabulary_failure_releases_encoded_audio(self):
        runtime, worker, _ = self.make_runtime("ctc_rag")
        runtime._create_request_corrector = Mock(side_effect=RuntimeError("vocabulary failed"))
        with self.assertRaisesRegex(RuntimeError, "vocabulary failed"):
            runtime.transcribe_batch([np.zeros(1600), np.zeros(1600)], hotwords=["甲"])
        worker.context_release.assert_called_once_with([1, 2])
        runtime.llm.generate.assert_not_called()

    def test_invalid_retrieval_workers_fail_before_engine_start(self):
        for workers in (0, -1, True, 1.5):
            with self.subTest(workers=workers), self.assertRaisesRegex(ValueError, "rag_workers"):
                VLLMContextASR("unused", rag_workers=workers)


class RequestVocabularyCacheTest(unittest.TestCase):
    def test_failed_build_preserves_cached_vocabulary(self):
        cache = RequestVocabularyCache(max_entries=2)
        original = cache.get(("a",), lambda words: object())
        with self.assertRaisesRegex(RuntimeError, "failed"):
            cache.get(("b",), Mock(side_effect=RuntimeError("failed")))
        self.assertIs(cache.get(("a",), lambda words: object()), original)
        self.assertEqual(cache.info()["entries"], 1)

    def test_lru_limits_and_oversized_requests(self):
        cache = RequestVocabularyCache(max_entries=2, max_text_bytes=10)
        factory = Mock(side_effect=lambda words: object())
        a = cache.get(("a",), factory)
        cache.get(("b",), factory)
        self.assertIs(cache.get(("a",), factory), a)
        cache.get(("c",), factory)
        cache.get(("b",), factory)
        self.assertEqual(factory.call_count, 4)
        cache.get(("oversized-vocabulary",), factory)
        self.assertEqual(cache.info(), {"entries": 2, "vocabulary_text_bytes": 4,
                                        "hits": 1, "misses": 5})


class EmbeddingCacheTest(unittest.TestCase):
    def test_cache_keeps_tensor_storage_validates_shapes_and_never_reuses_ids(self):
        cache = EmbeddingCache()
        features = torch.randn(13, 8)
        ids = cache.add_batch([features])
        restored = cache.get_batch(ids, [13], 8)[0]
        self.assertEqual(restored.data_ptr(), features.data_ptr())
        self.assertEqual(cache.bytes, features.numel() * features.element_size())
        with self.assertRaises(ValueError):
            cache.get_batch(ids, [12], 8)
        cache.release(ids)
        cache.release(ids)
        self.assertEqual((len(cache), cache.bytes), (0, 0))
        with self.assertRaises(ValueError):
            cache.get_batch(ids, [13], 8)
        self.assertNotEqual(cache.add_batch([features]), ids)

    def test_capacity_failure_is_atomic(self):
        cache = EmbeddingCache(max_bytes=32)
        ids = cache.add_batch([torch.zeros(4)])
        with self.assertRaisesRegex(RuntimeError, "cache is full"):
            cache.add_batch([torch.zeros(4), torch.zeros(4)])
        self.assertEqual((len(cache), cache.bytes), (1, 16))
        cache.release(ids)
        self.assertEqual(cache.bytes, 0)


class EmbeddingTransportTest(unittest.TestCase):
    def test_floating_point_embeddings_round_trip_without_precision_loss(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                tensor = torch.randn(3, 8).to(dtype=dtype).T
                restored = unpack_embeddings(pack_embeddings(tensor))
                torch.testing.assert_close(restored, tensor, atol=0, rtol=0)

    def test_malformed_embedding_payloads_are_rejected(self):
        payload = pack_embeddings(torch.ones(3, 8))
        for invalid in (
            {**payload, "dtype": "int64"},
            {**payload, "shape": [0, 8]},
            {**payload, "data": payload["data"][:-1]},
        ):
            with self.assertRaises(ValueError):
                unpack_embeddings(invalid)


if __name__ == "__main__":
    unittest.main()

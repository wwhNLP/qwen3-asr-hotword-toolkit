"""Synchronous, single-GPU vLLM context-ASR pipeline."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from importlib.metadata import version
from pathlib import Path
from threading import Lock

import numpy as np

from . import ARCHITECTURE, register
from .requests import (RequestPhonemeCache, RequestVocabularyCache, hotword_target,
                       normalize_options, validate_mode_options)
from utils.asr_metrics import transcript_text


def select_hotwords(result, max_hotwords):
    ordered = dict.fromkeys(
        hotword for _, hotword, _ in list(result.matches) + list(result.similars)
    )
    return list(ordered)[:max_hotwords]


def read_audio(path):
    import soundfile as sf
    from math import gcd
    from scipy.signal import resample_poly

    waveform, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    waveform = waveform.mean(axis=1)
    if sample_rate != 16000:
        divisor = gcd(sample_rate, 16000)
        waveform = resample_poly(waveform, 16000 // divisor, sample_rate // divisor)
    return np.ascontiguousarray(waveform, dtype=np.float32)


class VLLMContextASR:
    """Batch neural inference in vLLM; overlap CPU retrieval with GPU preparation.

    Calls are serialized because preparation RPC and generation share one
    engine. transcribe_batch batches encoding, CTC and final decoding.
    This adapter targets vLLM 0.22.1 and an unquantized, merged checkpoint.
    """

    def __init__(
        self,
        model,
        *,
        mode="ctc_rag",
        hot_txt=None,
        rag_threshold=0.85,
        max_ctx_hotwords=10,
        max_new_tokens=512,
        max_audio_seconds=30,
        prepare_batch_size=4,
        embedding_transport="device",
        rag_workers=4,
        rag_refiner="auto",
        request_cache_size=8,
        **engine_kwargs,
    ):
        if mode not in {"asr", "ctc", "ctc_rag"}:
            raise ValueError("mode must be asr, ctc or ctc_rag.")
        if max_ctx_hotwords < 0 or max_new_tokens <= 0 or max_audio_seconds <= 0:
            raise ValueError("Hotword limit must be nonnegative and token/audio limits positive.")
        if not isinstance(prepare_batch_size, int) or prepare_batch_size < 1:
            raise ValueError("prepare_batch_size must be a positive integer.")
        if embedding_transport not in {"device", "cpu"}:
            raise ValueError("embedding_transport must be device or cpu.")
        if type(rag_workers) is not int or rag_workers < 1:
            raise ValueError("rag_workers must be a positive integer.")
        if rag_refiner not in {"auto", "python", "numba"}:
            raise ValueError("rag_refiner must be auto, python or numba.")
        request_cache = RequestVocabularyCache(max_entries=request_cache_size)
        if not 0 <= rag_threshold <= 1:
            raise ValueError("rag_threshold must be between 0 and 1.")
        if version("vllm").split("+")[0] != "0.22.1":
            raise RuntimeError("This adapter requires vllm==0.22.1 (CUDA build suffixes are supported).")
        for name in ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size"):
            if engine_kwargs.get(name, 1) != 1:
                raise ValueError("The context adapter currently supports one vLLM worker/GPU.")
        if engine_kwargs.get("quantization") is not None:
            raise ValueError("Use an unquantized, merged checkpoint for the context adapter.")
        if engine_kwargs.get("worker_extension_cls"):
            raise ValueError("Custom worker extensions are not supported by this adapter.")
        if hot_txt is None:
            hot_txt = Path(__file__).resolve().parents[1] / "hotword" / "hot.txt"
        hotword_text = Path(hot_txt).read_text(encoding="utf-8") if mode == "ctc_rag" else ""
        from vllm import LLM, SamplingParams
        from vllm.transformers_utils.processors.qwen3_asr import Qwen3ASRProcessor

        self._processor = Qwen3ASRProcessor.from_pretrained(
            str(model), revision=engine_kwargs.get("revision"), trust_remote_code=False
        )
        self._corrector = None
        if mode == "ctc_rag":
            from hotword import PhonemeCorrector
            self._corrector = PhonemeCorrector(threshold=rag_threshold, refiner=rag_refiner)
            self._corrector.update_hotwords(hotword_text)
        self.prepare_batch_size = prepare_batch_size
        self.embedding_transport = embedding_transport
        self.rag_workers = rag_workers
        self.rag_refiner = rag_refiner
        self.rag_threshold = rag_threshold
        self._request_cache = request_cache
        self._request_phonemes = RequestPhonemeCache(max_entries=8192 if request_cache_size else 0)
        self.max_ctx_hotwords = max_ctx_hotwords

        register()
        overrides = dict(engine_kwargs.pop("hf_overrides", {}) or {})
        overrides["architectures"] = [ARCHITECTURE]
        engine_kwargs.setdefault("gpu_memory_utilization", 0.8)
        engine_kwargs.setdefault("max_model_len", 4096)
        engine_kwargs.setdefault("max_num_seqs", 16)
        engine_kwargs.setdefault("dtype", "bfloat16")
        engine_kwargs["limit_mm_per_prompt"] = {"audio": 1}
        engine_kwargs["enable_mm_embeds"] = True
        engine_kwargs["worker_extension_cls"] = "qwen3_asr_vllm.worker.ContextWorkerExtension"
        self.llm = LLM(model=str(model), hf_overrides=overrides, **engine_kwargs)
        self.mode = mode
        self.max_audio_samples = int(max_audio_seconds * 16000)
        self.sampling_params = SamplingParams(
            temperature=0,
            max_tokens=max_new_tokens,
            stop_token_ids=[151643, 151645],
        )
        self._lock = Lock()
        self.llm.collective_rpc(
            "context_configure",
            args=(
                {
                    "mode": mode,
                    "hotword_text": "",
                    "threshold": rag_threshold,
                    "max_hotwords": max_ctx_hotwords,
                },
            )
        )

    def transcribe(self, audio, *, hotwords=None, max_ctx_hotwords=None, max_new_tokens=None):
        options = {"max_ctx_hotwords": max_ctx_hotwords, "max_new_tokens": max_new_tokens}
        return self.transcribe_batch([audio], hotwords=hotwords, request_options=[options])[0]

    def transcribe_batch(self, audios, *, hotwords=None, request_options=None):
        """Transcribe a batch with optional shared and per-audio hotword options."""
        waveforms = [read_audio(audio) if isinstance(audio, (str, Path)) else np.asarray(audio, dtype=np.float32) for audio in audios]
        options = normalize_options(len(waveforms), hotwords, request_options)
        validate_mode_options(self.mode, options)
        for waveform in waveforms:
            if waveform.ndim != 1 or not len(waveform) or not np.isfinite(waveform).all():
                raise ValueError("Audio must be a nonempty, finite, mono waveform at 16 kHz.")
            if len(waveform) > self.max_audio_samples:
                raise ValueError("Audio exceeds max_audio_seconds; split it into shorter segments first.")
        if not waveforms:
            return []
        with self._lock:
            prepared = []
            cached_ids = []
            # Reuse within this batch even when cross-call caching is disabled,
            # without retaining an unbounded number of request vocabularies.
            request_correctors = RequestVocabularyCache()
            try:
                # Only the main thread touches vLLM. While the worker encodes
                # chunk N+1, this CPU thread retrieves hotwords for chunk N.
                overlap = self.mode == "ctc_rag" and len(waveforms) > self.prepare_batch_size
                executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="asr-rag") if overlap else nullcontext()
                with executor as pool:
                    pending = []
                    for offset in range(0, len(waveforms), self.prepare_batch_size):
                        chunk = waveforms[offset:offset + self.prepare_batch_size]
                        (items,) = self.llm.collective_rpc(
                            "context_prepare_batch",
                            args=([waveform.tobytes() for waveform in chunk], self.mode,
                                  self.embedding_transport == "device"),
                        )
                        cached_ids.extend(key for item in items for key in item.get("audio_context_ids", []))
                        if len(items) != len(chunk):
                            raise RuntimeError("Worker returned an unexpected number of audio contexts.")
                        if pool is None:
                            prepared.extend(self._finish_context_batch(
                                items, options[offset:offset + len(chunk)], request_correctors
                            ))
                        else:
                            pending.append(pool.submit(self._finish_context_batch, items,
                                                       options[offset:offset + len(chunk)], request_correctors))
                    for future in pending:
                        prepared.extend(future.result())
                results = []
                if self.mode != "ctc":
                    requests = [self._generation_request(item) for item in prepared]
                    sampling = self._sampling_for_requests(options)
                    results = self.llm.generate(requests, sampling, use_tqdm=False)
                    if len(results) != len(prepared):
                        raise RuntimeError("Decoder returned an unexpected number of transcriptions.")
                outputs = []
                for index, item in enumerate(prepared):
                    text = item["rough"] if self.mode == "ctc" else results[index].outputs[0].text
                    output = {"mode": self.mode, "text": transcript_text(text)}
                    if self.mode == "ctc_rag":
                        output.update(rough=item["rough"], hotwords=item["hotwords"])
                    outputs.append(output)
                return outputs
            finally:
                # Also release on retrieval, prompt processing or decoding errors.
                if cached_ids:
                    self.llm.collective_rpc("context_release", args=(cached_ids,))

    def _create_request_corrector(self, hotwords):
        from hotword import PhonemeCorrector

        corrector = PhonemeCorrector(threshold=self.rag_threshold, refiner=self.rag_refiner)
        corrector.update_hotword_entries(hotwords, phoneme_lookup=self._request_phonemes.get)
        return corrector

    def _finish_context_batch(self, items, options, request_correctors):
        selected = [[] for _ in items]
        if self.mode == "ctc_rag":
            groups = {}
            for index, option in enumerate(options):
                limit = self.max_ctx_hotwords if option.max_ctx_hotwords is None else option.max_ctx_hotwords
                if limit and option.hotwords != ():
                    groups.setdefault(option.hotwords, []).append((index, limit))
            for words, members in groups.items():
                if words is None:
                    corrector = self._corrector
                else:
                    corrector = request_correctors.get(
                        words, lambda vocab: self._request_cache.get(vocab, self._create_request_corrector)
                    )
                retrieval = corrector.correct_batch(
                    [items[index]["rough"] for index, _ in members],
                    k=max(limit for _, limit in members), workers=self.rag_workers,
                )
                if len(retrieval) != len(members):
                    raise RuntimeError("Retriever returned an unexpected number of results.")
                for (index, limit), result in zip(members, retrieval):
                    selected[index] = select_hotwords(result, limit)
        for index, item in enumerate(items):
            if self.mode == "ctc_rag":
                hotwords = selected[index]
            else:
                hotwords = [target for entry in options[index].hotwords or ()
                            if (target := hotword_target(entry)) is not None]
                if options[index].max_ctx_hotwords is not None:
                    hotwords = hotwords[:options[index].max_ctx_hotwords]
            item["hotwords"] = hotwords
            if self.mode != "ctc":
                system = "可能出现的热词：" + "，".join(hotwords) if hotwords else ""
                item["prompt"] = self._processor.apply_chat_template([
                    {"role": "system", "content": system},
                    {"role": "user", "content": [{"type": "audio", "audio": ""}]},
                ], add_generation_prompt=True, tokenize=False)
        return items

    def _sampling_for_requests(self, options):
        if all(option.max_new_tokens is None for option in options):
            return self.sampling_params
        # Per-request clones prevent one limit from changing another request.
        sampling = []
        for option in options:
            if option.max_new_tokens is None:
                sampling.append(self.sampling_params)
            else:
                params = self.sampling_params.clone()
                params.max_tokens = option.max_new_tokens
                sampling.append(params)
        return sampling

    def request_cache_info(self):
        """Return counts and cached vocabulary text bytes, excluding object overhead."""
        with self._lock:
            return {**self._request_cache.info(), "phonemes": self._request_phonemes.info()}

    def clear_request_cache(self):
        """Discard cached request vocabularies and their reusable phonemes."""
        with self._lock:
            self._request_cache.clear()
            self._request_phonemes.clear()

    @staticmethod
    def _generation_request(item):
        import torch
        from .transport import unpack_embeddings

        audio = {"audio_output_lengths": torch.tensor(item["audio_output_lengths"], dtype=torch.long)}
        if "audio_context_ids" in item:
            audio["audio_context_ids"] = torch.tensor(item["audio_context_ids"], dtype=torch.long)
        else:
            audio["audio_embeds"] = unpack_embeddings(item["audio_embeds"])
        return {"prompt": item["prompt"], "multi_modal_data": {"audio": audio}}

    def runtime_stats(self):
        with self._lock:
            (stats,) = self.llm.collective_rpc("context_stats")
            return stats

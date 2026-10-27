"""Synchronous, single-GPU vLLM context-ASR pipeline."""

from importlib.metadata import version
from pathlib import Path
from threading import Lock

import numpy as np

from . import ARCHITECTURE, register
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
    """Keep audio encoding, CTC, phoneme retrieval and decoding in vLLM.

    Calls are serialized because preparation RPC and generation share one
    engine. Use transcribe_batch to batch final decoding across utterances.
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
        **engine_kwargs,
    ):
        if mode not in {"asr", "ctc", "ctc_rag"}:
            raise ValueError("mode must be asr, ctc or ctc_rag.")
        if max_ctx_hotwords < 0 or max_new_tokens <= 0 or max_audio_seconds <= 0:
            raise ValueError("Hotword limit must be nonnegative and token/audio limits positive.")
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
                    "hotword_text": hotword_text,
                    "threshold": rag_threshold,
                    "max_hotwords": max_ctx_hotwords,
                },
            )
        )

    def transcribe(self, audio, *, hotwords=None):
        return self.transcribe_batch([audio], hotwords=hotwords)[0]

    def transcribe_batch(self, audios, *, hotwords=None):
        waveforms = [read_audio(audio) if isinstance(audio, (str, Path)) else np.asarray(audio, dtype=np.float32) for audio in audios]
        for waveform in waveforms:
            if waveform.ndim != 1 or not len(waveform) or not np.isfinite(waveform).all():
                raise ValueError("Audio must be a nonempty, finite, mono waveform at 16 kHz.")
            if len(waveform) > self.max_audio_samples:
                raise ValueError("Audio exceeds max_audio_seconds; split it into shorter segments first.")
        if not waveforms:
            return []
        with self._lock:
            prepared = []
            for waveform in waveforms:
                (item,) = self.llm.collective_rpc(
                    "context_prepare", args=(waveform.tobytes(), self.mode, list(hotwords or []))
                )
                if self.mode != "ctc":
                    import torch
                    from .transport import unpack_embeddings

                    item = dict(item)
                    item["audio_embeds"] = unpack_embeddings(item["audio_embeds"])
                    item["audio_output_lengths"] = torch.tensor(item["audio_output_lengths"], dtype=torch.long)
                prepared.append(item)
            results = []
            if self.mode != "ctc":
                requests = [
                    {
                        "prompt": item["prompt"],
                        "multi_modal_data": {
                            "audio": {
                                "audio_embeds": item["audio_embeds"],
                                "audio_output_lengths": item["audio_output_lengths"],
                            }
                        },
                    }
                    for item in prepared
                ]
                results = self.llm.generate(requests, self.sampling_params, use_tqdm=False)
            outputs = []
            for index, item in enumerate(prepared):
                text = item["rough"] if self.mode == "ctc" else results[index].outputs[0].text
                output = {"mode": self.mode, "text": transcript_text(text)}
                if self.mode == "ctc_rag":
                    output.update(rough=item["rough"], hotwords=item["hotwords"])
                outputs.append(output)
            return outputs

    def runtime_stats(self):
        with self._lock:
            (stats,) = self.llm.collective_rpc("context_stats")
            return stats

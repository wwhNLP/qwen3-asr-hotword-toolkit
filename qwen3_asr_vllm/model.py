"""Native vLLM audio + CTC + RAG model, with precomputed audio input support.

The public runtime uses named worker RPCs for preparation and LLM.generate for
decoding. Both run on the same vLLM model instance. Audio embeddings can stay in
worker-local GPU storage; only IDs cross the process boundary. No separate
Transformers inference model is loaded.
"""

import copy
from collections.abc import Mapping
from itertools import accumulate

import torch
from torch import nn
from vllm.forward_context import set_forward_context
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.qwen3_asr import (
    Qwen3ASRDummyInputsBuilder,
    Qwen3ASRForConditionalGeneration,
    Qwen3ASRMultiModalDataParser,
    Qwen3ASRMultiModalProcessor,
    Qwen3ASRProcessingInfo,
)
from vllm.model_executor.models.qwen3_omni_moe_thinker import (
    Qwen3OmniMoeAudioEncoderLayer,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig
from vllm.multimodal.parse import DictEmbeddingItems
from vllm.multimodal.processing import PromptReplacement
from vllm.transformers_utils.processors.qwen3_asr import Qwen3ASRProcessor


def embedding_fields(data):
    if "audio_context_ids" in data:
        return {
            "audio_context_ids": MultiModalFieldConfig.batched("audio", keep_on_cpu=True),
            "audio_output_lengths": MultiModalFieldConfig.batched("audio", keep_on_cpu=True),
        }
    return {
        "audio_embeds": MultiModalFieldConfig.flat_from_sizes(
            "audio", data["audio_output_lengths"], dim=0
        ),
        "audio_output_lengths": MultiModalFieldConfig.batched("audio"),
    }


class ContextAudioParser(Qwen3ASRMultiModalDataParser):
    def _parse_audio_data(self, data):
        if isinstance(data, Mapping) and ("audio_embeds" in data or "audio_context_ids" in data):
            embedding_key = "audio_context_ids" if "audio_context_ids" in data else "audio_embeds"
            return DictEmbeddingItems(
                data,
                modality="audio",
                required_fields={embedding_key, "audio_output_lengths"},
                fields_factory=embedding_fields,
            )
        return super()._parse_audio_data(data)


class ContextProcessingInfo(Qwen3ASRProcessingInfo):
    def get_data_parser(self):
        return ContextAudioParser(
            target_sr=self.get_feature_extractor().sampling_rate,
            expected_hidden_size=self._get_expected_hidden_size(),
        )


class ContextMultiModalProcessor(Qwen3ASRMultiModalProcessor):
    def _call_hf_processor(self, prompt, mm_data, mm_kwargs, tok_kwargs):
        if not mm_data:
            # An embedding request needs text tokenization only. Calling the
            # audio processor here would try to expand a token with no waveform.
            return self.info.get_tokenizer()([prompt], **tok_kwargs)
        return super()._call_hf_processor(prompt, mm_data, mm_kwargs, tok_kwargs)

    def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
        if "audio_embeds" in hf_inputs or "audio_context_ids" in hf_inputs:
            return embedding_fields(hf_inputs)
        return super()._get_mm_fields_config(hf_inputs, hf_processor_mm_kwargs)

    def _get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
        data = out_mm_kwargs.get_data()
        if "audio_output_lengths" not in data:
            return super()._get_prompt_updates(
                mm_items, hf_processor_mm_kwargs, out_mm_kwargs
            )
        lengths = data["audio_output_lengths"].reshape(-1).tolist()
        if any(length <= 0 for length in lengths):
            raise ValueError("Precomputed audio embeddings must contain at least one frame.")
        audio_token = self.info.get_hf_processor().audio_token
        audio_token_id = self.info.get_tokenizer().convert_tokens_to_ids(audio_token)
        return [
            PromptReplacement(
                modality="audio",
                target=audio_token,
                replacement=lambda index: [audio_token_id] * lengths[index],
            )
        ]


class VLLMCTCBranch(nn.Module):
    """CTC branch using vLLM's parallel linear and encoder attention layers."""

    def __init__(self, config, prefix="ctc_branch"):
        super().__init__()
        layer_config = copy.deepcopy(config.audio_config)
        layer_config.d_model = config.audio_config.output_dim
        layer_config.encoder_attention_heads = config.ctc_num_heads
        layer_config.encoder_ffn_dim = config.ctc_ffn_dim
        self.layers = nn.ModuleList(
            Qwen3OmniMoeAudioEncoderLayer(layer_config, prefix=f"{prefix}.layers.{index}")
            for index in range(config.ctc_num_layers)
        )
        self.norm = nn.LayerNorm(layer_config.d_model)
        self.ctc_head = nn.Linear(layer_config.d_model, config.ctc_vocab_size)
        self.blank_id = config.ctc_blank_id
        self.vocab_size = config.ctc_vocab_size

    def forward(self, hidden_states, cu_seqlens, max_seqlen=None):
        if max_seqlen is None:
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
        for layer in self.layers:
            hidden_states = layer(hidden_states, cu_seqlens, max_seqlen)
        return self.ctc_head(self.norm(hidden_states))

    def load_weights(self, weights):
        params = dict(self.named_parameters())
        loaded = set()
        for name, weight in weights:
            for projection, shard in (("q_proj", "q"), ("k_proj", "k"), ("v_proj", "v")):
                source = f"self_attn.{projection}."
                if source in name:
                    name = name.replace(source, "self_attn.qkv.")
                    param = params[name]
                    param.weight_loader(param, weight, shard)
                    break
            else:
                param = params[name]
                getattr(param, "weight_loader", default_weight_loader)(param, weight)
            loaded.add(name)
        return loaded


@MULTIMODAL_REGISTRY.register_processor(
    ContextMultiModalProcessor,
    info=ContextProcessingInfo,
    dummy_inputs=Qwen3ASRDummyInputsBuilder,
)
class Qwen3ASRContextForConditionalGeneration(Qwen3ASRForConditionalGeneration):
    def __init__(self, *, vllm_config, prefix=""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.ctc_branch = (
            VLLMCTCBranch(self.config, prefix=f"{prefix}.ctc_branch".lstrip("."))
            if getattr(self.config, "add_ctc_branch", False)
            else None
        )
        self._context_processor = None
        self._context_vocab = None
        self._context_corrector = None
        self._max_ctx_hotwords = 10
        self._encoder_forward_calls = 0
        self._encoder_batches = 0
        self._ctc_forward_calls = 0
        from .transport import EmbeddingCache
        self._context_cache = EmbeddingCache()

    def configure_context(self, *, mode, hotword_text, threshold, max_hotwords):
        if mode != "asr":
            if self.ctc_branch is None:
                raise ValueError("ctc/ctc_rag requires a checkpoint with thinker_config.add_ctc_branch=True.")
            from utils.paraformer_vocab import get_vocab

            self._context_vocab = get_vocab()
            if self._context_vocab.vocab_size != self.ctc_branch.vocab_size:
                raise ValueError("The checkpoint CTC vocabulary does not match Paraformer vocab8404.")
            if self._context_vocab.blank_id != self.ctc_branch.blank_id:
                raise ValueError("The checkpoint CTC blank ID does not match the vocabulary.")
        if mode == "ctc_rag":
            from hotword import PhonemeCorrector

            self._context_corrector = PhonemeCorrector(threshold=threshold)
            self._context_corrector.update_hotwords(hotword_text)
        self._max_ctx_hotwords = max_hotwords

    def _process_audio_input(self, audio_input):
        self._encoder_forward_calls += len(audio_input["audio_feature_lengths"])
        self._encoder_batches += 1
        return super()._process_audio_input(audio_input)

    def prepare_context(self, waveform, *, mode, hotwords):
        """Compatibility RPC for clients using the original CPU payload path."""
        from .runtime import select_hotwords

        (item,) = self.prepare_context_batch([waveform], mode=mode, cache_on_device=False)
        if mode == "ctc_rag":
            result = self._context_corrector.correct(item["rough"], k=self._max_ctx_hotwords)
            hotwords = select_hotwords(result, self._max_ctx_hotwords)
        system = "可能出现的热词：" + "，".join(hotwords) if hotwords else ""
        item["hotwords"] = hotwords
        item["prompt"] = self._context_processor.apply_chat_template([
            {"role": "system", "content": system},
            {"role": "user", "content": [{"type": "audio", "audio": ""}]},
        ], add_generation_prompt=True, tokenize=False)
        return item

    @torch.inference_mode()
    def prepare_context_batch(self, waveforms, *, mode, cache_on_device=True):
        """One packed encoder/CTC forward for variable-length utterances."""
        from .transport import pack_embeddings

        if mode not in {"asr", "ctc", "ctc_rag"}:
            raise ValueError("Unknown context mode.")
        if not waveforms:
            return []
        if self._context_processor is None:
            model_config = self.vllm_config.model_config
            self._context_processor = Qwen3ASRProcessor.from_pretrained(
                model_config.model, revision=model_config.revision, trust_remote_code=False
            )
        features = self._context_processor.feature_extractor(
            waveforms, sampling_rate=16000, return_tensors="pt", padding=True,
            truncation=False, return_attention_mask=True,
        )
        # Strip padding before concatenation: the native encoder expects packed
        # mel features plus each utterance's true length.
        frame_lengths = features["attention_mask"].sum(-1).tolist()
        if any(length <= 0 for length in frame_lengths):
            raise ValueError("Audio features must contain at least one frame.")
        device, dtype = self.audio_tower.device, self.audio_tower.dtype
        mel = torch.cat([
            features["input_features"][index, :, :length]
            for index, length in enumerate(frame_lengths)
        ], dim=1).to(device=device, dtype=dtype)
        lengths = torch.tensor(frame_lengths, device=device, dtype=torch.long)
        with set_forward_context(None, self.vllm_config):
            embeddings = self._process_audio_input(
                {"input_features": mel, "audio_feature_lengths": lengths}
            )
            output_lengths = [len(item) for item in embeddings]
            rough = [""] * len(embeddings)
            if mode != "asr":
                if self.ctc_branch is None or self._context_vocab is None:
                    raise RuntimeError("CTC resources were not configured on the vLLM worker.")
                # Sequence boundaries prevent CTC attention crossing utterances.
                cu_seqlens = torch.tensor(list(accumulate([0, *output_lengths])), device=device, dtype=torch.int32)
                # vLLM's attention wrapper calls max_seqlen.item() per layer.
                # The length is already known on the CPU; keeping this scalar
                # there avoids a GPU synchronization for each CTC layer.
                max_seqlen = torch.tensor(max(output_lengths), dtype=torch.int32, device="cpu")
                logits = self.ctc_branch(torch.cat(embeddings, dim=0), cu_seqlens, max_seqlen)
                self._ctc_forward_calls += 1
                frame_ids = logits.argmax(-1).cpu().tolist()
                offset = 0
                for index, length in enumerate(output_lengths):
                    rough[index] = self._context_vocab.ctc_greedy_decode(frame_ids[offset:offset + length])
                    offset += length
        prepared = [{"rough": text, "audio_output_lengths": [length]}
                    for text, length in zip(rough, output_lengths)]
        if mode != "ctc":
            if cache_on_device:
                for item, key in zip(prepared, self._context_cache.add_batch(embeddings)):
                    item["audio_context_ids"] = [key]
            else:
                for item, tensor in zip(prepared, embeddings):
                    item["audio_embeds"] = pack_embeddings(tensor)
        return prepared

    def release_context(self, ids):
        self._context_cache.release(ids)

    def embed_multimodal(self, **kwargs):
        ids = kwargs.get("audio_context_ids")
        if ids is not None:
            return self._context_cache.get_batch(
                ids.reshape(-1).tolist(),
                kwargs["audio_output_lengths"].reshape(-1).tolist(),
                self.config.text_config.hidden_size,
            )
        embeddings = kwargs.get("audio_embeds")
        if embeddings is None:
            return super().embed_multimodal(**kwargs)
        lengths = kwargs["audio_output_lengths"].reshape(-1).tolist()
        if isinstance(embeddings, torch.Tensor):
            if embeddings.ndim == 2:
                embeddings = embeddings.split(lengths, dim=0)
            elif embeddings.ndim == 3:
                embeddings = embeddings.unbind(0)
            else:
                raise ValueError("audio_embeds must be a packed 2D tensor or a batched 3D tensor.")
        embeddings = tuple(embeddings)
        if len(embeddings) != len(lengths) or any(
            embedding.shape != (length, self.config.text_config.hidden_size)
            for embedding, length in zip(embeddings, lengths)
        ):
            raise ValueError("Audio embedding lengths or hidden size do not match the model.")
        return embeddings

    def get_mrope_input_positions(self, input_tokens, mm_features):
        # Qwen3-ASR uses the same monotonically increasing positions on all
        # three RoPE axes; embedding requests already contain expanded tokens.
        positions = torch.arange(len(input_tokens)).view(1, -1).expand(3, -1)
        return positions.clone(), 0

    def context_runtime_stats(self):
        return {
            "encoder_forward_calls": self._encoder_forward_calls,
            "encoder_batches": self._encoder_batches,
            "ctc_forward_calls": self._ctc_forward_calls,
            "context_cache_items": len(self._context_cache),
            "context_cache_bytes": self._context_cache.bytes,
        }

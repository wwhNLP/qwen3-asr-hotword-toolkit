"""Native vLLM audio + CTC + RAG model, with precomputed audio input support.

The public runtime uses named worker RPCs for preparation and LLM.generate for
decoding. Both run on the same vLLM model instance. Only CPU audio embeddings
cross the process boundary; no separate Transformers inference model is loaded.
"""

import copy
from collections.abc import Mapping

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
    return {
        "audio_embeds": MultiModalFieldConfig.flat_from_sizes(
            "audio", data["audio_output_lengths"], dim=0
        ),
        "audio_output_lengths": MultiModalFieldConfig.batched("audio"),
    }


class ContextAudioParser(Qwen3ASRMultiModalDataParser):
    def _parse_audio_data(self, data):
        if isinstance(data, Mapping) and "audio_embeds" in data:
            return DictEmbeddingItems(
                data,
                modality="audio",
                required_fields={"audio_embeds", "audio_output_lengths"},
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
        if "audio_embeds" in hf_inputs:
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

    def forward(self, hidden_states, cu_seqlens):
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
        self._encoder_forward_calls += 1
        return super()._process_audio_input(audio_input)

    @torch.inference_mode()
    def prepare_context(self, waveform, *, mode, hotwords):
        from .runtime import select_hotwords
        from .transport import pack_embeddings

        if self._context_processor is None:
            model_config = self.vllm_config.model_config
            self._context_processor = Qwen3ASRProcessor.from_pretrained(
                model_config.model, revision=model_config.revision, trust_remote_code=False
            )
        processor = self._context_processor
        features = processor.feature_extractor(
            waveform,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            truncation=False,
            return_attention_mask=True,
        )
        device, dtype = self.audio_tower.device, self.audio_tower.dtype
        lengths = features["attention_mask"].sum(-1).to(device=device, dtype=torch.long)
        mel = features["input_features"][0, :, : int(lengths[0])].to(device=device, dtype=dtype)
        with set_forward_context(None, self.vllm_config):
            (audio_embeds,) = self._process_audio_input(
                {"input_features": mel, "audio_feature_lengths": lengths}
            )
            rough = ""
            if mode != "asr":
                if self.ctc_branch is None or self._context_vocab is None:
                    raise RuntimeError("CTC resources were not configured on the vLLM worker.")
                cu_seqlens = torch.tensor([0, len(audio_embeds)], device=device, dtype=torch.int32)
                frame_ids = self.ctc_branch(audio_embeds, cu_seqlens).argmax(-1).tolist()
                rough = self._context_vocab.ctc_greedy_decode(frame_ids)
        if mode == "ctc_rag":
            result = self._context_corrector.correct(rough)
            hotwords = select_hotwords(result, self._max_ctx_hotwords)
        system = "可能出现的热词：" + "，".join(hotwords) if hotwords else ""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": [{"type": "audio", "audio": ""}]},
        ]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        prepared = {"rough": rough, "hotwords": hotwords, "prompt": prompt}
        if mode != "ctc":
            prepared["audio_embeds"] = pack_embeddings(audio_embeds)
            prepared["audio_output_lengths"] = [len(audio_embeds)]
        return prepared

    def embed_multimodal(self, **kwargs):
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
        return {"encoder_forward_calls": self._encoder_forward_calls}

"""ms-swift 4.4.2 external plugin: local Qwen3-ASR + trainable CTC branch.

Use --external_plugins training/swift_plugin.py and explicitly select
--model_type qwen3_asr_context_sft or qwen3_asr_context_ctc.
"""

from functools import lru_cache
from pathlib import Path
import sys

import torch
from torch.nn.utils.rnn import pad_sequence

# ms-swift imports external plugins by file path, including in worker processes.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swift.model import Model, ModelGroup, ModelLoader, ModelMeta, register_model
from swift.model.utils import use_submodel_func
from swift.template import register_template
from swift.template.templates.qwen import Qwen3ASRTemplate, QwenTemplateMeta

from qwen3_asr_model.configuration_qwen3_asr import Qwen3ASRConfig
from qwen3_asr_model.modeling_qwen3_asr import Qwen3ASRForConditionalGeneration
from qwen3_asr_model.processing_qwen3_asr import Qwen3ASRProcessor, _get_feat_extract_output_lengths
from utils.paraformer_vocab import ParaformerVocab


@lru_cache(maxsize=1)
def ctc_vocab():
    return ParaformerVocab()


class ContextLoader(ModelLoader):
    ctc_training = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.auto_config_cls = Qwen3ASRConfig
        self.auto_model_cls = Qwen3ASRForConditionalGeneration
        self.auto_tokenizer_cls = Qwen3ASRProcessor

    def get_config(self, model_dir):
        config = super().get_config(model_dir)
        thinker = config.thinker_config
        thinker.ctc_only = self.ctc_training
        if self.ctc_training:
            thinker.add_ctc_branch = True
            if thinker.ctc_vocab_size != ctc_vocab().vocab_size or thinker.ctc_blank_id != ctc_vocab().blank_id:
                raise ValueError("CTC config must match the bundled Paraformer vocabulary.")
        elif thinker.add_ctc_branch:
            raise ValueError("Run hotword SFT before attaching/training CTC. Use a base/SFT checkpoint without CTC.")
        thinker.text_config.use_cache = False
        return config

    def get_model(self, model_dir, config, processor, model_kwargs):
        model = super().get_model(model_dir, config, processor, model_kwargs)
        # Delegate generation/checkpointing as upstream does, keeping the outer
        # forward intact: swift otherwise re-infers label_names from the patched
        # CTC forward and incorrectly requires CTC targets during SFT evaluation.
        use_submodel_func(model, "thinker", func_list=[
            "generate", "get_input_embeddings", "gradient_checkpointing_enable",
        ])
        return model


class CTCLoader(ContextLoader):
    ctc_training = True


class ContextSFTTemplate(Qwen3ASRTemplate):
    support_padding_free = False


class ContextCTCTemplate(ContextSFTTemplate):
    model_accepts_loss_kwargs = False

    def _encode(self, inputs):
        if len(inputs.audios) != 1 or len(inputs.messages) != 2 or inputs.messages[-1]["role"] != "assistant":
            raise ValueError("CTC training requires one audio, one user turn, and one reference assistant turn.")
        target = inputs.messages[-1]["content"]
        ids = ctc_vocab().encode_text(target, strip_punct=True)
        if not ids or ctc_vocab().unk_id in ids:
            raise ValueError(f"Empty/unknown CTC target; normalize the transcript first: {target!r}")
        encoded = super()._encode(inputs)
        # Reject unalignable samples rather than silently training on zero_infinity loss.
        mel_length = int(encoded["feature_attention_mask"].sum())
        frame_length = int(_get_feat_extract_output_lengths(torch.tensor(mel_length)))
        minimum = len(ids) + sum(a == b for a, b in zip(ids, ids[1:]))
        if minimum > frame_length:
            raise ValueError(f"CTC target requires {minimum} frames but audio has {frame_length}: {target!r}")
        encoded["ctc_labels"] = ids
        return encoded

    def _data_collator_mm_data(self, batch):
        result = super()._data_collator_mm_data(batch)
        targets = [torch.tensor(row["ctc_labels"], dtype=torch.long) for row in batch]
        result["ctc_labels"] = pad_sequence(targets, batch_first=True, padding_value=ctc_vocab().blank_id)
        result["ctc_label_lengths"] = torch.tensor([len(x) for x in targets], dtype=torch.long)
        return result

    def compute_sft_loss(self, model, inputs, num_items_in_batch=None, trainer=None):
        outputs = model(**inputs)
        # CTC is a per-example loss; LM token-count rescaling/accuracy is inapplicable.
        outputs.logits = None
        return outputs


for stage, loader, template in (
    ("sft", ContextLoader, ContextSFTTemplate),
    ("ctc", CTCLoader, ContextCTCTemplate),
):
    name = f"qwen3_asr_context_{stage}"
    register_template(QwenTemplateMeta(
        name, template_cls=template, system_prefix=None, default_system=None,
        prefix=["<|im_start|>system\n{{SYSTEM}}<|im_end|>\n"],
    ), exist_ok=True)
    register_model(ModelMeta(
        name,
        [ModelGroup([Model("Qwen/Qwen3-ASR-0.6B", "Qwen/Qwen3-ASR-0.6B")], name)],
        loader,
        model_arch="qwen3_asr",
        is_multimodal=True,
        requires=["transformers==4.57.6"],
        tags=["audio"],
    ), exist_ok=True)

"""vLLM plugin; importing this package does not load a Transformers model."""

ARCHITECTURE = "Qwen3ASRContextForConditionalGeneration"


def register():
    from vllm import ModelRegistry

    if ARCHITECTURE not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            ARCHITECTURE, f"qwen3_asr_vllm.model:{ARCHITECTURE}"
        )


def __getattr__(name):
    if name == "VLLMContextASR":
        from .runtime import VLLMContextASR

        return VLLMContextASR
    raise AttributeError(name)

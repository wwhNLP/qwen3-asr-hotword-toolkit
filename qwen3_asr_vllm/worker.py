"""Named vLLM worker RPCs: no callable/pickle serialization is required."""


class ContextWorkerExtension:
    def context_configure(self, settings):
        return self.get_model().configure_context(**settings)

    def context_prepare(self, waveform_bytes, mode, hotwords):
        import numpy as np

        waveform = np.frombuffer(waveform_bytes, dtype=np.float32).copy()
        return self.get_model().prepare_context(waveform, mode=mode, hotwords=hotwords)

    def context_stats(self):
        return self.get_model().context_runtime_stats()

    def context_ctc_weight_digests(self):
        """Diagnostic hashes in checkpoint naming, including unfused Q/K/V."""
        import hashlib

        import torch

        branch = self.get_model().ctc_branch
        if branch is None:
            return {}
        digests = {}

        def record(name, parameter):
            data = parameter.detach().cpu().contiguous()
            digests[name] = {
                "dtype": str(data.dtype).removeprefix("torch."),
                "sha256": hashlib.sha256(data.view(torch.uint8).numpy().tobytes()).hexdigest(),
            }

        for name, parameter in branch.named_parameters():
            if "self_attn.qkv." in name:
                for projection, shard in zip(("q_proj", "k_proj", "v_proj"), parameter.chunk(3, dim=0)):
                    record(name.replace("self_attn.qkv.", f"self_attn.{projection}."), shard)
            else:
                record(name, parameter)
        return digests

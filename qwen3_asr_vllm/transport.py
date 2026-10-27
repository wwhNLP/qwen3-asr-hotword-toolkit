"""Lossless tensor payloads for vLLM's untyped utility RPC channel."""


def pack_embeddings(tensor):
    import torch

    tensor = tensor.detach().cpu().contiguous()
    return {
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": list(tensor.shape),
        "data": tensor.view(torch.uint8).numpy().tobytes(),
    }


def unpack_embeddings(payload):
    import torch

    if payload["dtype"] not in {"float16", "bfloat16", "float32"}:
        raise ValueError("Unsupported audio embedding dtype.")
    shape = payload["shape"]
    if len(shape) != 2 or any(not isinstance(size, int) or size <= 0 for size in shape):
        raise ValueError("Audio embeddings must have a nonempty (frames, hidden_size) shape.")
    dtype = getattr(torch, payload["dtype"])
    expected_bytes = shape[0] * shape[1] * torch.empty((), dtype=dtype).element_size()
    if len(payload["data"]) != expected_bytes:
        raise ValueError("Audio embedding byte length does not match the declared shape.")
    return torch.frombuffer(bytearray(payload["data"]), dtype=dtype).reshape(shape)

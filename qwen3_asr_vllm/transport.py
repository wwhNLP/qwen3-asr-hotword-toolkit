"""Worker-local embeddings and lossless CPU payloads for utility RPCs."""


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


class EmbeddingCache:
    """Worker-local tensor storage, bounded and explicitly released per request.

    IDs never repeat during the worker lifetime, including after release. This
    prevents vLLM's multimodal/prefix caches from confusing unrelated audio.
    """

    def __init__(self, max_bytes=256 * 1024 * 1024):
        self.max_bytes = max_bytes
        self.bytes = 0
        self._next_id = 1
        self._items = {}

    def add_batch(self, tensors):
        sizes = [tensor.numel() * tensor.element_size() for tensor in tensors]
        if self.bytes + sum(sizes) > self.max_bytes:
            raise RuntimeError("Audio context cache is full; reduce batch size or use embedding_transport='cpu'.")
        ids = []
        for tensor, size in zip(tensors, sizes):
            key = self._next_id
            self._next_id += 1
            self._items[key] = tensor.detach()
            self.bytes += size
            ids.append(key)
        return ids

    def get_batch(self, ids, lengths, hidden_size):
        if len(ids) != len(lengths):
            raise ValueError("Audio context IDs and lengths must have the same size.")
        result = []
        for key, length in zip(ids, lengths):
            tensor = self._items.get(key)
            if tensor is None:
                raise ValueError(f"Unknown or released audio context ID: {key}.")
            if tuple(tensor.shape) != (length, hidden_size):
                raise ValueError("Cached audio embedding length or hidden size does not match.")
            result.append(tensor)
        return tuple(result)

    def release(self, ids):
        for key in ids:
            tensor = self._items.pop(key, None)
            if tensor is not None:
                self.bytes -= tensor.numel() * tensor.element_size()

    def __len__(self):
        return len(self._items)

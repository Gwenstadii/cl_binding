"""Hash exact tensor storage bytes, including BF16, without numerical conversion."""

from hashlib import sha256


def tensors_hash(values):
    import torch
    result = sha256()
    for name,value in sorted(values.items()):
        raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        result.update(name.encode())
        result.update(raw.numpy().tobytes())
    return result.hexdigest()

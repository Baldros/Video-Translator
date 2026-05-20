import torch


def get_best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_default_dtype(device: str):
    if device in {"cuda", "mps"}:
        return torch.float16
    return torch.float32

from __future__ import annotations
import math
from typing import List, Optional
import numpy as np
import torch

def pick_device(device: str) -> torch.device:
    if device == "cuda":
        return torch.device("cuda")
    if device == "cpu":
        return torch.device("cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def clamp01(x: float) -> float:
    if x is None or isinstance(x, str):
        return 0.0
    return float(max(0.0, min(1.0, x)))

def safe_mean(xs: List[float]) -> float:
    if not xs:
        return 0.0
    return float(np.mean(np.array(xs, dtype=np.float32)))

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

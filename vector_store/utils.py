import numpy as np
from typing import Tuple


def l2_normalize_vectors(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    # Ensure computations happen in float32 to avoid upcasting to float64 which
    # causes FAISS to reject the array. Return a C-contiguous float32 array.
    v32 = v.astype('float32', copy=False)
    norms = np.linalg.norm(v32, axis=1, keepdims=True)
    norms = np.maximum(norms, eps).astype('float32', copy=False)
    out = v32 / norms
    return np.ascontiguousarray(out, dtype='float32')


def to_float32(v: np.ndarray) -> np.ndarray:
    return v.astype('float32')

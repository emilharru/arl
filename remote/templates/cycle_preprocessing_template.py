#!/usr/bin/env python3
"""Template for proposer-generated preprocessing code.

Contract requirements:
- Keep function name/signature: `apply_preprocessing(x: np.ndarray) -> np.ndarray`
- Do not perform train/validation split loading in this module
- Preserve sample axis (axis 0)
- Returned arrays should be finite `np.float32`
"""

from __future__ import annotations

import numpy as np


def apply_preprocessing(x: np.ndarray) -> np.ndarray:
    """Apply per-signal preprocessing before binary expert training."""
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim < 2:
        raise ValueError(f"Expected preprocessed array with at least 2 dims (N,...), got shape={arr.shape}")


    raise NotImplementedError("Fill PREPROCESSING_PIPELINE snippet")


    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr

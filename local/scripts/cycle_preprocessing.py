#!/usr/bin/env python3
"""Preprocessing utilities for cycle runner."""

from __future__ import annotations

import numpy as np


def apply_preprocessing(x: np.ndarray) -> np.ndarray:
    """Apply baseline preprocessing (float32 conversion + finite cleanup)."""
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim < 2:
        raise ValueError(f"Expected preprocessed array with at least 2 dims (N,...), got shape={arr.shape}")
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr

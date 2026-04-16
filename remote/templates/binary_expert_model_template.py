#!/usr/bin/env python3
"""Template for proposer-generated binary expert architectures.

Contract requirements:
- Keep class name: `BinaryExpertModel`
- Keep method names/signatures: `extract_features`, `forward`
- Model can be any PyTorch architecture (not tied to Inception)
- `extract_features` must return shape `(B, D)`
- `forward` must return logits shape `(B, n_classes)`
- The penultimate embedding dim `D` must match ensemble expectations
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def _normalize_to_bct(x: torch.Tensor) -> torch.Tensor:
    """Normalize input to (B, C, T) while preserving sample axis."""
    if x.dim() == 2:
        return x.unsqueeze(1)
    if x.dim() == 3:
        return x
    if x.dim() > 3:
        return x.reshape(x.size(0), -1, x.size(-1))
    raise ValueError(f"Expected input with at least 2 dims, got shape={tuple(x.shape)}")


class BinaryExpertModel(nn.Module):
    """Template binary expert model used by local/scripts/run_cycle.py."""

    def __init__(
        self,
        in_ch: int,
        n_classes: int,
        fs: float,
        min_seq_len: int,
        dts=(0.05, 0.15, 0.5, 1.5),
        k_min: int = 7,
        k_max_cap: int = 129,
        width: int = 32,
        depth: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if in_ch < 1:
            raise ValueError("in_ch must be >= 1")
        if n_classes < 1:
            raise ValueError("n_classes must be >= 1")
        if min_seq_len < 1:
            raise ValueError("min_seq_len must be >= 1")


        self.embedding_dim = int(width)


        raise NotImplementedError("Fill MODEL_INIT snippet")


    def extract_features(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = _normalize_to_bct(x)
        if lengths is not None:
            lengths = lengths.to(device=x.device, dtype=torch.long).clamp(min=1, max=x.size(-1))


        raise NotImplementedError("Fill EXTRACT_FEATURES snippet")


    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        feat = self.extract_features(x, lengths)
        if feat.dim() != 2:
            raise ValueError(f"extract_features must return shape (B, D), got {tuple(feat.shape)}")


        return self.head(feat)


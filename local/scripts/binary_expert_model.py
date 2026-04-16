#!/usr/bin/env python3
"""Binary expert model definitions for cycle training."""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn


def _odd(k: int) -> int:
    return k if (k % 2 == 1) else k + 1


def kernel_sizes_from_timescales(
    fs: float,
    L: int,
    dts: Sequence[float] = (0.05, 0.15, 0.5, 1.5),
    k_min: int = 7,
    k_max_cap: int = 129,
) -> List[int]:
    """Build odd kernel sizes from time scales, bounded by sequence length."""
    if fs <= 0:
        raise ValueError("fs must be > 0")
    if L <= 0:
        raise ValueError("L must be > 0")
    if k_min < 1:
        raise ValueError("k_min must be >= 1")

    k_max = min(k_max_cap, max(k_min, L // 4))
    ks = []
    for dt in dts:
        k = int(round(fs * float(dt)))
        k = max(k_min, min(k, k_max))
        ks.append(_odd(k))

    ks = sorted(set(ks))
    if not ks:
        ks = [_odd(k_min)]
    return ks


def _best_num_groups(num_channels: int, max_groups: int = 8) -> int:
    g = min(max_groups, num_channels)
    while g > 1 and (num_channels % g) != 0:
        g -= 1
    return g


class MaskedGlobalAvgPool1d(nn.Module):
    """Mask-aware temporal average for padded sequence batches."""

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:

        B, _, T = x.shape
        lengths = lengths.to(device=x.device, dtype=torch.long).clamp(min=1, max=T)
        t = torch.arange(T, device=x.device).view(1, 1, T)
        mask = (t < lengths.view(B, 1, 1)).to(x.dtype)
        x_sum = (x * mask).sum(dim=-1)
        denom = mask.sum(dim=-1).clamp(min=1.0)
        return x_sum / denom


class InceptionBlock1D(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_out: int,
        kernel_sizes: Sequence[int],
        bottleneck: int = 32,
        dropout: float = 0.1,
        use_pool_branch: bool = True,
    ):
        super().__init__()
        if c_in < 1 or c_out < 1:
            raise ValueError("c_in and c_out must be >= 1")
        if len(kernel_sizes) < 1:
            raise ValueError("kernel_sizes must be non-empty")

        self.kernel_sizes = list(kernel_sizes)
        c_mid = min(bottleneck, c_in) if c_in > 1 else c_in
        self.bottleneck = nn.Conv1d(c_in, c_mid, kernel_size=1, bias=False) if c_in > 1 else nn.Identity()
        self.branches = nn.ModuleList(
            [nn.Conv1d(c_mid, c_out, kernel_size=k, padding=k // 2, bias=False) for k in self.kernel_sizes]
        )

        self.use_pool_branch = use_pool_branch
        if use_pool_branch:
            self.pool_branch = nn.Sequential(
                nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
                nn.Conv1d(c_in, c_out, kernel_size=1, bias=False),
            )

        total_out = c_out * (len(self.kernel_sizes) + (1 if use_pool_branch else 0))
        g = _best_num_groups(total_out, max_groups=8)
        self.norm = nn.GroupNorm(num_groups=g, num_channels=total_out)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Conv1d(total_out, c_out, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.bottleneck(x)
        outs = [conv(z) for conv in self.branches]
        if self.use_pool_branch:
            outs.append(self.pool_branch(x))
        y = torch.cat(outs, dim=1)
        y = self.drop(self.act(self.norm(y)))
        return self.proj(y)


class BinaryExpertModel(nn.Module):
    """Cycle baseline binary expert model for raw one-channel time series."""

    def __init__(
        self,
        in_ch: int,
        n_classes: int,
        fs: float,
        min_seq_len: int,
        dts: Sequence[float] = (0.05, 0.15, 0.5, 1.5),
        k_min: int = 7,
        k_max_cap: int = 129,
        width: int = 32,
        depth: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if min_seq_len < 1:
            raise ValueError("min_seq_len must be >= 1")

        ks = kernel_sizes_from_timescales(
            fs=fs,
            L=min_seq_len,
            dts=dts,
            k_min=k_min,
            k_max_cap=k_max_cap,
        )

        self.kernel_sizes = ks
        self.stem = nn.Conv1d(in_ch, width, kernel_size=7, padding=3, bias=False)
        self.blocks = nn.ModuleList(
            [
                InceptionBlock1D(
                    c_in=width,
                    c_out=width,
                    kernel_sizes=ks,
                    bottleneck=max(8, width // 2),
                    dropout=dropout,
                    use_pool_branch=True,
                )
                for _ in range(depth)
            ]
        )

        g = _best_num_groups(width, max_groups=8)
        self.post_norm = nn.GroupNorm(num_groups=g, num_channels=width)
        self.post_act = nn.GELU()

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.mask_pool = MaskedGlobalAvgPool1d()
        self.head = nn.Linear(width, n_classes)

    def extract_features(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.stem(x)
        res = x
        for blk in self.blocks:
            x = blk(x)
            x = self.post_act(self.post_norm(x + res))
            res = x

        if lengths is None:
            return self.pool(x).squeeze(-1)
        return self.mask_pool(x, lengths)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        feat = self.extract_features(x, lengths)
        return self.head(feat)

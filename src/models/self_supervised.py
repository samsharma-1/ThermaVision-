"""Masked patch reconstruction for self-supervised pretraining (Phase 4)."""

from __future__ import annotations

import torch
import torch.nn as nn


class MaskedReconstructionHead(nn.Module):
    """Reconstruct masked IR patches from encoder features."""

    def __init__(self, in_channels: int = 1024, out_channels: int = 1) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 512, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.decoder(features)


def create_random_mask(
    batch_size: int,
    height: int,
    width: int,
    mask_ratio: float,
    device: torch.device,
) -> torch.Tensor:
    """Binary mask: 1 = visible, 0 = masked."""
    mask = torch.ones(batch_size, 1, height, width, device=device)
    n_mask = int(height * width * mask_ratio)
    for b in range(batch_size):
        flat_idx = torch.randperm(height * width, device=device)[:n_mask]
        mask[b, 0].view(-1)[flat_idx] = 0.0
    return mask


def apply_mask(ir: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return ir * mask

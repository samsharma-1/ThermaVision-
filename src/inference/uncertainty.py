"""MC Dropout uncertainty estimation (Phase 5)."""

from __future__ import annotations

import numpy as np
import torch

from src.models import UNetGenerator


@torch.no_grad()
def predict_with_uncertainty(
    generator: UNetGenerator,
    ir: torch.Tensor,
    n_samples: int = 20,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run MC Dropout inference and return (mean_rgb, pixel_variance).

    ir: (1, 1, H, W) or (B, 1, H, W)
    Returns mean and variance in [0, 1] RGB space.
    """
    generator.enable_mc_dropout()
    samples = []
    for _ in range(n_samples):
        out = generator(ir.to(device))
        out_01 = ((out + 1.0) / 2.0).clamp(0, 1)
        samples.append(out_01.cpu())

    stack = torch.stack(samples, dim=0)  # (N, B, 3, H, W)
    mean = stack.mean(dim=0)
    variance = stack.var(dim=0).mean(dim=1, keepdim=True)  # avg across RGB -> (B, 1, H, W)
    generator.eval()
    return mean, variance


def uncertainty_heatmap(variance: torch.Tensor) -> np.ndarray:
    """Normalize variance to [0, 255] uint8 heatmap."""
    v = variance.squeeze().numpy()
    v = (v - v.min()) / (v.max() - v.min() + 1e-8)
    return (v * 255).astype(np.uint8)

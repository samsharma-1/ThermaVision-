"""Evaluation metrics: PSNR, SSIM, LPIPS."""

from __future__ import annotations

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def _to_numpy_rgb(tensor: torch.Tensor) -> np.ndarray:
    """Convert (B,3,H,W) or (3,H,W) in [-1,1] or [0,1] to uint8 numpy."""
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    x = tensor.detach().cpu().float()
    if x.min() < 0:
        x = (x + 1.0) / 2.0
    x = x.clamp(0, 1).permute(0, 2, 3, 1).numpy()
    return (x * 255).astype(np.uint8)


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_np = _to_numpy_rgb(pred)
    target_np = _to_numpy_rgb(target)
    scores = []
    for p, t in zip(pred_np, target_np):
        scores.append(peak_signal_noise_ratio(t, p, data_range=255))
    return float(np.mean(scores))


def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_np = _to_numpy_rgb(pred)
    target_np = _to_numpy_rgb(target)
    scores = []
    for p, t in zip(pred_np, target_np):
        scores.append(
            structural_similarity(t, p, channel_axis=2, data_range=255)
        )
    return float(np.mean(scores))


def compute_lpips(pred: torch.Tensor, target: torch.Tensor, device: str = "cpu") -> float:
    import lpips

    loss_fn = lpips.LPIPS(net="alex").to(device)
    p = pred.to(device)
    t = target.to(device)
    if p.min() >= 0:
        p = p * 2.0 - 1.0
        t = t * 2.0 - 1.0
    with torch.no_grad():
        scores = loss_fn(p, t)
    return float(scores.mean().item())


def evaluate_batch(
    pred: torch.Tensor,
    target: torch.Tensor,
    device: str = "cpu",
) -> dict[str, float]:
    return {
        "psnr": compute_psnr(pred, target),
        "ssim": compute_ssim(pred, target),
        "lpips": compute_lpips(pred, target, device),
    }

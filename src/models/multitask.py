"""Lightweight detection decoder for multi-task training (Phase 3)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DetectionHead(nn.Module):
    """
    Simple heatmap-based detection head on encoder bottleneck features.

    For full mAP evaluation, use scripts/run_detection.py with a pretrained
    aerial detector (YOLO) on colorized outputs.
    """

    def __init__(self, in_channels: int = 1024, num_classes: int = 80) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Upsample to input resolution for pseudo-detection supervision
        out = self.conv(features)
        return F.interpolate(out, scale_factor=16, mode="bilinear", align_corners=False)

    @staticmethod
    def pseudo_target_from_rgb(rgb: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Create a coarse pseudo-detection target from RGB edges (placeholder)."""
        gray = rgb.mean(dim=1, keepdim=True)
        sobel_x = F.conv2d(gray, torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], device=rgb.device).float(), padding=1)
        sobel_y = F.conv2d(gray, torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], device=rgb.device).float(), padding=1)
        edges = torch.sqrt(sobel_x ** 2 + sobel_y ** 2)
        edges = (edges > edges.mean() * threshold).float()
        return edges.expand(-1, 1, -1, -1)

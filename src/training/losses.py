"""Composite training losses: adversarial + L1 + VGG perceptual."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


class GANLoss(nn.Module):
    def __init__(self, gan_mode: str = "lsgan") -> None:
        super().__init__()
        self.gan_mode = gan_mode
        if gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        elif gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unknown GAN mode: {gan_mode}")

    def forward(self, pred: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
        return self.loss(pred, target)


class PerceptualLoss(nn.Module):
    """VGG16 feature matching loss."""

    def __init__(self) -> None:
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_FEATURES)
        self.features = nn.Sequential(*list(vgg.features[:16])).eval()
        for p in self.features.parameters():
            p.requires_grad = False
        self.l1 = nn.L1Loss()

        # ImageNet normalization for VGG
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        # Generator outputs Tanh [-1, 1] -> [0, 1]
        x = (x + 1.0) / 2.0
        return (x - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_n = self._normalize(pred)
        target_n = self._normalize(target)
        return self.l1(self.features(pred_n), self.features(target_n))


class CompositeLoss(nn.Module):
    def __init__(
        self,
        lambda_l1: float = 100.0,
        lambda_perceptual: float = 10.0,
    ) -> None:
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_perceptual = lambda_perceptual
        self.gan_loss = GANLoss("lsgan")
        self.l1 = nn.L1Loss()
        self.perceptual = PerceptualLoss()

    def discriminator_loss(
        self, d_real: torch.Tensor, d_fake: torch.Tensor
    ) -> torch.Tensor:
        loss_real = self.gan_loss(d_real, True)
        loss_fake = self.gan_loss(d_fake, False)
        return (loss_real + loss_fake) * 0.5

    def generator_loss(
        self,
        d_fake: torch.Tensor,
        fake_rgb: torch.Tensor,
        real_rgb: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        loss_g_gan = self.gan_loss(d_fake, True)
        loss_l1 = self.l1(fake_rgb, real_rgb) * self.lambda_l1
        loss_perc = self.perceptual(fake_rgb, real_rgb) * self.lambda_perceptual
        total = loss_g_gan + loss_l1 + loss_perc
        return {
            "total": total,
            "gan": loss_g_gan,
            "l1": loss_l1,
            "perceptual": loss_perc,
        }

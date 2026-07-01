"""U-Net generator and PatchGAN discriminator (pix2pix-style)."""

from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetGenerator(nn.Module):
    """U-Net generator: 1-channel IR -> 3-channel RGB."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 3,
        base_filters: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dropout_rate = dropout

        # Encoder
        self.enc1 = DoubleConv(in_channels, base_filters)
        self.enc2 = DoubleConv(base_filters, base_filters * 2)
        self.enc3 = DoubleConv(base_filters * 2, base_filters * 4)
        self.enc4 = DoubleConv(base_filters * 4, base_filters * 8)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(base_filters * 8, base_filters * 16, dropout=dropout)

        # Decoder
        self.up4 = nn.ConvTranspose2d(base_filters * 16, base_filters * 8, 2, stride=2)
        self.dec4 = DoubleConv(base_filters * 16, base_filters * 8, dropout=dropout)
        self.up3 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, 2, stride=2)
        self.dec3 = DoubleConv(base_filters * 8, base_filters * 4)
        self.up2 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, 2, stride=2)
        self.dec2 = DoubleConv(base_filters * 4, base_filters * 2)
        self.up1 = nn.ConvTranspose2d(base_filters * 2, base_filters, 2, stride=2)
        self.dec1 = DoubleConv(base_filters * 2, base_filters)

        self.out_conv = nn.Sequential(
            nn.Conv2d(base_filters, out_channels, 1),
            nn.Tanh(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return bottleneck features for multi-task / self-supervised heads."""
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        return self.bottleneck(self.pool(e4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)

    def enable_mc_dropout(self) -> None:
        """Enable dropout at inference for MC Dropout uncertainty (Phase 5)."""
        for m in self.modules():
            if isinstance(m, nn.Dropout2d):
                m.train()


class PatchGANDiscriminator(nn.Module):
    """70x70 PatchGAN discriminator on concatenated (IR, RGB)."""

    def __init__(self, in_channels: int = 4, base_filters: int = 64) -> None:
        super().__init__()

        def block(ic: int, oc: int, stride: int, normalize: bool = True) -> nn.Sequential:
            layers: list[nn.Module] = [nn.Conv2d(ic, oc, 4, stride=stride, padding=1, bias=not normalize)]
            if normalize:
                layers.append(nn.BatchNorm2d(oc))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            block(in_channels, base_filters, 2, normalize=False),
            block(base_filters, base_filters * 2, 2),
            block(base_filters * 2, base_filters * 4, 2),
            block(base_filters * 4, base_filters * 8, 1),
            nn.Conv2d(base_filters * 8, 1, 4, padding=1),
        )

    def forward(self, ir: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([ir, rgb], dim=1)
        return self.model(x)

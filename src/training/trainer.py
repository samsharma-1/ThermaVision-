"""Main GAN training loop."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models import PatchGANDiscriminator, UNetGenerator
from src.registration import align_batch_ir
from src.training.losses import CompositeLoss


class Trainer:
    def __init__(
        self,
        config: dict[str, Any],
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str,
    ) -> None:
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        model_cfg = config["model"]
        train_cfg = config["training"]

        self.generator = UNetGenerator(
            in_channels=model_cfg["generator"]["in_channels"],
            out_channels=model_cfg["generator"]["out_channels"],
            base_filters=model_cfg["generator"]["base_filters"],
            dropout=model_cfg["generator"].get("dropout", 0.1),
        ).to(device)

        self.discriminator = PatchGANDiscriminator(
            in_channels=model_cfg["discriminator"]["in_channels"],
            base_filters=model_cfg["discriminator"]["base_filters"],
        ).to(device)

        self.criterion = CompositeLoss(
            lambda_l1=train_cfg.get("lambda_l1", 100.0),
            lambda_perceptual=train_cfg.get("lambda_perceptual", 10.0),
        )

        self.opt_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=train_cfg.get("lr_g", 2e-4),
            betas=(train_cfg.get("beta1", 0.5), train_cfg.get("beta2", 0.999)),
        )
        self.opt_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=train_cfg.get("lr_d", 2e-4),
            betas=(train_cfg.get("beta1", 0.5), train_cfg.get("beta2", 0.999)),
        )

        self.use_registration = train_cfg.get("use_registration", False)
        self.checkpoint_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(train_cfg.get("log_dir", "logs"))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history: list[dict[str, float]] = []

    def _maybe_align(self, ir: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        if not self.use_registration:
            return ir
        ir_np = ir.permute(0, 2, 3, 1).cpu().numpy()
        rgb_np = rgb.permute(0, 2, 3, 1).cpu().numpy()
        aligned_np, _ = align_batch_ir(ir_np, rgb_np, self.config)
        aligned = torch.from_numpy(aligned_np).permute(0, 3, 1, 2).float()
        if aligned.shape[1] == 1:
            pass
        elif aligned.ndim == 4 and aligned.shape[-1] == 1:
            aligned = aligned.permute(0, 3, 1, 2)
        return aligned.to(self.device)

    def train_epoch(self, epoch: int) -> dict[str, float]:
        self.generator.train()
        self.discriminator.train()
        totals = {"g_total": 0.0, "d_total": 0.0, "g_gan": 0.0, "g_l1": 0.0, "g_perc": 0.0}
        n = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            ir = batch["ir"].to(self.device)
            rgb = batch["rgb"].to(self.device)
            rgb_target = (rgb * 2.0) - 1.0  # match Tanh output range

            ir = self._maybe_align(ir, rgb)

            # --- Discriminator ---
            self.opt_d.zero_grad()
            with torch.no_grad():
                fake_rgb = self.generator(ir)
            d_real = self.discriminator(ir, rgb_target)
            d_fake = self.discriminator(ir, fake_rgb.detach())
            loss_d = self.criterion.discriminator_loss(d_real, d_fake)
            loss_d.backward()
            self.opt_d.step()

            # --- Generator ---
            self.opt_g.zero_grad()
            fake_rgb = self.generator(ir)
            d_fake_for_g = self.discriminator(ir, fake_rgb)
            g_losses = self.criterion.generator_loss(d_fake_for_g, fake_rgb, rgb_target)
            g_losses["total"].backward()
            self.opt_g.step()

            totals["d_total"] += loss_d.item()
            totals["g_total"] += g_losses["total"].item()
            totals["g_gan"] += g_losses["gan"].item()
            totals["g_l1"] += g_losses["l1"].item()
            totals["g_perc"] += g_losses["perceptual"].item()
            n += 1
            pbar.set_postfix(g=totals["g_total"] / n, d=totals["d_total"] / n)

        return {k: v / max(n, 1) for k, v in totals.items()}

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        self.generator.eval()
        totals = {"g_l1": 0.0}
        n = 0
        for batch in self.val_loader:
            ir = batch["ir"].to(self.device)
            rgb = batch["rgb"].to(self.device)
            rgb_target = (rgb * 2.0) - 1.0
            ir = self._maybe_align(ir, rgb)
            fake = self.generator(ir)
            totals["g_l1"] += nn.functional.l1_loss(fake, rgb_target).item()
            n += 1
        return {k: v / max(n, 1) for k, v in totals.items()}

    def save_checkpoint(self, epoch: int, tag: str = "latest") -> Path:
        path = self.checkpoint_dir / f"checkpoint_{tag}.pt"
        torch.save(
            {
                "epoch": epoch,
                "generator": self.generator.state_dict(),
                "discriminator": self.discriminator.state_dict(),
                "opt_g": self.opt_g.state_dict(),
                "opt_d": self.opt_d.state_dict(),
                "config": self.config,
            },
            path,
        )
        return path

    def train(self, epochs: int | None = None) -> None:
        epochs = epochs or self.config["training"].get("epochs", 50)
        save_every = self.config["training"].get("save_every", 5)

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()
            record = {"epoch": epoch, **train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}}
            self.history.append(record)
            print(f"Epoch {epoch}: train_g={train_metrics['g_total']:.4f} val_l1={val_metrics['g_l1']:.4f}")

            if epoch % save_every == 0 or epoch == epochs:
                self.save_checkpoint(epoch, tag=f"epoch{epoch}")

        self.save_checkpoint(epochs, tag="latest")
        with (self.log_dir / "train_history.json").open("w") as f:
            json.dump(self.history, f, indent=2)

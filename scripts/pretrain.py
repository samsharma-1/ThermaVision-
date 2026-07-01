#!/usr/bin/env python3
"""Self-supervised masked reconstruction pretraining (Phase 4)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import create_dataloaders
from src.models import MaskedReconstructionHead, UNetGenerator
from src.models.self_supervised import apply_mask, create_random_mask
from src.utils.config import get_device, load_config
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    set_seed(config.get("project", {}).get("seed", 42))
    device = get_device(config)
    ss_cfg = config.get("self_supervised", {})
    epochs = args.epochs or ss_cfg.get("pretrain_epochs", 30)
    mask_ratio = ss_cfg.get("mask_ratio", 0.4)

    train_loader, _, _ = create_dataloaders(config)
    model_cfg = config["model"]["generator"]
    encoder = UNetGenerator(
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"],
        base_filters=model_cfg["base_filters"],
    ).to(device)
    recon_head = MaskedReconstructionHead(in_channels=model_cfg["base_filters"] * 16).to(device)

    opt = torch.optim.Adam(
        list(encoder.parameters()) + list(recon_head.parameters()), lr=2e-4
    )
    criterion = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        encoder.train()
        recon_head.train()
        total = 0.0
        n = 0
        for batch in tqdm(train_loader, desc=f"Pretrain {epoch}"):
            ir = batch["ir"].to(device)
            b, _, h, w = ir.shape
            mask = create_random_mask(b, h, w, mask_ratio, device)
            masked_ir = apply_mask(ir, mask)
            features = encoder.encode(masked_ir)
            recon = recon_head(features)
            loss = criterion(recon * (1 - mask), ir * (1 - mask))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            n += 1
        print(f"Epoch {epoch}: recon_loss={total / max(n, 1):.4f}")

    ckpt_dir = Path(config["training"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / "pretrained_encoder.pt"
    torch.save({"encoder": encoder.state_dict(), "recon_head": recon_head.state_dict()}, path)
    print(f"Saved pretrained encoder to {path}")


if __name__ == "__main__":
    main()

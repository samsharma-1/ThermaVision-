#!/usr/bin/env python3
"""Train baseline IR->RGB pix2pix model (Phase 1)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import create_dataloaders
from src.training.trainer import Trainer
from src.utils.config import get_device, load_config
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--registration", action="store_true", help="Enable Phase 2 alignment")
    parser.add_argument("--misalign", action="store_true", help="Inject synthetic misalignment")
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    if args.registration:
        config["training"]["use_registration"] = True
    set_seed(config.get("project", {}).get("seed", 42))
    device = get_device(config)
    print(f"Device: {device}")

    train_loader, val_loader, _ = create_dataloaders(
        config, apply_misalign=args.misalign
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    trainer = Trainer(config, train_loader, val_loader, device)
    trainer.train(epochs=args.epochs)
    print(f"Checkpoints saved to {trainer.checkpoint_dir}")


if __name__ == "__main__":
    main()

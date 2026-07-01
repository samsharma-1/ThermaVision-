#!/usr/bin/env python3
"""Evaluate model: PSNR / SSIM / LPIPS + save comparison images."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import create_dataloaders
from src.eval.harness import load_generator, run_evaluation, run_registration_ablation
from src.utils.config import get_device, load_config
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint_latest.pt")
    parser.add_argument("--ablation", action="store_true", help="Run registration ablation (Phase 2)")
    parser.add_argument("--output", type=str, default="results")
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    set_seed(config.get("project", {}).get("seed", 42))
    device = get_device(config)
    output_dir = ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    _, _, test_loader = create_dataloaders(config)
    ckpt_path = ROOT / args.checkpoint
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        print("Train first: python scripts/train.py")
        sys.exit(1)

    generator = load_generator(ckpt_path, device)

    if args.ablation:
        results = run_registration_ablation(generator, test_loader, device, config, output_dir)
    else:
        results = run_evaluation(
            generator, test_loader, device, config,
            use_registration=config["training"].get("use_registration", False),
            output_dir=output_dir,
        )
        print(f"\nPSNR={results['psnr']:.2f}  SSIM={results['ssim']:.4f}  LPIPS={results['lpips']:.4f}")
        with (output_dir / "metrics.json").open("w") as f:
            json.dump(results, f, indent=2)

    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()

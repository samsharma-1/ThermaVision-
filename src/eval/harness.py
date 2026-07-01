"""Full evaluation harness and result export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.eval.metrics import evaluate_batch
from src.models import UNetGenerator
from src.registration import align_batch_ir


def load_generator(checkpoint_path: Path, device: str) -> UNetGenerator:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    model_cfg = config.get("model", {}).get("generator", {})
    gen = UNetGenerator(
        in_channels=model_cfg.get("in_channels", 1),
        out_channels=model_cfg.get("out_channels", 3),
        base_filters=model_cfg.get("base_filters", 64),
        dropout=model_cfg.get("dropout", 0.1),
    )
    gen.load_state_dict(ckpt["generator"])
    gen.to(device)
    gen.eval()
    return gen


@torch.no_grad()
def run_evaluation(
    generator: UNetGenerator,
    loader: DataLoader,
    device: str,
    config: dict[str, Any],
    use_registration: bool = False,
    output_dir: Path | None = None,
    max_samples: int | None = None,
) -> dict[str, float]:
    totals = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0}
    n = 0
    output_dir = Path(output_dir) if output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    for batch in tqdm(loader, desc="Evaluating"):
        ir = batch["ir"].to(device)
        rgb = batch["rgb"].to(device)

        if use_registration:
            ir_np = ir.permute(0, 2, 3, 1).cpu().numpy()
            rgb_np = rgb.permute(0, 2, 3, 1).cpu().numpy()
            aligned_np, _ = align_batch_ir(ir_np, rgb_np, config)
            ir = torch.from_numpy(aligned_np).permute(0, 3, 1, 2).float().to(device)

        fake = generator(ir)
        fake_01 = (fake + 1.0) / 2.0
        metrics = evaluate_batch(fake_01, rgb, device)
        for k in totals:
            totals[k] += metrics[k]
        n += 1

        if output_dir and n <= 4:
            _save_comparison(
                ir[0], rgb[0], fake_01[0],
                output_dir / f"comparison_{n:02d}.png",
            )

        if max_samples and n >= max_samples:
            break

    return {k: v / max(n, 1) for k, v in totals.items()}


def _save_comparison(
    ir: torch.Tensor,
    rgb: torch.Tensor,
    fake: torch.Tensor,
    path: Path,
) -> None:
    def _to_img(t: torch.Tensor) -> np.ndarray:
        if t.shape[0] == 1:
            arr = t.squeeze(0).cpu().numpy()
            arr = np.stack([arr] * 3, axis=-1)
        else:
            arr = t.permute(1, 2, 0).cpu().numpy()
        return np.clip(arr, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(_to_img(ir), cmap="gray" if ir.shape[0] == 1 else None)
    axes[0].set_title("IR Input")
    axes[0].axis("off")
    axes[1].imshow(_to_img(fake))
    axes[1].set_title("Colorized")
    axes[1].axis("off")
    axes[2].imshow(_to_img(rgb))
    axes[2].set_title("RGB Target")
    axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def run_registration_ablation(
    generator: UNetGenerator,
    loader: DataLoader,
    device: str,
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, dict[str, float]]:
    """Compare metrics with vs. without registration alignment."""
    output_dir.mkdir(parents=True, exist_ok=True)
    without = run_evaluation(generator, loader, device, config, use_registration=False)
    with_reg = run_evaluation(generator, loader, device, config, use_registration=True)

    results = {"without_registration": without, "with_registration": with_reg}
    with (output_dir / "registration_ablation.json").open("w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Registration Ablation ===")
    print(f"Without: PSNR={without['psnr']:.2f} SSIM={without['ssim']:.4f} LPIPS={without['lpips']:.4f}")
    print(f"With:    PSNR={with_reg['psnr']:.2f} SSIM={with_reg['ssim']:.4f} LPIPS={with_reg['lpips']:.4f}")
    return results

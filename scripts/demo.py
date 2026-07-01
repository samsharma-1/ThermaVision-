#!/usr/bin/env python3
"""Gradio demo: IR input -> colorized RGB + uncertainty heatmap (Phase 6)."""

from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.degradation import degrade_rgb_to_ir
from src.eval.harness import load_generator
from src.inference.uncertainty import predict_with_uncertainty, uncertainty_heatmap
from src.utils.config import get_device, load_config


def _load_models(config_path: str, checkpoint: str):
    config = load_config(ROOT / config_path)
    device = get_device(config)
    ckpt = ROOT / checkpoint
    if not ckpt.exists():
        return None, device, config
    gen = load_generator(ckpt, device)
    return gen, device, config


def process_image(
    image: np.ndarray,
    generator,
    device: str,
    mc_samples: int,
    show_uncertainty: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    if image is None:
        return np.zeros((256, 256, 3), dtype=np.uint8), None

    pil = Image.fromarray(image).convert("RGB")
    rgb = np.asarray(pil, dtype=np.float32) / 255.0
    ir = degrade_rgb_to_ir(rgb)
    ir_t = torch.from_numpy(ir).permute(2, 0, 1).unsqueeze(0).float()

    if generator is None:
        fake = (rgb * 255).astype(np.uint8)
        return fake, None

    if show_uncertainty and mc_samples > 1:
        mean, var = predict_with_uncertainty(generator, ir_t, mc_samples, device)
        fake = (mean.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        heat = uncertainty_heatmap(var)
        cmap = plt.get_cmap("hot")
        heat_rgb = (cmap(heat / 255.0)[:, :, :3] * 255).astype(np.uint8)
        return fake, heat_rgb

    with torch.no_grad():
        out = generator(ir_t.to(device))
        fake = ((out + 1) / 2).squeeze(0).permute(1, 2, 0).cpu().numpy()
        fake = (fake * 255).astype(np.uint8)
    return fake, None


def build_demo() -> gr.Blocks:
    config_path = "config/default.yaml"
    checkpoint = "checkpoints/checkpoint_latest.pt"
    config = load_config(ROOT / config_path)
    generator, device, config = _load_models(config_path, checkpoint)
    mc_samples = config.get("uncertainty", {}).get("mc_samples", 20)

    with gr.Blocks(title="IR → RGB Satellite Interpretation") as demo:
        gr.Markdown(
            "# Self-Supervised Registration-Aware Multi-Task Fusion\n"
            "Upload a satellite/RGB image (synthetic IR is generated on the fly) "
            "or use a grayscale IR image."
        )
        with gr.Row():
            inp = gr.Image(label="Input (RGB or IR)", type="numpy")
            out_rgb = gr.Image(label="Colorized Output")
            out_unc = gr.Image(label="Uncertainty Heatmap")

        mc = gr.Slider(1, 30, value=mc_samples, step=1, label="MC Dropout samples")
        unc_toggle = gr.Checkbox(value=True, label="Show uncertainty map")

        btn = gr.Button("Colorize", variant="primary")

        def _run(img, n_samples, show_unc):
            rgb_out, heat = process_image(img, generator, device, int(n_samples), show_unc)
            return rgb_out, heat if heat is not None else np.zeros((256, 256, 3), dtype=np.uint8)

        btn.click(_run, [inp, mc, unc_toggle], [out_rgb, out_unc])

        if generator is None:
            gr.Markdown("*No checkpoint found — run `python scripts/train.py` first.*")

    return demo


def main() -> None:
    config = load_config(ROOT / "config/default.yaml")
    port = config.get("demo", {}).get("port", 7860)
    demo = build_demo()
    demo.launch(server_port=port)


if __name__ == "__main__":
    main()

# Self-Supervised Registration-Aware Multi-Task Fusion for Infrared Satellite Interpretation

> **A self-supervised, registration-aware multi-task framework for infrared-to-visible satellite interpretation.**

Hackathon build implementing all six phases from the project plan: baseline pix2pix colorization, registration-aware fusion, multi-task detection scaffold, self-supervised pretraining, MC Dropout uncertainty, and Gradio demo packaging.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare data (synthetic RGB fallback if downloads fail)
python scripts/prepare_data.py --count 50 --synthetic-only

# 3. Train baseline (Phase 1)
python scripts/train.py --epochs 20

# 4. Evaluate PSNR / SSIM / LPIPS
python scripts/evaluate.py --checkpoint checkpoints/checkpoint_latest.pt

# 5. Registration ablation (Phase 2)
python scripts/train.py --epochs 20 --misalign --registration
python scripts/evaluate.py --ablation

# 6. Self-supervised pretrain (Phase 4)
python scripts/pretrain.py --epochs 30
python scripts/train.py --epochs 20  # fine-tune from pretrained encoder manually

# 7. Launch demo (Phase 6)
python scripts/demo.py
```

## Project Structure

```
config/default.yaml       # All hyperparameters
src/
  data/                   # Synthetic IR degradation + dataset
  models/                 # U-Net, PatchGAN, multi-task & SSL heads
  training/               # Losses + trainer
  registration/           # ORB/SIFT alignment (Phase 2)
  eval/                   # PSNR, SSIM, LPIPS harness
  inference/              # MC Dropout uncertainty (Phase 5)
scripts/
  prepare_data.py         # Download/generate RGB training images
  train.py                # Phase 1 (+ optional registration)
  evaluate.py             # Metrics + ablation tables
  pretrain.py             # Phase 4 masked reconstruction
  run_detection.py        # Phase 3 detection comparison scaffold
  demo.py                 # Gradio live demo
```

## Phases Implemented

| Phase | Status | Entry point |
|-------|--------|-------------|
| 0 — Setup | Done | `prepare_data.py`, `config/default.yaml` |
| 1 — Baseline pix2pix | Done | `train.py` |
| 2 — Registration | Done | `--registration`, `--misalign`, `evaluate.py --ablation` |
| 3 — Multi-task detection | Scaffold | `run_detection.py`, `DetectionHead` |
| 4 — Self-supervised | Done | `pretrain.py` |
| 5 — Uncertainty | Done | MC Dropout in `demo.py` |
| 6 — Packaging | Done | `demo.py`, `results/` exports |

## Dataset Strategy

1. **Preferred:** Real VIIRS + Sentinel-2/Landsat co-registered pairs → place RGB in `data/rgb/`, IR in `data/ir/`
2. **Fallback (default):** Synthetic IR from RGB via luminance + contrast + noise (`src/data/degradation.py`)

Report PSNR/SSIM transparently when using synthetic pairs — metrics partly measure reconstruction of your own degradation function.

## Pitch Results Checklist

- [ ] Baseline PSNR / SSIM / LPIPS → `results/metrics.json`
- [ ] Registration ablation → `results/registration_ablation.json`
- [ ] Detection three-way comparison → `results/detection_comparison.json`
- [ ] Self-supervised label-efficiency table (pretrain + 20% pairs vs paired-only)
- [ ] Uncertainty heatmap → Gradio demo
- [ ] 3–4 before/after images → `results/comparison_*.png`

## Team Parallel Tracks

- **Person A:** `src/data/`, `src/registration/`
- **Person B:** `src/models/`, `src/training/`
- **Person C:** `scripts/pretrain.py`
- **Person D:** `src/eval/`, `scripts/demo.py`, `scripts/evaluate.py`

## License

MIT — hackathon prototype.

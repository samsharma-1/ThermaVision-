# Self-Supervised Registration-Aware Multi-Task Fusion for Infrared Satellite Interpretation
### Full-Scope Build Plan — 24–36 Hour Window

## Project Framing

> **"A self-supervised, registration-aware multi-task framework for infrared-to-visible satellite interpretation."**

This framing gives room to discuss:
- Robustness to misalignment
- Reduced dependence on labeled pairs
- Improved downstream interpretation (not just visual colorization)

**Scope note:** all six phases below are included as planned. Time-boxes and checkpoints are there so you always know where you stand — if a checkpoint is missed, document the current state honestly and keep moving rather than losing time chasing a perfect result on one phase at the expense of the rest.

---

## Team Setup (assume ~4 people, parallelize wherever possible)

- **Person A** — data pipeline + registration
- **Person B** — baseline model (generator/discriminator/losses) + multi-task head
- **Person C** — self-supervised pretraining track (can run in parallel to A/B once data pipeline exists)
- **Person D** — eval harness, uncertainty estimation, demo/packaging (starts early, builds scaffolding while others train)

Running tracks in parallel rather than strictly sequentially is what makes fitting all six phases into this window realistic.

---

## Hour-by-Hour Plan

| Hours | Phase | Task |
|---|---|---|
| 0–2 | Phase 0 | Confirm compute access (Colab Pro/Kaggle/local GPU). Lock dataset approach (VIIRS+Landsat/Sentinel-2, SpaceNet/SEN12MS, or synthetic degraded-RGB fallback — decide fast, don't rabbit-hole). Repo/env setup. Team splits as above. |
| 2–9 | Phase 1 | Build U-Net generator + PatchGAN discriminator. Loss: adversarial + L1 + VGG perceptual. Train end-to-end. **Checkpoint (hour 9): must have a visibly working before/after IR→RGB output.** Compute baseline PSNR/SSIM/LPIPS. |
| 9–15 | Phase 2 | Registration-aware fusion. Implement classical alignment (ORB/SIFT + homography, or mutual-information-based matching if cross-spectral keypoint matching struggles — see risk note below). Warp IR to align with RGB. Retrain with aligned vs. unaligned data. **Checkpoint (hour 15): ablation table (PSNR/SSIM with vs. without registration) exists, even if partial.** |
| 15–21 | Phase 3 | Multi-task head. Add a detection decoder off the shared encoder (or fine-tune a pretrained aerial-imagery detector — see risk note on COCO/YOLO below — on colorized outputs). Train jointly with colorization loss, weighted combination. **Checkpoint (hour 21): three-way detection comparison (real RGB vs. colorized vs. raw IR) exists in some form.** |
| 21–27 | Phase 4 | Self-supervised pretraining. Pretrain encoder on unlabeled IR using masked patch reconstruction (simpler to get right under time pressure than contrastive learning). Fine-tune on paired data, compare against a deliberately small paired subset (e.g. 20%) to show label-efficiency. **Checkpoint (hour 27): comparison table, pretrained+fine-tuned vs. paired-only-small-subset.** |
| 27–30 | Phase 5 | Uncertainty estimation. Enable dropout in the generator, run N inference passes (MC Dropout), compute pixel-wise variance, produce an uncertainty heatmap overlay. Fastest phase — low risk, mostly inference-time. |
| 30–36 | Phase 6 | Packaging. Consolidate results table across all phases. Pick 3–4 best before/after comparisons. Build demo (Gradio/notebook). Write pitch narrative. Rehearse. **Reserve this block fully — do not let earlier phases eat into it.** |

---

## Checkpoint Discipline (keeps all six phases alive without losing the whole project)

At each checkpoint above: if the phase is working, move on as planned. If it's not fully working, **capture whatever partial result exists** (a partial ablation number, a qualitative-only comparison, a training curve showing direction even without full convergence) and move to the next phase on schedule. A partial result you can show and explain honestly is worth more at pitch time than a late, unfinished attempt at perfection on one phase.

---

## Priority Order If Genuinely Behind at Any Point

Phase 1 (baseline) → Phase 2 (registration) → Phase 3 (multi-task) → Phase 6 (packaging, protect this) → Phase 4/5 (whichever has partial progress, finish that one first).

---

## Known Technical Risks — Mitigations (keep these in mind while building, not as reasons to cut)

- **Cross-modal registration (Phase 2):** ORB/SIFT keypoint matching is designed for same-modality images; IR and RGB have different appearance statistics (a warm road at night in IR isn't necessarily bright in RGB), so classical matching can be noisy. If it's not converging well, fall back to mutual-information-based alignment, or inject synthetic misalignment (small affine/elastic perturbation) into your paired data so the correction module has a controlled, guaranteed-solvable problem to demonstrate value on.
- **Multi-task detection (Phase 3):** generic COCO-pretrained YOLO doesn't transfer well to nadir/satellite views — different object categories and viewing angle. Prefer a detector pretrained on aerial imagery (DOTA, xView) if you can find and load one quickly; otherwise treat detection numbers as directional/qualitative rather than a hard benchmark claim.
- **PSNR/SSIM on synthetic pairs (Phase 1/6):** if your "ground truth" RGB is the same image you degraded to create synthetic IR, these metrics partly measure reconstruction of your own degradation function. Report this transparently, and if possible show a few qualitative results on real, unpaired IR images (no ground truth available) to demonstrate generalization beyond the synthetic distribution.
- **Multi-task loss balancing (Phase 3):** colorization loss and detection loss can dominate each other if summed naively. Log both loss curves separately during training so imbalance is caught early rather than after a full run.
- **Self-supervised pretraining (Phase 4):** the phase most likely to run long — masked reconstruction pretraining needs its own convergence before fine-tuning is meaningful. If Person C's track is behind at hour 27, capture the pretraining loss curve and a qualitative reconstruction example even without full fine-tuning comparison — it still supports the "designed for label efficiency" narrative.

---

## Results to Report (Pitch-Ready)

- [ ] Baseline PSNR / SSIM / LPIPS
- [ ] Registration ablation: with vs. without alignment
- [ ] Multi-task detection: real RGB vs. colorized output vs. raw IR (mAP or qualitative)
- [ ] Self-supervised: pretrained+fine-tuned vs. paired-only-small-subset
- [ ] Uncertainty heatmap demo image
- [ ] 3–4 best before/after comparison images
- [ ] *(If found)* qualitative results on real unpaired IR samples

---

## Pitch Narrative Order

1. Problem: IR satellite imagery is low-contrast, monochrome, hard to interpret
2. Why naive colorization fails: misalignment between sensors, label/data scarcity
3. Architecture: registration-aware, multi-task, self-supervised-ready pipeline
4. Ablation results across all four dimensions (registration, multi-task, self-supervision, uncertainty)
5. Downstream utility: detection improvement over raw IR
6. What's next: temporal/cross-sensor fusion as future work
7. Live or recorded demo

---

## Background Research Notes (Prior Work)

**Existing IR-to-RGB colorization research:**
- GAN-based thermal-to-RGB translation using coarse-to-fine generators with composite (content + adversarial + perceptual + total variation) loss
- U-Net-based generators with dense convolutional blocks, skip connections, and contrastive loss for IR colorization
- Multi-task deep neural networks performing simultaneous super-resolution + colorization on satellite imagery
- ML-based (linear/kernel/random forest regression, elastic maps) conversion of panchromatic nighttime satellite images to RGB
- Published survey/benchmark comparing DL colorization + super-resolution methods on IR/lidar imagery

**Implication:** the base architecture space (pix2pix/CycleGAN-style, U-Net+GAN, SRGAN) is well-established — differentiation comes from the registration-awareness, multi-task heads, and self-supervision layered on top, not from the base image-translation architecture itself.

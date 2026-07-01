# Self-Supervised Registration-Aware Multi-Task Fusion for Infrared Satellite Interpretation

## Project Framing

Instead of "IR to RGB translation," frame the project as:

> **"A self-supervised, registration-aware multi-task framework for infrared-to-visible satellite interpretation."**

This framing gives room to discuss:
- Robustness to misalignment
- Reduced dependence on labeled pairs
- Improved downstream interpretation (not just visual colorization)

---

## Phase 0: Before Writing Any Code (Day 0)

- [ ] **Confirm compute access** — Colab Pro, Kaggle (30 hrs/week free GPU), university cluster, or local GPU. This determines how ambitious the plan can be.
- [ ] **Lock the dataset** (max half a day — don't let this become a rabbit hole):
  - Search for VIIRS Day/Night Band + Sentinel-2/Landsat co-registered scenes
  - Check SpaceNet, SEN12MS, or any SAR-optical paired dataset as a fallback proxy
  - Prepare a synthetic-pairing fallback: degrade real RGB satellite images into IR-like grayscale using a luminance + noise + contrast curve, in case real pairs don't pan out
- [ ] **Set up repo/environment** — PyTorch, torchvision, OpenCV, rasterio/GDAL (if handling geotagged bands), ultralytics/YOLO (if going multi-task)
- [ ] **Write the one-paragraph framing** so the whole team is aligned from hour one (see title above)

---

## Phase 1: Baseline (Non-Negotiable Safety Net)

- [ ] Build U-Net generator + PatchGAN discriminator
- [ ] Implement loss: adversarial + L1 + VGG perceptual loss
- [ ] Get a training loop running end-to-end, even on ~100 image pairs
- [ ] Produce first before/after IR → RGB comparison image
- [ ] Compute baseline PSNR / SSIM / LPIPS on a held-out split

**Checkpoint goal:** by the end of this phase, you have something submittable even if everything else fails.

---

## Phase 2: Registration-Aware Fusion (First Differentiator — Cheap, High Payoff)

- [ ] Implement classical alignment: ORB or SIFT keypoint matching + homography estimation (OpenCV) between IR and RGB pairs
- [ ] Warp IR input to align with RGB before feeding into the network
- [ ] Retrain baseline model with aligned vs. unaligned data
- [ ] Build the **ablation table**: PSNR/SSIM with registration vs. without — this is a concrete result slide
- [ ] *(Optional stretch, only if Phase 2 finishes early)* Try a lightweight learned Spatial Transformer Network (STN) as an upgrade and compare against classical alignment

---

## Phase 3: Multi-Task Head (Second Differentiator — Most Judge-Visible Payoff)

- [ ] Add a second decoder head off the same encoder for object detection (or reuse a pretrained YOLO on colorized outputs if full multi-task training is too heavy)
- [ ] Train jointly: colorization loss + detection loss, weighted combination
- [ ] Run detection (mAP) on: real RGB ground truth vs. colorized output vs. raw IR input — **this three-way comparison is the strongest, most quotable result**
- [ ] Package as: *"colorization improves downstream object detection by X% mAP over raw IR"*

---

## Phase 4: Self-Supervised Pretraining (Stretch Goal — Only If Time Remains)

- [ ] Pretrain encoder on unlabeled IR images using a simple pretext task — masked patch reconstruction is easiest to implement correctly under time pressure (simpler than contrastive learning)
- [ ] Fine-tune the pretrained encoder on the small paired dataset
- [ ] Compare: paired-data-only training vs. pretrained+fine-tuned, on a deliberately small paired subset (e.g., 20% of pairs) — demonstrates label-efficiency, the key claim
- [ ] If this doesn't converge cleanly by the internal deadline, **drop it and soften the title/claims** rather than presenting something broken

---

## Phase 5: Uncertainty Estimation (Cheap Add-On If Phase 3 Finishes With Spare Time)

- [ ] Enable dropout layers in the generator, run inference N times (MC Dropout) at test time
- [ ] Compute pixel-wise variance across runs → produce an uncertainty heatmap overlay
- [ ] Show one demo image: IR input → colorized output → uncertainty map, side by side

---

## Phase 6: Packaging for Submission/Pitch

- [ ] Build a results table: PSNR/SSIM/LPIPS across baseline vs. each added component (this ablation table *is* the novelty proof)
- [ ] Pick 3–4 best-looking before/after image comparisons for the demo slide
- [ ] Write the pitch narrative in this order:
  1. Problem
  2. Why naive colorization fails (misalignment, label scarcity)
  3. Your architecture
  4. Ablation results
  5. Downstream utility (detection improvement)
  6. What's next (temporal/cross-sensor fusion as future work)
- [ ] Prepare a live or recorded demo — a working notebook or simple Gradio/Streamlit interface where you drop in an IR image and see the colorized + uncertainty output is far more convincing live than static slides
- [ ] One slide explicitly stating what's implemented vs. what's "designed for but future work" (temporal/cross-sensor fusion) — judges respect honesty about scope more than vague overclaiming

---

## Priority Order If Time Runs Short

**Phase 1 (baseline) → Phase 2 (registration) → Phase 3 (multi-task detection) → Phase 6 (packaging) → Phase 4/5 (stretch, whichever has more remaining time)**

---

## Reference: "Outside the Crowd" Ideas Considered

| Idea | Status | Notes |
|---|---|---|
| Registration-aware fusion | **In plan (Phase 2)** | Classical alignment first; STN as stretch upgrade |
| Multi-task learning (enhancement + colorization + detection) | **In plan (Phase 3)** | Best ROI — reuses infrastructure, gives a concrete mAP number |
| Self-supervised pretraining | **In plan (Phase 4, stretch)** | Highest payoff for label scarcity, but most time/compute hungry |
| Uncertainty estimation | **In plan (Phase 5, stretch)** | Cheap to add via MC Dropout; strong "wow factor" if time allows |
| Temporal fusion | Deferred — future work | Needs multi-pass imagery data not yet confirmed available |
| Cross-sensor fusion | Deferred — future work | Needs auxiliary sensor data not yet confirmed available |
| Downstream decision layer (road extraction, change detection, etc.) | Deferred — future work | Good extension beyond detection if time allows |

---

## Background Research Notes (Prior Work)

**Existing IR-to-RGB colorization research:**
- GAN-based thermal-to-RGB translation using coarse-to-fine generators with composite (content + adversarial + perceptual + total variation) loss
- U-Net-based generators with dense convolutional blocks, skip connections, and contrastive loss for IR colorization
- Multi-task deep neural networks performing simultaneous super-resolution + colorization on satellite imagery
- ML-based (linear/kernel/random forest regression, elastic maps) conversion of panchromatic nighttime satellite images to RGB
- Published survey/benchmark comparing DL colorization + super-resolution methods on IR/lidar imagery

**Implication:** the base architecture space (pix2pix/CycleGAN-style, U-Net+GAN, SRGAN) is well-established — differentiation must come from the domain-specific losses, registration-awareness, multi-task heads, and self-supervision layered on top, not from the base image-translation architecture itself.

"""Classical IR-RGB alignment via ORB/SIFT + homography."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _to_uint8_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = img[..., 0] if img.shape[-1] == 1 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if img.dtype != np.uint8:
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


def _to_uint8_rgb(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.uint8:
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img


def align_ir_to_rgb(
    ir: np.ndarray,
    rgb: np.ndarray,
    method: str = "orb",
    max_features: int = 500,
    ransac_threshold: float = 5.0,
) -> tuple[np.ndarray, bool]:
    """
    Warp IR to align with RGB using keypoint matching + homography.

    Returns (aligned_ir, success). On failure, returns original IR unchanged.
    """
    ir_gray = _to_uint8_gray(ir)
    rgb_gray = cv2.cvtColor(_to_uint8_rgb(rgb), cv2.COLOR_RGB2GRAY)

    if method.lower() == "sift":
        detector = cv2.SIFT_create(nfeatures=max_features)
    else:
        detector = cv2.ORB_create(nfeatures=max_features)

    kp1, des1 = detector.detectAndCompute(ir_gray, None)
    kp2, des2 = detector.detectAndCompute(rgb_gray, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return ir, False

    norm = cv2.NORM_HAMMING if method.lower() == "orb" else cv2.NORM_L2
    matcher = cv2.BFMatcher(norm, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda m: m.distance)

    if len(matches) < 4:
        return ir, False

    pts_ir = np.float32([kp1[m.queryIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)
    pts_rgb = np.float32([kp2[m.trainIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts_ir, pts_rgb, cv2.RANSAC, ransac_threshold)
    if H is None or mask is None or mask.sum() < 4:
        return ir, False

    h, w = rgb_gray.shape[:2]
    if ir.ndim == 3 and ir.shape[-1] == 1:
        aligned = cv2.warpPerspective(ir[..., 0], H, (w, h), borderMode=cv2.BORDER_REFLECT)
        aligned = aligned[..., np.newaxis]
    else:
        aligned = cv2.warpPerspective(ir, H, (w, h), borderMode=cv2.BORDER_REFLECT)

    return aligned.astype(np.float32), True


def align_batch_ir(
    ir_batch: np.ndarray,
    rgb_batch: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, list[bool]]:
    """Align a batch of IR images to their RGB counterparts."""
    reg_cfg = config.get("registration", {})
    method = reg_cfg.get("method", "orb")
    max_features = reg_cfg.get("max_features", 500)
    ransac_threshold = reg_cfg.get("ransac_threshold", 5.0)

    aligned_list: list[np.ndarray] = []
    success_list: list[bool] = []
    for i in range(ir_batch.shape[0]):
        aligned, ok = align_ir_to_rgb(
            ir_batch[i],
            rgb_batch[i],
            method=method,
            max_features=max_features,
            ransac_threshold=ransac_threshold,
        )
        aligned_list.append(aligned)
        success_list.append(ok)

    return np.stack(aligned_list), success_list

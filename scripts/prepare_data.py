#!/usr/bin/env python3
"""Download sample satellite RGB images for synthetic IR pairing."""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

# Small public-domain / CC0 satellite-style samples (placeholder URLs)
SAMPLE_URLS = [
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/Golden_Gate_Bridge%2C_San_Francisco_%28HDR%29.jpg/640px-Golden_Gate_Bridge%2C_San_Francisco_%28HDR%29.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a6/Paris_skyline_from_Gallery_of_Paleontology_and_Comparative_Anatomy%2C_Jardin_des_plantes%2C_Paris%2C_France_2.jpg/640px-Paris_skyline_from_Gallery_of_Paleontology_and_Comparative_Anatomy%2C_Jardin_des_plantes%2C_Paris%2C_France_2.jpg",
]


def download_samples(output_dir: Path, count: int = 20) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for i in range(count):
        url = SAMPLE_URLS[i % len(SAMPLE_URLS)]
        ext = ".jpg"
        out_path = output_dir / f"sample_{i:03d}{ext}"
        if out_path.exists():
            saved += 1
            continue
        try:
            urllib.request.urlretrieve(url, out_path)
            saved += 1
            print(f"Downloaded {out_path.name}")
        except Exception as e:
            print(f"Failed {url}: {e}", file=sys.stderr)
    return saved


def generate_synthetic_fallback(output_dir: Path, count: int = 50) -> int:
    """Generate procedural RGB patches if downloads fail."""
    import numpy as np
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        path = output_dir / f"synthetic_{i:03d}.png"
        if path.exists():
            continue
        h, w = 512, 512
        x = np.linspace(0, 1, w)
        y = np.linspace(0, 1, h)
        xx, yy = np.meshgrid(x, y)
        r = np.clip(0.3 + 0.4 * np.sin(xx * 10 + i) + 0.2 * yy, 0, 1)
        g = np.clip(0.2 + 0.5 * np.cos(yy * 8 + i * 0.3), 0, 1)
        b = np.clip(0.4 + 0.3 * np.sin((xx + yy) * 6), 0, 1)
        rgb = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
        Image.fromarray(rgb).save(path)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare RGB training data")
    parser.add_argument("--output", type=str, default="data/rgb")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--synthetic-only", action="store_true")
    args = parser.parse_args()

    out = Path(args.output)
    if args.synthetic_only:
        n = generate_synthetic_fallback(out, args.count)
        print(f"Generated {n} synthetic RGB images in {out}")
    else:
        n = download_samples(out, min(args.count, 10))
        if n < args.count:
            print(f"Only {n} downloads succeeded; generating synthetic fallback...")
            generate_synthetic_fallback(out, args.count)
        print(f"Data ready in {out} ({len(list(out.glob('*')))} files)")


if __name__ == "__main__":
    main()

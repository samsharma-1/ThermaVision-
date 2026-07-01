#!/usr/bin/env python3
"""Three-way detection comparison placeholder (Phase 3)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    """
    Run YOLO detection on real RGB vs colorized vs raw IR.

    Requires a trained checkpoint and ultralytics. Detection mAP on satellite
    imagery is directional — prefer aerial-pretrained weights when available.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint_latest.pt")
    parser.add_argument("--output", type=str, default="results/detection_comparison.json")
    args = parser.parse_args()

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    # Scaffold result structure for pitch deck
    placeholder = {
        "note": "Replace with real mAP after running YOLO on exported images",
        "real_rgb": {"mAP": None, "qualitative": "baseline"},
        "colorized": {"mAP": None, "qualitative": "expected improvement over IR"},
        "raw_ir": {"mAP": None, "qualitative": "lowest — monochrome, low contrast"},
        "next_steps": [
            "Export test-set predictions to results/predictions/{rgb,colorized,ir}/",
            "Run ultralytics YOLOv8 with aerial-pretrained weights (DOTA/xView)",
            "Fill mAP values into this JSON for the three-way slide",
        ],
    }
    with output.open("w") as f:
        json.dump(placeholder, f, indent=2)
    print(f"Detection comparison scaffold written to {output}")
    print("Implement full pipeline once baseline colorization checkpoint exists.")


if __name__ == "__main__":
    main()

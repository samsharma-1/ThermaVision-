from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device(config: dict[str, Any]) -> str:
    import torch

    requested = config.get("project", {}).get("device", "cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested

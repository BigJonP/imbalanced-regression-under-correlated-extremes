"""Load and validate run configs. One YAML file per run."""

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Read one run's YAML config. Every config must carry an integer `seed`."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"{path}: config must be a mapping, got {type(config).__name__}")
    validate_config(config, source=str(path))
    return config


def validate_config(config: dict[str, Any], source: str = "<dict>") -> None:
    seed = config.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"{source}: config needs an integer `seed`, got {seed!r}")

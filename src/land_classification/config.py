"""Config loader. Single source of truth: configs/default.yaml."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["_project_root"] = str(PROJECT_ROOT)
    return cfg


def ensure_dirs(cfg: dict[str, Any]) -> None:
    root = Path(cfg["_project_root"])
    for key in ("metrics_dir", "figures_dir", "predictions_dir"):
        (root / cfg["results"][key]).mkdir(parents=True, exist_ok=True)
    (root / cfg["data"]["root"]).mkdir(parents=True, exist_ok=True)

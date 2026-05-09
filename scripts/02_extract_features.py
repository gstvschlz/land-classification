"""Extract hand-crafted features for the traditional baselines and cache them."""
from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from land_classification.config import load_config
from land_classification.data.dataset import EuroSATIndex
from land_classification.data.features import cache_features


def main():
    cfg = load_config()
    data_dir = ROOT / cfg["data"]["root"] / "EuroSAT_RGB"
    idx = EuroSATIndex(data_dir)
    out = ROOT / cfg["features"]["cache_path"]
    X, y = cache_features(
        idx.paths, idx.labels, out_path=out,
        hsv_bins=cfg["features"]["hsv_bins"],
        glcm_distances=cfg["features"]["glcm_distances"],
        glcm_angles_deg=cfg["features"]["glcm_angles_deg"],
    )
    print(f"Wrote {X.shape} features to {out}; labels {y.shape}")


if __name__ == "__main__":
    main()

"""Download EuroSAT-RGB and write the deterministic stratified-fold indices."""
from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from land_classification.config import load_config, ensure_dirs
from land_classification.data.download import ensure_eurosat
from land_classification.data.dataset import EuroSATIndex, make_folds


def main():
    cfg = load_config()
    ensure_dirs(cfg)
    data_dir = ensure_eurosat(Path(ROOT) / cfg["data"]["root"])
    print(f"Dataset ready at {data_dir}")

    index = EuroSATIndex(data_dir)
    print(f"Total samples: {len(index)}; classes: {index.class_names}")

    folds_path = ROOT / cfg["results"]["root"] / "folds.json"
    folds = make_folds(index.labels, n_folds=cfg["n_folds"], seed=cfg["seed"],
                       save_to=folds_path)
    sizes = [(len(tr), len(te)) for tr, te in folds]
    print(f"Wrote {len(folds)} folds to {folds_path}; sizes={sizes}")


if __name__ == "__main__":
    main()

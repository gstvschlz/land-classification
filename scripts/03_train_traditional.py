"""Train Random Forest + XGBoost over the shared 5 folds."""
from __future__ import annotations
import argparse, sys, copy
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from land_classification.config import load_config
from land_classification.data.dataset import EuroSATIndex, load_folds
from land_classification.models.traditional import fit_random_forest, fit_xgboost
from land_classification.training.cv import run_cv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="2 folds, light hyperparams (smoke test)")
    args = parser.parse_args()

    cfg = load_config()
    data_dir = ROOT / cfg["data"]["root"] / "EuroSAT_RGB"
    feat_path = ROOT / cfg["features"]["cache_path"]
    folds_path = ROOT / cfg["results"]["root"] / "folds.json"

    idx = EuroSATIndex(data_dir)
    data = np.load(feat_path)
    X, y = data["X"], data["y"]
    assert X.shape[0] == len(idx), "feature cache out of sync with dataset"
    folds = load_folds(folds_path)
    if args.fast:
        folds = folds[:2]

    out_dir = ROOT / cfg["results"]["metrics_dir"]
    seed = cfg["seed"]
    nc = cfg["data"]["num_classes"]

    # ----- Random Forest -----
    rf_params = copy.deepcopy(cfg["models"]["random_forest"])
    if args.fast:
        rf_params["n_estimators"] = 50

    def run_rf(k, tr_idx, te_idx):
        m = fit_random_forest(X[tr_idx], y[tr_idx], rf_params, seed=seed)
        probs = m.predict_proba(X[te_idx])
        return probs, y[te_idx], {}, {
            "train_seconds": m.train_seconds, "n_params": m.n_params,
        }

    run_cv("RandomForest", folds, run_rf, num_classes=nc, out_dir=out_dir)

    # ----- XGBoost -----
    def run_xgb(k, tr_idx, te_idx):
        params = copy.deepcopy(cfg["models"]["xgboost"])
        if args.fast:
            params["n_estimators"] = 100
            params["early_stopping_rounds"] = 20
        # internal val split for early stopping
        tr_sub, va_sub = train_test_split(
            tr_idx, test_size=0.15, stratify=y[tr_idx], random_state=seed,
        )
        m = fit_xgboost(X[tr_sub], y[tr_sub], X[va_sub], y[va_sub],
                        params, seed=seed, num_classes=nc)
        probs = m.predict_proba(X[te_idx])
        return probs, y[te_idx], {}, {
            "train_seconds": m.train_seconds, "n_params": m.n_params,
        }

    run_cv("XGBoost", folds, run_xgb, num_classes=nc, out_dir=out_dir)


if __name__ == "__main__":
    main()

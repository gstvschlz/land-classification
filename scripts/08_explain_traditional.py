"""Compute feature importance / SHAP for RF and XGB on fold-0.

Refits the two traditional models on fold-0 (cheap) so we can hold the raw
estimators in memory for permutation importance + SHAP. Outputs:

  results/explanations/{RandomForest,XGBoost}/importances.json
  results/explanations/{RandomForest,XGBoost}/shap.npz
  results/figures/feature_importance_{Model}.png
  results/figures/group_importance_{Model}.png
  results/figures/shap_summary_{Model}.png
"""
from __future__ import annotations
import argparse, copy, json, sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from land_classification.config import load_config
from land_classification.data.dataset import load_folds
from land_classification.models.traditional import fit_random_forest, fit_xgboost
from land_classification.explain.traditional import (
    feature_layout, builtin_importance, permutation_importances,
    shap_values_tree, group_importances, stratified_subsample,
)
from land_classification.evaluation.plots import (
    feature_importance_bars, group_importance_bars, shap_summary,
)


class _XGBPredictAdapter:
    """sklearn-style estimator that delegates to the XGB FittedModel.

    Required because permutation_importance expects ``.predict`` and works
    on raw (un-scaled) X — our FittedModel.predict does the scaling.
    """
    def __init__(self, fitted):
        self._f = fitted
        self._classes = np.unique(fitted.estimator.classes_) \
            if hasattr(fitted.estimator, "classes_") else None

    def fit(self, X, y):  # required by sklearn interface
        return self

    def predict(self, X):
        return self._f.predict(X)

    def score(self, X, y):
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    feat_path = ROOT / cfg["features"]["cache_path"]
    folds_path = ROOT / cfg["results"]["root"] / "folds.json"
    nc = cfg["data"]["num_classes"]
    seed = cfg["seed"]

    data = np.load(feat_path)
    X, y = data["X"], data["y"]
    folds = load_folds(folds_path)
    tr_idx, te_idx = folds[0]

    feat_names, feat_groups = feature_layout(
        hsv_bins=cfg["features"]["hsv_bins"],
        glcm_angles_deg=tuple(cfg["features"]["glcm_angles_deg"]),
    )
    assert len(feat_names) == X.shape[1]

    out_root = ROOT / cfg["results"]["root"] / "explanations"
    out_root.mkdir(parents=True, exist_ok=True)
    fig_dir = ROOT / cfg["results"]["figures_dir"]
    fig_dir.mkdir(parents=True, exist_ok=True)

    n_perm = 3 if args.fast else cfg["explain"]["permutation_repeats"]
    n_shap = 400 if args.fast else cfg["explain"]["shap_max_samples"]

    # ------------------------------------------------------------------ RF
    rf_params = copy.deepcopy(cfg["models"]["random_forest"])
    if args.fast:
        rf_params["n_estimators"] = 80
    print(f"[RF] fitting fold-0 ({len(tr_idx)} samples)…")
    rf = fit_random_forest(X[tr_idx], y[tr_idx], rf_params, seed=seed)

    print("[RF] built-in importance")
    rf_builtin = builtin_importance(rf.estimator, X.shape[1])
    print(f"[RF] permutation importance (n_repeats={n_perm})")
    rf_perm = permutation_importances(
        rf.estimator, X[te_idx], y[te_idx],
        n_repeats=n_perm, seed=seed, n_jobs=-1,
    )
    rf_perm_mean = np.array(rf_perm["importances_mean"], dtype=np.float64)

    print(f"[RF] SHAP TreeExplainer (subsample ≤ {n_shap})")
    sub_idx_te = stratified_subsample(X[te_idx], y[te_idx], n_shap, seed=seed)
    Xs = X[te_idx][sub_idx_te]
    ys = y[te_idx][sub_idx_te]
    rf_shap = shap_values_tree(rf.estimator, Xs)

    rf_dir = out_root / "RandomForest"; rf_dir.mkdir(exist_ok=True)
    with open(rf_dir / "importances.json", "w") as f:
        json.dump({
            "feature_names": feat_names, "feature_groups": feat_groups,
            "builtin": rf_builtin.tolist(),
            "permutation_mean": rf_perm["importances_mean"],
            "permutation_std": rf_perm["importances_std"],
            "shap_mean_abs": rf_shap["mean_abs_per_feature"],
            "group_builtin": group_importances(rf_builtin, feat_groups),
            "group_permutation": group_importances(rf_perm_mean, feat_groups),
            "group_shap": group_importances(
                np.array(rf_shap["mean_abs_per_feature"]), feat_groups,
            ),
        }, f, indent=2)
    np.savez_compressed(rf_dir / "shap.npz",
                        signed_avg=rf_shap["signed_avg"],
                        X=Xs, y=ys)

    feature_importance_bars(
        rf_builtin, feat_names, feat_groups,
        fig_dir / "feature_importance_RandomForest.png",
        title="Random Forest — importance (Gini, top 25)", top_k=25,
    )
    group_importance_bars(
        group_importances(rf_perm_mean, feat_groups),
        fig_dir / "group_importance_RandomForest.png",
        title="Random Forest — importance by group (permutation)",
    )
    shap_summary(
        rf_shap["signed_avg"], Xs, feat_names,
        fig_dir / "shap_summary_RandomForest.png",
        title="Random Forest — SHAP (mean over classes)", top_k=20,
    )

    # ------------------------------------------------------------------ XGB
    xgb_params = copy.deepcopy(cfg["models"]["xgboost"])
    if args.fast:
        xgb_params["n_estimators"] = 200
        xgb_params["early_stopping_rounds"] = 20
    print(f"[XGB] fitting fold-0 ({len(tr_idx)} samples)…")
    tr_sub, va_sub = train_test_split(
        tr_idx, test_size=0.15, stratify=y[tr_idx], random_state=seed,
    )
    xg = fit_xgboost(X[tr_sub], y[tr_sub], X[va_sub], y[va_sub],
                     xgb_params, seed=seed, num_classes=nc)

    print("[XGB] built-in importance (gain)")
    xg_builtin = builtin_importance(xg.estimator, X.shape[1])
    print(f"[XGB] permutation importance (n_repeats={n_perm})")
    xg_perm = permutation_importances(
        _XGBPredictAdapter(xg), X[te_idx], y[te_idx],
        n_repeats=n_perm, seed=seed, n_jobs=1,  # avoid pickling the booster
    )
    xg_perm_mean = np.array(xg_perm["importances_mean"], dtype=np.float64)

    print(f"[XGB] SHAP TreeExplainer (subsample ≤ {n_shap})")
    sub_idx_te = stratified_subsample(X[te_idx], y[te_idx], n_shap, seed=seed)
    # XGB was trained on scaled features
    Xs_raw = X[te_idx][sub_idx_te]
    ys = y[te_idx][sub_idx_te]
    Xs = xg.preprocess.transform(Xs_raw).astype(np.float32)
    xg_shap = shap_values_tree(xg.estimator, Xs)

    xg_dir = out_root / "XGBoost"; xg_dir.mkdir(exist_ok=True)
    with open(xg_dir / "importances.json", "w") as f:
        json.dump({
            "feature_names": feat_names, "feature_groups": feat_groups,
            "builtin": xg_builtin.tolist(),
            "permutation_mean": xg_perm["importances_mean"],
            "permutation_std": xg_perm["importances_std"],
            "shap_mean_abs": xg_shap["mean_abs_per_feature"],
            "group_builtin": group_importances(xg_builtin, feat_groups),
            "group_permutation": group_importances(xg_perm_mean, feat_groups),
            "group_shap": group_importances(
                np.array(xg_shap["mean_abs_per_feature"]), feat_groups,
            ),
        }, f, indent=2)
    np.savez_compressed(xg_dir / "shap.npz",
                        signed_avg=xg_shap["signed_avg"],
                        X=Xs_raw, y=ys)  # raw X for plotting interpretability

    feature_importance_bars(
        xg_builtin, feat_names, feat_groups,
        fig_dir / "feature_importance_XGBoost.png",
        title="XGBoost — importance (gain, top 25)", top_k=25,
    )
    group_importance_bars(
        group_importances(xg_perm_mean, feat_groups),
        fig_dir / "group_importance_XGBoost.png",
        title="XGBoost — importance by group (permutation)",
    )
    shap_summary(
        xg_shap["signed_avg"], Xs_raw, feat_names,
        fig_dir / "shap_summary_XGBoost.png",
        title="XGBoost — SHAP (mean over classes)", top_k=20,
    )

    print("Done. Wrote explanations under", out_root)


if __name__ == "__main__":
    main()

"""Explainability helpers for the traditional baselines.

The 87-dim feature vector comes from :mod:`land_classification.data.features`.
We mirror its layout here so feature names / groups stay in sync with the
extractor — never hard-code, always derive from the same constants.
"""
from __future__ import annotations
from typing import Any
import numpy as np
from sklearn.inspection import permutation_importance

from ..data.features import GLCM_PROPS, feature_dim


# Display order matches the concatenation in ``features.extract_one``.
GROUP_ORDER = ("rgb_stats", "hsv_hist", "glcm", "hu", "edge", "vari")


def feature_layout(hsv_bins: int = 16,
                   glcm_angles_deg: tuple[float, ...] = (0, 45, 90, 135),
                   ) -> tuple[list[str], list[str]]:
    """Return (feature_names, feature_groups) of length ``feature_dim``."""
    names: list[str] = []
    groups: list[str] = []

    # 1) RGB stats: mean, std, skew x R, G, B
    for stat in ("mean", "std", "skew"):
        for ch in ("R", "G", "B"):
            names.append(f"{ch}_{stat}")
            groups.append("rgb_stats")

    # 2) HSV histogram: H, S, V x bins
    for ch in ("H", "S", "V"):
        for b in range(hsv_bins):
            names.append(f"{ch}_bin{b:02d}")
            groups.append("hsv_hist")

    # 3) GLCM: prop x angle (averaged over distances)
    for prop in GLCM_PROPS:
        for ang in glcm_angles_deg:
            names.append(f"GLCM_{prop}_{int(ang)}deg")
            groups.append("glcm")

    # 4) Hu moments: 7
    for i in range(7):
        names.append(f"Hu_{i+1}")
        groups.append("hu")

    # 5) edge density: 1
    names.append("edge_density"); groups.append("edge")

    # 6) VARI mean / std: 2
    names.append("VARI_mean"); groups.append("vari")
    names.append("VARI_std"); groups.append("vari")

    expected = feature_dim(hsv_bins=hsv_bins, n_angles=len(glcm_angles_deg))
    assert len(names) == expected, (len(names), expected)
    return names, groups


def builtin_importance(estimator: Any, n_features: int) -> np.ndarray:
    """Return per-feature importance for RF (Gini) or XGB (gain)."""
    if hasattr(estimator, "feature_importances_") and \
            estimator.__class__.__name__ != "XGBClassifier":
        return np.asarray(estimator.feature_importances_, dtype=np.float64)
    # XGBoost: prefer gain via Booster, fall back to feature_importances_
    try:
        booster = estimator.get_booster()
        score = booster.get_score(importance_type="gain")
        out = np.zeros(n_features, dtype=np.float64)
        for k, v in score.items():
            # XGB names features as 'f{idx}' when no names supplied
            idx = int(k.lstrip("f"))
            if 0 <= idx < n_features:
                out[idx] = v
        if out.sum() > 0:
            return out / out.sum()
    except Exception:
        pass
    return np.asarray(estimator.feature_importances_, dtype=np.float64)


def permutation_importances(predict_callable, X: np.ndarray, y: np.ndarray,
                            n_repeats: int = 8, seed: int = 0,
                            n_jobs: int = -1) -> dict:
    """Wrap sklearn permutation_importance.

    ``predict_callable`` must be a thing with a ``.predict`` method (sklearn
    style). For our XGB wrapper that scales internally, pass an adapter.
    """
    res = permutation_importance(
        predict_callable, X, y,
        n_repeats=n_repeats, random_state=seed, n_jobs=n_jobs,
    )
    return {
        "importances_mean": res.importances_mean.astype(float).tolist(),
        "importances_std": res.importances_std.astype(float).tolist(),
    }


def stratified_subsample(X: np.ndarray, y: np.ndarray, max_n: int,
                         seed: int = 0) -> np.ndarray:
    """Return indices of a stratified subsample of size ≤ max_n."""
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    per_class = max(1, max_n // len(classes))
    idx_list = []
    for c, k in zip(classes, counts):
        all_idx = np.flatnonzero(y == c)
        take = min(per_class, len(all_idx))
        idx_list.append(rng.choice(all_idx, size=take, replace=False))
    out = np.concatenate(idx_list)
    rng.shuffle(out)
    return out[:max_n]


def shap_values_tree(estimator: Any, X_sample: np.ndarray,
                     max_classes_for_persample: int = 10) -> dict:
    """TreeExplainer SHAP. Returns dict with mean(|SHAP|) per (sample, feature)
    aggregated across classes (the array is what plots / downstream code use).

    Also returns per-class arrays for the user to drill into if desired.
    """
    import shap
    explainer = shap.TreeExplainer(estimator)
    sv = explainer.shap_values(X_sample)
    # Normalise possible return shapes:
    #   - list of arrays of shape (n, p), one per class (older sklearn / XGB)
    #   - 3D array of shape (n, p, n_classes) (newer xgboost / shap)
    #   - 2D (binary) — not expected here
    if isinstance(sv, list):
        per_class = [np.asarray(a, dtype=np.float64) for a in sv]
        stacked = np.stack(per_class, axis=-1)  # (n, p, c)
    else:
        stacked = np.asarray(sv, dtype=np.float64)
        if stacked.ndim == 2:
            stacked = stacked[:, :, None]
        per_class = [stacked[..., c] for c in range(stacked.shape[-1])]

    mean_abs_per_feature = np.mean(np.abs(stacked), axis=(0, 2))  # (p,)
    # 2D summary array averaged across classes (signed) for beeswarm:
    signed_avg = np.mean(stacked, axis=2)  # (n, p)
    return {
        "stacked": stacked,
        "per_class": per_class,
        "mean_abs_per_feature": mean_abs_per_feature.tolist(),
        "signed_avg": signed_avg,
    }


def group_importances(per_feature: np.ndarray,
                      groups: list[str]) -> dict[str, float]:
    out: dict[str, float] = {g: 0.0 for g in GROUP_ORDER}
    for v, g in zip(np.asarray(per_feature, dtype=np.float64), groups):
        out[g] = out.get(g, 0.0) + float(v)
    return out

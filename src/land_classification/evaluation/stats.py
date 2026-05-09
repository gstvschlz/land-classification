"""Significance tests between models using their per-sample predictions."""
from __future__ import annotations
import numpy as np
from scipy.stats import binomtest


def mcnemar_test(pred_a: np.ndarray, pred_b: np.ndarray,
                 y: np.ndarray) -> dict:
    """Exact McNemar test on disagreement counts (b, c).

    b = #samples model A right, model B wrong; c = the reverse.
    Under H0 (equal accuracy) b ~ Binomial(b+c, 0.5).
    """
    a_right = pred_a == y
    b_right = pred_b == y
    b = int(((a_right) & (~b_right)).sum())
    c = int(((~a_right) & (b_right)).sum())
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p_value": 1.0, "statistic": 0.0}
    p_value = float(binomtest(b, n=n, p=0.5, alternative="two-sided").pvalue)
    statistic = (abs(b - c) - 1) ** 2 / max(1, b + c)  # continuity-corrected chi^2
    return {"b": b, "c": c, "p_value": p_value, "statistic": float(statistic)}


def paired_bootstrap_accuracy(pred_a: np.ndarray, pred_b: np.ndarray,
                              y: np.ndarray, n_iter: int = 2000,
                              seed: int = 0) -> dict:
    """Paired bootstrap on per-sample (correct_a - correct_b)."""
    rng = np.random.default_rng(seed)
    diff = (pred_a == y).astype(np.float64) - (pred_b == y).astype(np.float64)
    n = len(diff)
    idx = rng.integers(0, n, size=(n_iter, n))
    samples = diff[idx].mean(axis=1)
    obs = float(diff.mean())
    lo, hi = float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))
    # two-sided p-value: probability that resampled mean has opposite sign
    p_value = float(2.0 * min((samples >= 0).mean(), (samples <= 0).mean()))
    return {"diff_acc": obs, "ci_low": lo, "ci_high": hi,
            "p_value": min(1.0, p_value)}


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Holm step-down adjustment. Returns adjusted p-values in original order."""
    p = np.asarray(p_values, dtype=np.float64)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, p[i] * (n - rank))
        adj[i] = min(1.0, running)
    return adj.tolist()

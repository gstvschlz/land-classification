"""Strict classification metrics with bootstrap confidence intervals."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, cohen_kappa_score,
    precision_recall_fscore_support, roc_auc_score, confusion_matrix,
    top_k_accuracy_score,
)


@dataclass
class FoldMetrics:
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    cohen_kappa: float
    top2_accuracy: float
    macro_auroc: float
    per_class_f1: list[float]
    per_class_precision: list[float]
    per_class_recall: list[float]
    confusion: list[list[int]]
    ece: float

    def to_dict(self) -> dict:
        return asdict(self)


def expected_calibration_error(probs: np.ndarray, y: np.ndarray,
                               n_bins: int = 15) -> float:
    """Standard ECE across max-confidence bins."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = conf[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def compute_fold_metrics(probs: np.ndarray, y: np.ndarray,
                         n_classes: int) -> FoldMetrics:
    pred = probs.argmax(axis=1)
    p, r, f, _ = precision_recall_fscore_support(
        y, pred, labels=list(range(n_classes)), zero_division=0,
    )
    try:
        auroc = roc_auc_score(y, probs, multi_class="ovr",
                              labels=list(range(n_classes)), average="macro")
    except ValueError:
        auroc = float("nan")
    return FoldMetrics(
        accuracy=float(accuracy_score(y, pred)),
        balanced_accuracy=float(balanced_accuracy_score(y, pred)),
        macro_f1=float(f1_score(y, pred, average="macro", zero_division=0)),
        cohen_kappa=float(cohen_kappa_score(y, pred)),
        top2_accuracy=float(top_k_accuracy_score(y, probs, k=2,
                                                 labels=list(range(n_classes)))),
        macro_auroc=float(auroc),
        per_class_f1=[float(x) for x in f],
        per_class_precision=[float(x) for x in p],
        per_class_recall=[float(x) for x in r],
        confusion=confusion_matrix(y, pred,
                                   labels=list(range(n_classes))).tolist(),
        ece=expected_calibration_error(probs, y),
    )


def bootstrap_ci(values: np.ndarray, alpha: float = 0.05,
                 n_iter: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    samples = rng.choice(values, size=(n_iter, n), replace=True)
    means = samples.mean(axis=1)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi

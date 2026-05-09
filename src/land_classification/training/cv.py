"""Cross-validation orchestrator.

A single source of truth for the CV loop. Each model type plugs in via a
``run_fold`` callable that, given the train/val/test indices for a fold,
returns ``(test_probs, test_labels, history_dict, extra_metadata)``.
"""
from __future__ import annotations
from pathlib import Path
import json
from typing import Callable
import numpy as np

from ..evaluation.metrics import compute_fold_metrics, FoldMetrics


FoldCallable = Callable[
    [int, np.ndarray, np.ndarray],  # fold_idx, train_idx, test_idx
    tuple[np.ndarray, np.ndarray, dict, dict],
]


def run_cv(model_name: str, folds: list[tuple[np.ndarray, np.ndarray]],
           run_fold: FoldCallable, *, num_classes: int,
           out_dir: str | Path) -> dict:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    pred_dir = out / model_name
    pred_dir.mkdir(parents=True, exist_ok=True)
    fold_metrics: list[dict] = []
    histories: list[dict] = []
    extras: list[dict] = []

    for k, (tr_idx, te_idx) in enumerate(folds):
        print(f"\n=== {model_name} :: fold {k+1}/{len(folds)} "
              f"(train={len(tr_idx)} test={len(te_idx)}) ===")
        probs, y_true, hist, extra = run_fold(k, tr_idx, te_idx)
        m: FoldMetrics = compute_fold_metrics(probs, y_true, num_classes)
        fold_metrics.append(m.to_dict())
        histories.append(hist)
        extras.append(extra)

        np.savez_compressed(pred_dir / f"fold{k}_predictions.npz",
                            probs=probs, y_true=y_true, test_idx=te_idx)
        print(f"  acc={m.accuracy:.4f}  macroF1={m.macro_f1:.4f}  "
              f"kappa={m.cohen_kappa:.4f}  AUROC={m.macro_auroc:.4f}")

    summary = {
        "model": model_name,
        "n_folds": len(folds),
        "folds": fold_metrics,
        "histories": histories,
        "extras": extras,
    }
    with open(out / f"{model_name}.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary

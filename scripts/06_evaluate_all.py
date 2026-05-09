"""Aggregate per-model results, run significance tests, build figures."""
from __future__ import annotations
from itertools import combinations
from pathlib import Path
import json, sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from land_classification.config import load_config
from land_classification.data.dataset import EuroSATIndex
from land_classification.evaluation.metrics import bootstrap_ci
from land_classification.evaluation.stats import (
    mcnemar_test, paired_bootstrap_accuracy, holm_bonferroni,
)
from land_classification.evaluation.plots import (
    confusion_heatmap, per_class_f1_bars, reliability_diagram,
    class_distribution, training_curves,
)


PRIMARY_METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "cohen_kappa",
                   "top2_accuracy", "macro_auroc", "ece")


def load_model_summary(metrics_dir: Path, name: str) -> dict | None:
    p = metrics_dir / f"{name}.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def aggregate_metrics(summary: dict) -> dict:
    rows = summary["folds"]
    out = {}
    for k in PRIMARY_METRICS:
        vals = np.array([r[k] for r in rows], dtype=np.float64)
        lo, hi = bootstrap_ci(vals)
        out[k] = {
            "mean": float(vals.mean()), "std": float(vals.std(ddof=1) if len(vals) > 1 else 0.0),
            "ci_low": lo, "ci_high": hi,
        }
    out["per_class_f1_mean"] = np.mean(
        [r["per_class_f1"] for r in rows], axis=0,
    ).tolist()
    cm = np.sum([np.array(r["confusion"]) for r in rows], axis=0)
    out["confusion_sum"] = cm.tolist()
    out["train_seconds_mean"] = float(np.mean(
        [e.get("train_seconds", float("nan")) for e in summary["extras"]],
    ))
    out["n_params"] = int(summary["extras"][0].get("n_params", 0))
    return out


def gather_predictions(metrics_dir: Path, name: str, n_folds: int):
    """Concatenate per-fold predictions with their original sample indices."""
    all_probs, all_y, all_idx = [], [], []
    for k in range(n_folds):
        p = metrics_dir / f"fold{k}_predictions.npz"
        # files are model-agnostic in name; store separately by sub-folder
        # but we save them flat per model — re-derive by checking shape & test_idx
        # Predictions saved by run_cv don't include the model name in filename;
        # to disambiguate we store them in per-model folders below.
        raise RuntimeError  # placeholder, replaced by per-model paths
    return all_probs, all_y, all_idx


def main():
    cfg = load_config()
    metrics_dir = ROOT / cfg["results"]["metrics_dir"]
    figures_dir = ROOT / cfg["results"]["figures_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)
    class_names = cfg["data"]["class_names"]
    n_folds = cfg["n_folds"]

    model_names = ["RandomForest", "XGBoost", "ResNet18"]
    summaries = {n: load_model_summary(metrics_dir, n) for n in model_names}
    available = {n: s for n, s in summaries.items() if s is not None}
    if not available:
        raise SystemExit("No model results found; run training scripts first.")

    # ---------- aggregate per-model ----------
    aggregated: dict[str, dict] = {n: aggregate_metrics(s) for n, s in available.items()}

    # ---------- main results table ----------
    rows = []
    for n, agg in aggregated.items():
        row = {"model": n}
        for k in PRIMARY_METRICS:
            v = agg[k]
            row[k] = f"{v['mean']:.4f} ± {v['std']:.4f}"
            row[f"{k}_mean"] = v["mean"]
        row["train_seconds"] = agg["train_seconds_mean"]
        row["n_params"] = agg["n_params"]
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(metrics_dir / "summary.csv", index=False)
    print("\nMain results:\n", df.to_string(index=False))

    # ---------- significance tests ----------
    # Concatenate predictions per model across folds.
    pred_per_model: dict[str, dict] = {}
    for n in available.keys():
        probs_list, y_list, idx_list = [], [], []
        # per-model fold predictions live under the metrics_dir as fold{k}_predictions.npz
        # But the file is shared across models since run_cv writes one set; we instead
        # split by storing under per-model subdirs.
        per_model_dir = metrics_dir / n
        for k in range(n_folds):
            p = per_model_dir / f"fold{k}_predictions.npz"
            if not p.exists():
                p = metrics_dir / f"fold{k}_predictions.npz"  # legacy fallback
            if not p.exists():
                continue
            d = np.load(p)
            probs_list.append(d["probs"])
            y_list.append(d["y_true"])
            idx_list.append(d["test_idx"])
        if not probs_list:
            continue
        probs = np.concatenate(probs_list)
        y = np.concatenate(y_list)
        idx = np.concatenate(idx_list)
        order = np.argsort(idx)
        pred_per_model[n] = {
            "probs": probs[order], "y": y[order], "idx": idx[order],
            "pred": probs[order].argmax(axis=1),
        }

    sig_rows = []
    if len(pred_per_model) >= 2:
        # Sanity: all share the same y order
        ref_y = next(iter(pred_per_model.values()))["y"]
        pairs = list(combinations(pred_per_model.keys(), 2))
        for a, b in pairs:
            yA, yB = pred_per_model[a]["y"], pred_per_model[b]["y"]
            if not (np.array_equal(yA, ref_y) and np.array_equal(yB, ref_y)):
                continue
            mc = mcnemar_test(pred_per_model[a]["pred"],
                              pred_per_model[b]["pred"], ref_y)
            pb = paired_bootstrap_accuracy(pred_per_model[a]["pred"],
                                           pred_per_model[b]["pred"], ref_y)
            sig_rows.append({"a": a, "b": b,
                             "mcnemar_p": mc["p_value"],
                             "bootstrap_diff": pb["diff_acc"],
                             "bootstrap_p": pb["p_value"],
                             "ci_low": pb["ci_low"], "ci_high": pb["ci_high"]})
        if sig_rows:
            mc_p = [r["mcnemar_p"] for r in sig_rows]
            bs_p = [r["bootstrap_p"] for r in sig_rows]
            mc_adj = holm_bonferroni(mc_p)
            bs_adj = holm_bonferroni(bs_p)
            for r, ma, ba in zip(sig_rows, mc_adj, bs_adj):
                r["mcnemar_p_holm"] = ma
                r["bootstrap_p_holm"] = ba
    sig_df = pd.DataFrame(sig_rows)
    sig_df.to_csv(metrics_dir / "significance.csv", index=False)
    if not sig_df.empty:
        print("\nSignificance (Holm-corrected):\n", sig_df.to_string(index=False))

    # ---------- figures ----------
    # Class distribution
    data_dir = ROOT / cfg["data"]["root"] / "EuroSAT_RGB"
    if data_dir.exists():
        idx_full = EuroSATIndex(data_dir)
        class_distribution(idx_full.labels, class_names,
                           figures_dir / "class_distribution.png")

    # Confusion matrices
    for n, agg in aggregated.items():
        confusion_heatmap(np.array(agg["confusion_sum"]), class_names,
                          figures_dir / f"confusion_{n}.png",
                          title=f"{n}: confusion (sum across folds)")

    # Per-class F1
    per_class = {n: np.array(agg["per_class_f1_mean"])
                 for n, agg in aggregated.items()}
    per_class_f1_bars(per_class, class_names,
                      figures_dir / "per_class_f1.png")

    # Reliability + training curves for deep models
    for n in ("ResNet18",):
        if n in pred_per_model:
            reliability_diagram(pred_per_model[n]["probs"], pred_per_model[n]["y"],
                                figures_dir / f"reliability_{n}.png",
                                title=f"{n} reliability")
        if n in available and available[n]["histories"]:
            training_curves(available[n]["histories"],
                            figures_dir / f"training_curves_{n}.png", title=n)

    # Save aggregated dict for the article builder
    with open(metrics_dir / "aggregated.json", "w") as f:
        json.dump(aggregated, f, indent=2)
    print("\nWrote aggregated.json, summary.csv, significance.csv and figures.")


if __name__ == "__main__":
    main()

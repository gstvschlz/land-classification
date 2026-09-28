"""Plotting helpers. All produce PNG files at a target path.

Every figure obeys the sober editorial style defined in :mod:`style`:
serif body, muted palette, light grid, no top/right spines. English labels.
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterable
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from .style import (
    apply_style, PALETTE, MODEL_COLORS, GROUP_COLORS,
    SEQUENTIAL_CMAP, DIVERGING_CMAP,
)

apply_style()


def _save(fig, out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# Headline evaluation plots                                                   #
# --------------------------------------------------------------------------- #

def confusion_heatmap(cm: np.ndarray, class_names: list[str],
                      out_path: str | Path, title: str) -> Path:
    cm = np.asarray(cm, dtype=np.float64)
    cm_norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(7.0, 5.8))
    im = ax.imshow(cm_norm, cmap=SEQUENTIAL_CMAP, vmin=0.0, vmax=1.0,
                   aspect="auto")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    ax.grid(False)
    # Annotate counts; light text on dark cells, dark text on light cells.
    threshold = 0.55
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm_norm[i, j]
            color = "#F4EFE6" if v > threshold else "#222222"
            ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center",
                    fontsize=8, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Row-normalised share", fontsize=9)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=8)
    return _save(fig, out_path)


def per_class_f1_bars(per_model_f1: dict[str, np.ndarray],
                      class_names: list[str],
                      out_path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    n_models = len(per_model_f1)
    width = 0.78 / max(1, n_models)
    x = np.arange(len(class_names))
    for i, (name, vals) in enumerate(per_model_f1.items()):
        color = MODEL_COLORS.get(name, PALETTE[i % len(PALETTE)])
        ax.bar(x + i * width, vals, width=width * 0.92,
               color=color, edgecolor="white", linewidth=0.4, label=name)
    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(class_names, rotation=35, ha="right")
    ax.set_ylabel("F1 (mean over folds)")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("F1 per class and model")
    ax.legend(loc="lower right", ncol=n_models, handlelength=1.2)
    ax.grid(axis="x", visible=False)
    return _save(fig, out_path)


def reliability_diagram(probs: np.ndarray, y: np.ndarray,
                        out_path: str | Path, title: str,
                        n_bins: int = 15) -> Path:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    accs = np.zeros(n_bins); confs = np.zeros(n_bins); weights = np.zeros(n_bins)
    for i in range(n_bins):
        if i == 0:
            mask = (conf >= bins[i]) & (conf <= bins[i + 1])
        else:
            mask = (conf > bins[i]) & (conf <= bins[i + 1])
        if mask.sum() > 0:
            accs[i] = correct[mask].mean()
            confs[i] = conf[mask].mean()
            weights[i] = mask.sum() / len(y)

    fig, (ax, axw) = plt.subplots(
        2, 1, figsize=(5.2, 5.4), sharex=True,
        gridspec_kw={"height_ratios": [4, 1], "hspace": 0.08},
    )
    ax.plot([0, 1], [0, 1], color="#888888", linewidth=0.9,
            linestyle="--", label="ideal")
    bar_w = 1.0 / n_bins * 0.92
    ax.bar(centers, accs, width=bar_w, color=MODEL_COLORS["ResNet18"],
           edgecolor="white", linewidth=0.4, label="accuracy", alpha=0.85)
    # gap = confidence - accuracy (positive = over-confident)
    gap = confs - accs
    for c, a, g in zip(centers, accs, gap):
        if abs(g) < 1e-9:
            continue
        ax.bar(c, g, width=bar_w, bottom=a, color="#A1505A", alpha=0.45,
               edgecolor="none")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.02); ax.set_xlim(0.0, 1.0)
    ax.set_title(title)
    handles = [
        Patch(facecolor=MODEL_COLORS["ResNet18"], label="accuracy"),
        Patch(facecolor="#A1505A", alpha=0.45, label="overconfidence"),
    ]
    ax.legend(handles=handles, loc="upper left")
    axw.bar(centers, weights, width=bar_w, color="#888888", alpha=0.55,
            edgecolor="none")
    axw.set_ylabel("Frac.")
    axw.set_xlabel("Confidence")
    axw.set_ylim(0, max(weights.max() * 1.1, 1e-3))
    return _save(fig, out_path)


def class_distribution(labels: np.ndarray, class_names: list[str],
                       out_path: str | Path) -> Path:
    counts = np.bincount(labels, minlength=len(class_names))
    order = np.argsort(-counts)
    fig, ax = plt.subplots(figsize=(8.0, 3.4))
    bars = ax.bar(range(len(class_names)), counts[order],
                  color=[PALETTE[i % len(PALETTE)] for i in order],
                  edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels([class_names[i] for i in order], rotation=35, ha="right")
    ax.set_ylabel("Imagens")
    ax.set_title("Class distribution — EuroSAT-RGB")
    for b, c in zip(bars, counts[order]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 30,
                str(int(c)), ha="center", va="bottom", fontsize=8,
                color="#444")
    ax.grid(axis="x", visible=False)
    ax.margins(x=0.01)
    return _save(fig, out_path)


def training_curves(history_per_fold: list[dict], out_path: str | Path,
                    title: str) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    n = len(history_per_fold)
    # per-fold thin lines + mean overlay
    losses_tr, losses_va, accs = [], [], []
    for k, h in enumerate(history_per_fold):
        epochs = np.arange(1, len(h["train_loss"]) + 1)
        col = PALETTE[k % len(PALETTE)]
        axes[0].plot(epochs, h["train_loss"], color=col, alpha=0.35, linewidth=1.0)
        axes[0].plot(epochs, h["val_loss"], color=col, alpha=0.35,
                     linewidth=1.0, linestyle="--")
        axes[1].plot(epochs, h["val_acc"], color=col, alpha=0.4, linewidth=1.0,
                     label=f"fold {k}")
        losses_tr.append(h["train_loss"]); losses_va.append(h["val_loss"])
        accs.append(h["val_acc"])
    if losses_tr:
        L = min(len(x) for x in losses_tr)
        ep = np.arange(1, L + 1)
        m_tr = np.mean([x[:L] for x in losses_tr], axis=0)
        m_va = np.mean([x[:L] for x in losses_va], axis=0)
        m_acc = np.mean([x[:L] for x in accs], axis=0)
        axes[0].plot(ep, m_tr, color="#222", linewidth=1.6, label="train (mean)")
        axes[0].plot(ep, m_va, color="#222", linewidth=1.6, linestyle="--",
                     label="val. (mean)")
        axes[1].plot(ep, m_acc, color="#222", linewidth=1.8, label="mean")
    axes[0].set_title(f"{title}: loss"); axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss"); axes[0].legend(handlelength=1.5)
    axes[1].set_title(f"{title}: accuracy (val.)"); axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].legend(ncol=2, handlelength=1.5)
    fig.tight_layout()
    return _save(fig, out_path)


# --------------------------------------------------------------------------- #
# Explainability plots                                                        #
# --------------------------------------------------------------------------- #

def feature_importance_bars(importances: np.ndarray, feature_names: list[str],
                            groups: list[str], out_path: str | Path,
                            title: str, top_k: int = 25) -> Path:
    """Horizontal bars of the top-K features, coloured by feature group."""
    importances = np.asarray(importances, dtype=np.float64)
    order = np.argsort(-importances)[:top_k][::-1]  # ascending for barh
    names = [feature_names[i] for i in order]
    vals = importances[order]
    cols = [GROUP_COLORS.get(groups[i], "#888") for i in order]

    fig, ax = plt.subplots(figsize=(6.5, max(3.5, 0.22 * top_k + 1.0)))
    ax.barh(range(len(order)), vals, color=cols, edgecolor="white",
            linewidth=0.4)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Importance")
    ax.set_title(title)
    used_groups = sorted(set(groups[i] for i in order))
    handles = [Patch(facecolor=GROUP_COLORS.get(g, "#888"), label=g)
               for g in used_groups]
    ax.legend(handles=handles, loc="lower right", ncol=2,
              handlelength=1.0, fontsize=7)
    ax.grid(axis="y", visible=False)
    return _save(fig, out_path)


def group_importance_bars(group_importances: dict[str, float],
                          out_path: str | Path, title: str) -> Path:
    """Aggregated importance per feature group (color/HSV/GLCM/Hu/edge/VARI)."""
    items = sorted(group_importances.items(), key=lambda kv: kv[1], reverse=True)
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    cols = [GROUP_COLORS.get(k, "#888") for k in names]
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    ax.bar(names, vals, color=cols, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Aggregated importance")
    ax.set_title(title)
    ax.grid(axis="x", visible=False)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    return _save(fig, out_path)


def shap_summary(shap_values_2d: np.ndarray, X: np.ndarray,
                 feature_names: list[str], out_path: str | Path,
                 title: str, top_k: int = 20) -> Path:
    """Custom beeswarm-ish summary that respects the editorial style.

    ``shap_values_2d`` is (n_samples, n_features) — for multiclass models
    callers should pass the mean(|SHAP|) across classes or the per-class slice.
    """
    sv = np.asarray(shap_values_2d)
    X = np.asarray(X)
    mean_abs = np.mean(np.abs(sv), axis=0)
    order = np.argsort(-mean_abs)[:top_k][::-1]
    fig, ax = plt.subplots(figsize=(6.8, max(3.5, 0.28 * top_k + 1.0)))
    rng = np.random.default_rng(0)
    for row, fi in enumerate(order):
        x = sv[:, fi]
        # color by feature value (normalised)
        col_raw = X[:, fi]
        rng_lo, rng_hi = np.percentile(col_raw, [2, 98])
        if rng_hi - rng_lo < 1e-9:
            t = np.zeros_like(col_raw)
        else:
            t = np.clip((col_raw - rng_lo) / (rng_hi - rng_lo), 0, 1)
        jitter = rng.uniform(-0.18, 0.18, size=len(x))
        ax.scatter(x, np.full_like(x, row, dtype=float) + jitter,
                   c=t, cmap=DIVERGING_CMAP, s=6, alpha=0.55,
                   linewidths=0)
    ax.axvline(0, color="#888", linewidth=0.6)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([feature_names[i] for i in order], fontsize=8)
    ax.set_xlabel("SHAP value (impact on model output)")
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    sm = plt.cm.ScalarMappable(cmap=DIVERGING_CMAP,
                               norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Valor da feature (baixo → alto)", fontsize=8)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=7)
    return _save(fig, out_path)


def gradcam_grid(images: np.ndarray, cams: np.ndarray,
                 labels_true: list[str], labels_pred: list[str],
                 confidences: list[float], out_path: str | Path,
                 title: str = "Grad-CAM") -> Path:
    """images: (N, H, W, 3) uint8; cams: (N, H, W) float in [0, 1]."""
    n = len(images)
    cols = 6
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.9, rows * 2.05))
    axes = np.atleast_2d(axes)
    for k in range(rows * cols):
        r, c = divmod(k, cols)
        ax = axes[r, c]
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        if k >= n:
            ax.set_visible(False); continue
        img = images[k]
        cam = cams[k]
        ax.imshow(img)
        ax.imshow(cam, cmap=SEQUENTIAL_CMAP, alpha=0.55, vmin=0.0, vmax=1.0)
        ok = labels_true[k] == labels_pred[k]
        marker = "→"  # colour already encodes right/wrong
        col = "#3F6B4E" if ok else "#A1505A"
        ax.set_title(
            f"{labels_true[k]}\n{marker} {labels_pred[k]} ({confidences[k]:.2f})",
            fontsize=7.5, color=col, pad=3,
        )
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, out_path)


def occlusion_grid(images: np.ndarray, occ_maps: np.ndarray,
                   labels_true: list[str], out_path: str | Path,
                   title: str = "Occlusion sensitivity") -> Path:
    """images: (N, H, W, 3) uint8; occ_maps: (N, H, W) float (drop in prob)."""
    n = len(images)
    cols = 6
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.9, rows * 2.05))
    axes = np.atleast_2d(axes)
    for k in range(rows * cols):
        r, c = divmod(k, cols)
        ax = axes[r, c]
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        if k >= n:
            ax.set_visible(False); continue
        ax.imshow(images[k])
        m = occ_maps[k]
        m = (m - m.min()) / max(m.max() - m.min(), 1e-9)
        ax.imshow(m, cmap=SEQUENTIAL_CMAP, alpha=0.55, vmin=0.0, vmax=1.0)
        ax.set_title(labels_true[k], fontsize=7.5, color="#222", pad=3)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, out_path)

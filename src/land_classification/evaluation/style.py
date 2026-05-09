"""Sober editorial plot styling.

Centralises rcParams + palettes so every figure shares the same look:
serif body, muted palette, no top/right spines, light grid. Imported once
by ``plots.py``; ``apply_style`` is idempotent.
"""
from __future__ import annotations
from cycler import cycler
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap


# Muted ColorBrewer-leaning palette, ordered for the 10 EuroSAT classes.
PALETTE: list[str] = [
    "#5B7C99",  # 0 AnnualCrop          steel blue
    "#3F6B4E",  # 1 Forest               muted forest green
    "#8AAE7B",  # 2 HerbaceousVegetation sage
    "#9C7A5E",  # 3 Highway              warm taupe
    "#A1505A",  # 4 Industrial           muted brick
    "#C7A65E",  # 5 Pasture              ochre
    "#6F8E55",  # 6 PermanentCrop        olive green
    "#7E5A86",  # 7 Residential          dusty plum
    "#3F7E8A",  # 8 River                slate teal
    "#2C5070",  # 9 SeaLake              deep navy
]

MODEL_COLORS: dict[str, str] = {
    "RandomForest": "#3F6B4E",
    "XGBoost":      "#A1505A",
    "ResNet18":     "#2C5070",
}

GROUP_COLORS: dict[str, str] = {
    "rgb_stats": "#5B7C99",
    "hsv_hist":  "#7E5A86",
    "glcm":      "#9C7A5E",
    "hu":        "#C7A65E",
    "edge":      "#3F7E8A",
    "vari":      "#3F6B4E",
}

SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list(
    "sober_seq",
    ["#F4EFE6", "#D8C9AE", "#A98F6B", "#6F5A3D", "#3A2E1F"],
)
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "sober_div",
    ["#3F6B4E", "#9DBFA4", "#F4EFE6", "#D69A8A", "#A1505A"],
)


_APPLIED = False


def apply_style() -> None:
    """Set sober editorial rcParams. Safe to call repeatedly."""
    global _APPLIED
    if _APPLIED:
        return
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": [
            "Source Serif Pro", "Source Serif 4", "STIX Two Text",
            "Liberation Serif", "DejaVu Serif", "Times New Roman",
        ],
        "mathtext.fontset": "stix",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#9c9c9c",
        "grid.alpha": 0.22,
        "grid.linewidth": 0.5,
        "axes.titlesize": 11,
        "axes.titleweight": "regular",
        "axes.titlepad": 8,
        "axes.labelsize": 10,
        "axes.labelcolor": "#222222",
        "xtick.color": "#444444",
        "ytick.color": "#444444",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "figure.dpi": 140,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "axes.prop_cycle": cycler("color", list(MODEL_COLORS.values())),
        "image.cmap": "gray",
    })
    _APPLIED = True

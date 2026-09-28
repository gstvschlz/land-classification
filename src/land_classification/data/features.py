"""Hand-crafted features for the traditional baselines.

Layout (must match ``explain.traditional.feature_layout``):
RGB stats (9) | HSV histograms (3 x bins) | GLCM (props x angles) | Hu (7) | edge (1) | VARI (2)
"""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import skew
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import canny, graycomatrix, graycoprops
from skimage.measure import moments_central, moments_hu, moments_normalized
from tqdm import tqdm

GLCM_PROPS = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation")
GLCM_LEVELS = 32  # gray levels after quantisation; keeps the co-occurrence matrix small
EPS = 1e-8


def feature_dim(hsv_bins: int = 16, n_angles: int = 4) -> int:
    return 9 + 3 * hsv_bins + len(GLCM_PROPS) * n_angles + 7 + 1 + 2


def extract_one(path: str, hsv_bins: int = 16, glcm_distances=(1,),
                glcm_angles_deg=(0, 45, 90, 135)) -> np.ndarray:
    with Image.open(path) as img:
        rgb = np.asarray(img.convert("RGB"), dtype=np.float64) / 255.0
    ch = rgb.reshape(-1, 3)
    feats: list = [ch.mean(0), ch.std(0), skew(ch, axis=0)]

    hsv = rgb2hsv(rgb).reshape(-1, 3)
    for c in range(3):
        hist, _ = np.histogram(hsv[:, c], bins=hsv_bins, range=(0, 1), density=True)
        feats.append(hist)

    gray = rgb2gray(rgb)
    q = np.minimum((gray * GLCM_LEVELS).astype(np.uint8), GLCM_LEVELS - 1)
    glcm = graycomatrix(q, list(glcm_distances), np.deg2rad(glcm_angles_deg),
                        levels=GLCM_LEVELS, symmetric=True, normed=True)
    for prop in GLCM_PROPS:
        feats.append(graycoprops(glcm, prop).mean(axis=0))  # average over distances

    hu = moments_hu(moments_normalized(moments_central(gray)))
    feats.append(-np.sign(hu) * np.log10(np.abs(hu) + EPS))  # log scale: raw Hu spans many orders

    feats.append([canny(gray, sigma=1.0).mean()])

    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    vari = np.clip((g - r) / (g + r - b + EPS), -1.0, 1.0)  # denominator can vanish
    feats.append([vari.mean(), vari.std()])

    out = np.nan_to_num(np.concatenate([np.ravel(f) for f in feats]).astype(np.float32))
    assert out.shape[0] == feature_dim(hsv_bins, len(glcm_angles_deg))
    return out


def _worker(args):
    return extract_one(*args)


def cache_features(paths: list[str], labels: np.ndarray, out_path: str | Path, *,
                   hsv_bins: int = 16, glcm_distances=(1,),
                   glcm_angles_deg=(0, 45, 90, 135)) -> tuple[np.ndarray, np.ndarray]:
    """Extract features for every path once and cache them as ``X``/``y`` in an .npz."""
    out_path = Path(out_path)
    dim = feature_dim(hsv_bins, len(glcm_angles_deg))
    if out_path.exists():
        d = np.load(out_path)
        if d["X"].shape == (len(paths), dim):
            return d["X"], d["y"]

    jobs = [(p, hsv_bins, tuple(glcm_distances), tuple(glcm_angles_deg)) for p in paths]
    with ProcessPoolExecutor() as ex:
        X = np.stack(list(tqdm(ex.map(_worker, jobs, chunksize=64),
                               total=len(jobs), desc="features")))
    y = np.asarray(labels, dtype=np.int64)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, X=X, y=y)
    return X, y

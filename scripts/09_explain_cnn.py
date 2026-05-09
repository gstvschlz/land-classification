"""Grad-CAM and occlusion sensitivity for ResNet-18 on fold-0 examples.

Loads the fold-0 checkpoint saved by ``04_train_resnet.py`` and the per-fold
test predictions, picks ``k_correct`` correct + ``k_wrong`` confident-wrong
samples per class, and renders two figure grids.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from land_classification.config import load_config
from land_classification.data.dataset import EuroSATIndex, load_folds
from land_classification.models.resnet import build_resnet18
from land_classification.explain.cnn import (
    GradCAM, occlusion_sensitivity, pick_examples,
)
from land_classification.evaluation.plots import gradcam_grid, occlusion_grid


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _load_image(path: str, size: int) -> tuple[np.ndarray, torch.Tensor]:
    """Return (uint8 RGB array (size, size, 3), normalised tensor (1, 3, size, size))."""
    with Image.open(path) as img:
        img = img.convert("RGB").resize((size, size))
    arr = np.asarray(img, dtype=np.uint8)
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])(Image.fromarray(arr)).unsqueeze(0)
    return arr, t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    nc = cfg["data"]["num_classes"]
    image_size = cfg["data"]["image_size_deep"]
    class_names = cfg["data"]["class_names"]
    seed = cfg["seed"]

    metrics_dir = ROOT / cfg["results"]["metrics_dir"]
    fig_dir = ROOT / cfg["results"]["figures_dir"]
    out_root = ROOT / cfg["results"]["root"] / "explanations" / "ResNet18"
    out_root.mkdir(parents=True, exist_ok=True)

    ckpt_path = ROOT / cfg["results"]["root"] / "checkpoints" / "ResNet18" / "fold0.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"Missing checkpoint {ckpt_path}; run 04_train_resnet.py first.")

    pred_path = metrics_dir / "ResNet18" / "fold0_predictions.npz"
    if not pred_path.exists():
        raise SystemExit(f"Missing predictions {pred_path}; run 04_train_resnet.py first.")

    folds = load_folds(ROOT / cfg["results"]["root"] / "folds.json")
    _, te_idx = folds[0]
    data_dir = ROOT / cfg["data"]["root"] / "EuroSAT_RGB"
    idx = EuroSATIndex(data_dir)

    # Load predictions and align them to fold-0 test indices.
    d = np.load(pred_path)
    probs = d["probs"]; y_true = d["y_true"]; test_idx = d["test_idx"]
    # ``test_idx`` is the original index ordering used during prediction;
    # we just work in that order.
    k_correct = cfg["explain"]["cnn_examples_per_class"]
    k_wrong = cfg["explain"]["cnn_wrong_per_class"]
    if args.fast:
        k_correct = 1; k_wrong = 1
    chosen_local = pick_examples(probs, y_true, n_classes=nc,
                                 k_correct=k_correct, k_wrong=k_wrong, seed=seed)
    if args.fast:
        chosen_local = chosen_local[:18]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_resnet18(num_classes=nc, pretrained=False).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["state_dict"])
    model.eval()

    images_u8: list[np.ndarray] = []
    cams_all: list[np.ndarray] = []
    occ_all: list[np.ndarray] = []
    labels_true: list[str] = []
    labels_pred: list[str] = []
    confidences: list[float] = []

    patch = cfg["explain"]["occlusion_patch"]
    stride = cfg["explain"]["occlusion_stride"]
    if args.fast:
        stride = max(stride, 32)

    print(f"Generating Grad-CAM + occlusion for {len(chosen_local)} samples on {device}…")
    with GradCAM(model) as gcam:
        for j, local_i in enumerate(chosen_local):
            sample_global = int(test_idx[local_i])
            path = idx.samples[sample_global].path
            true_lbl = int(y_true[local_i])
            pred_lbl = int(probs[local_i].argmax())
            conf = float(probs[local_i].max())

            arr_u8, t = _load_image(path, image_size)
            t = t.to(device)
            cam, _ = gcam(t, target_class=pred_lbl)
            occ = occlusion_sensitivity(
                model, t, target_class=pred_lbl,
                patch=patch, stride=stride, device=device,
            )

            images_u8.append(arr_u8)
            cams_all.append(cam.squeeze(0).numpy())
            occ_all.append(occ)
            labels_true.append(class_names[true_lbl])
            labels_pred.append(class_names[pred_lbl])
            confidences.append(conf)
            if (j + 1) % 5 == 0 or j + 1 == len(chosen_local):
                print(f"  [{j+1}/{len(chosen_local)}]")

    images_u8 = np.stack(images_u8)
    cams_arr = np.stack(cams_all)
    occ_arr = np.stack(occ_all)

    np.savez_compressed(out_root / "gradcam.npz",
                        images=images_u8, cams=cams_arr,
                        labels_true=np.array(labels_true),
                        labels_pred=np.array(labels_pred),
                        confidences=np.array(confidences))
    np.savez_compressed(out_root / "occlusion.npz",
                        images=images_u8, occ=occ_arr,
                        labels_true=np.array(labels_true))

    gradcam_grid(images_u8, cams_arr, labels_true, labels_pred, confidences,
                 fig_dir / "gradcam_grid.png",
                 title="ResNet-18 — Grad-CAM (fold 0)")
    occlusion_grid(images_u8, occ_arr, labels_true,
                   fig_dir / "occlusion_grid.png",
                   title="ResNet-18 — sensibilidade por oclusão (fold 0)")
    print("Done. Wrote", out_root)


if __name__ == "__main__":
    main()

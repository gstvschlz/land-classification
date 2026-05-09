"""Fine-tune ResNet-18 over the shared CV folds."""
from __future__ import annotations
import argparse, sys, time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from land_classification.config import load_config
from land_classification.data.dataset import EuroSATIndex, EuroSATDataset, load_folds
from land_classification.models.resnet import build_resnet18, count_parameters
from land_classification.training.trainer import train_classifier, predict_proba
from land_classification.training.cv import run_cv


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(image_size: int):
    train_t = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomChoice([
            transforms.RandomRotation((0, 0)),
            transforms.RandomRotation((90, 90)),
            transforms.RandomRotation((180, 180)),
            transforms.RandomRotation((270, 270)),
        ]),
        transforms.ColorJitter(0.1, 0.1, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_t = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_t, eval_t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    seed = cfg["seed"]
    torch.manual_seed(seed); np.random.seed(seed)
    image_size = cfg["data"]["image_size_deep"]
    nc = cfg["data"]["num_classes"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("WARNING: CUDA not available; ResNet training will be slow.")

    data_dir = ROOT / cfg["data"]["root"] / "EuroSAT_RGB"
    folds_path = ROOT / cfg["results"]["root"] / "folds.json"
    idx = EuroSATIndex(data_dir)
    folds = load_folds(folds_path)
    if args.fast:
        folds = folds[:1]

    p = cfg["models"]["resnet18"]
    epochs = 2 if args.fast else p["epochs"]
    train_t, eval_t = build_transforms(image_size)

    ckpt_dir = ROOT / cfg["results"]["root"] / "checkpoints" / "ResNet18"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def run_fold(k, tr_idx, te_idx):
        # internal train/val split for early stopping
        tr_sub, va_sub = train_test_split(
            tr_idx, test_size=0.2, stratify=idx.labels[tr_idx], random_state=seed + k,
        )
        ds_tr = EuroSATDataset(idx, tr_sub, transform=train_t)
        ds_va = EuroSATDataset(idx, va_sub, transform=eval_t)
        ds_te = EuroSATDataset(idx, te_idx, transform=eval_t)
        loader_tr = DataLoader(ds_tr, batch_size=p["batch_size"], shuffle=True,
                               num_workers=4, pin_memory=True, persistent_workers=True)
        loader_va = DataLoader(ds_va, batch_size=p["batch_size"] * 2, shuffle=False,
                               num_workers=4, pin_memory=True, persistent_workers=True)
        loader_te = DataLoader(ds_te, batch_size=p["batch_size"] * 2, shuffle=False,
                               num_workers=4, pin_memory=True, persistent_workers=True)

        model = build_resnet18(num_classes=nc, pretrained=True)
        n_params = count_parameters(model)
        t0 = time.time()
        model, hist = train_classifier(
            model, loader_tr, loader_va,
            epochs=epochs, lr=p["lr"], weight_decay=p["weight_decay"],
            warmup_epochs=p["warmup_epochs"],
            label_smoothing=p["label_smoothing"],
            device=device, amp=True, patience=5,
        )
        train_seconds = time.time() - t0
        torch.save(
            {"state_dict": model.state_dict(), "num_classes": nc, "fold": k},
            ckpt_dir / f"fold{k}.pt",
        )
        probs, y_true = predict_proba(model, loader_te, device=device, amp=True)
        return probs, y_true, asdict(hist), {
            "train_seconds": train_seconds, "n_params": int(n_params),
            "device": device,
        }

    run_cv("ResNet18", folds, run_fold, num_classes=nc,
           out_dir=ROOT / cfg["results"]["metrics_dir"])


if __name__ == "__main__":
    main()

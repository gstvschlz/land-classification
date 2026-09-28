"""EuroSAT index, torch dataset and the shared stratified folds."""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Sample:
    path: str
    label: int


class EuroSATIndex:
    """Every patch under ``data_dir/<ClassName>/*.jpg``; labels follow sorted class names."""

    def __init__(self, data_dir: str | Path):
        data_dir = Path(data_dir)
        self.class_names = sorted(p.name for p in data_dir.iterdir() if p.is_dir())
        self.samples = [
            Sample(str(p), label)
            for label, name in enumerate(self.class_names)
            for p in sorted((data_dir / name).glob("*.jpg"))
        ]
        self.paths = [s.path for s in self.samples]
        self.labels = np.array([s.label for s in self.samples], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.samples)


class EuroSATDataset(Dataset):
    """``(transform(PIL image), label)`` for the samples in ``indices``."""

    def __init__(self, index: EuroSATIndex, indices, transform=None):
        self.index = index
        self.indices = np.asarray(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        s = self.index.samples[int(self.indices[i])]
        with Image.open(s.path) as img:
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, s.label


def make_folds(labels: np.ndarray, n_folds: int, seed: int,
               save_to: str | Path | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
    """Deterministic stratified K-fold; every model shares these splits."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = [(tr, te) for tr, te in skf.split(np.zeros(len(labels)), labels)]
    if save_to is not None:
        save_to = Path(save_to)
        save_to.parent.mkdir(parents=True, exist_ok=True)
        save_to.write_text(json.dumps(
            [{"train": tr.tolist(), "test": te.tolist()} for tr, te in folds]))
    return folds


def load_folds(path: str | Path) -> list[tuple[np.ndarray, np.ndarray]]:
    raw = json.loads(Path(path).read_text())
    return [(np.array(f["train"]), np.array(f["test"])) for f in raw]

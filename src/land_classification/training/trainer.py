"""Generic PyTorch training loop with AMP, cosine LR, warmup, and early stopping."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import math
import time
import copy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class TrainHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    epoch_seconds: list[float] = field(default_factory=list)


def _build_scheduler(optimizer, epochs: int, warmup_epochs: int,
                     steps_per_epoch: int):
    total_steps = epochs * steps_per_epoch
    warmup_steps = max(1, warmup_epochs * steps_per_epoch)

    def lr_lambda(step: int):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    warmup_epochs: int = 0,
    label_smoothing: float = 0.0,
    device: str = "cuda",
    amp: bool = True,
    patience: int = 5,
    param_groups: list[dict] | None = None,
) -> tuple[nn.Module, TrainHistory]:
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if param_groups is None:
        param_groups = [{"params": [p for p in model.parameters() if p.requires_grad]}]
    optimizer = torch.optim.AdamW(
        param_groups, lr=lr, weight_decay=weight_decay,
    )
    scheduler = _build_scheduler(optimizer, epochs, warmup_epochs, len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.startswith("cuda"))
    history = TrainHistory()
    best_acc, best_state, bad = -1.0, None, 0

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running = 0.0
        n = 0
        pbar = tqdm(train_loader, desc=f"epoch {ep}/{epochs}", leave=False)
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp and device.startswith("cuda")):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += float(loss.item()) * x.size(0)
            n += x.size(0)
            pbar.set_postfix(loss=running / max(1, n))
        train_loss = running / max(1, n)

        val_loss, val_acc = _evaluate(model, val_loader, criterion, device, amp)
        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.val_acc.append(val_acc)
        history.epoch_seconds.append(time.time() - t0)

        if val_acc > best_acc + 1e-4:
            best_acc = val_acc
            best_state = copy.deepcopy({k: v.detach().cpu()
                                        for k, v in model.state_dict().items()})
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def _evaluate(model, loader, criterion, device, amp):
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp and device.startswith("cuda")):
            logits = model(x)
            loss = criterion(logits, y)
        total_loss += float(loss.item()) * x.size(0)
        correct += int((logits.argmax(1) == y).sum().item())
        n += x.size(0)
    return total_loss / max(1, n), correct / max(1, n)


@torch.no_grad()
def predict_proba(model: nn.Module, loader: DataLoader, device: str = "cuda",
                  amp: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Return (probs[N, C], labels[N]) over the loader."""
    model.eval().to(device)
    probs_all, y_all = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp and device.startswith("cuda")):
            logits = model(x)
        probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
        probs_all.append(probs)
        y_all.append(y.numpy())
    return np.concatenate(probs_all), np.concatenate(y_all)

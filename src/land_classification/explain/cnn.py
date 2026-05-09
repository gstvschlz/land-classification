"""Grad-CAM and occlusion sensitivity for the ResNet-18 baseline."""
from __future__ import annotations
from typing import Iterable
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class GradCAM:
    """Grad-CAM on the last conv block of ResNet-18 (``layer4``).

    Hooks register on construction and are released by :meth:`close`. Use as
    a context manager to ensure cleanup even on exceptions.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None):
        self.model = model.eval()
        if target_layer is None:
            target_layer = model.layer4  # ResNet conv block #4
        self.target_layer = target_layer
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._h1 = target_layer.register_forward_hook(self._fwd)
        self._h2 = target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, _module, _inp, out):
        self._activations = out.detach()

    def _bwd(self, _module, _grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    def close(self):
        self._h1.remove(); self._h2.remove()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @torch.enable_grad()
    def __call__(self, x: torch.Tensor, target_class: int | torch.Tensor | None = None
                 ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute CAM. ``x`` is (B, 3, H, W); returns (cams (B, H, W), probs).

        ``target_class`` may be ``None`` (use predicted class) or a scalar /
        tensor of shape (B,).
        """
        x = x.requires_grad_(False)
        logits = self.model(x)
        probs = F.softmax(logits, dim=1).detach()
        if target_class is None:
            target = logits.argmax(dim=1)
        elif isinstance(target_class, int):
            target = torch.full((x.size(0),), int(target_class),
                                dtype=torch.long, device=x.device)
        else:
            target = torch.as_tensor(target_class, dtype=torch.long,
                                     device=x.device)
        score = logits.gather(1, target.view(-1, 1)).sum()
        self.model.zero_grad(set_to_none=True)
        score.backward()

        # Global-average pool gradients to obtain channel weights
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        cam = (weights * self._activations).sum(dim=1)             # (B, h, w)
        cam = F.relu(cam)
        cam = F.interpolate(cam.unsqueeze(1), size=x.shape[-2:],
                            mode="bilinear", align_corners=False).squeeze(1)
        # Normalise per-image to [0, 1]
        flat = cam.view(cam.size(0), -1)
        mn = flat.min(dim=1, keepdim=True).values
        mx = flat.max(dim=1, keepdim=True).values
        cam = (flat - mn) / (mx - mn + 1e-8)
        cam = cam.view(x.size(0), x.size(2), x.size(3))
        return cam.detach().cpu(), probs.cpu()


@torch.no_grad()
def occlusion_sensitivity(model: nn.Module, x: torch.Tensor,
                          target_class: int, *, patch: int = 32,
                          stride: int = 16, fill: float = 0.0,
                          device: str | torch.device = "cpu") -> np.ndarray:
    """Slide a square patch across ``x`` (1, 3, H, W) and record the drop in
    the target-class probability. Returns (H, W) float array (drop ≥ 0)."""
    model = model.eval().to(device)
    x = x.to(device)
    base = F.softmax(model(x), dim=1)[0, target_class].item()
    _, _, H, W = x.shape
    out = np.zeros((H, W), dtype=np.float64)
    counts = np.zeros((H, W), dtype=np.int32)
    for top in range(0, H - patch + 1, stride):
        # Build a batch over horizontal positions for speed
        rows = []
        for left in range(0, W - patch + 1, stride):
            x_occ = x.clone()
            x_occ[..., top:top + patch, left:left + patch] = fill
            rows.append(x_occ)
        if not rows:
            continue
        batch = torch.cat(rows, dim=0)
        probs = F.softmax(model(batch), dim=1)[:, target_class].cpu().numpy()
        for i, left in enumerate(range(0, W - patch + 1, stride)):
            drop = base - probs[i]
            out[top:top + patch, left:left + patch] += drop
            counts[top:top + patch, left:left + patch] += 1
    counts = np.clip(counts, 1, None)
    return out / counts


def pick_examples(probs: np.ndarray, y_true: np.ndarray, n_classes: int,
                  k_correct: int = 2, k_wrong: int = 1,
                  seed: int = 0) -> list[int]:
    """For each class, return indices of high-confidence correct predictions
    plus a confident-wrong (most over-confident misclassification)."""
    rng = np.random.default_rng(seed)
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    out: list[int] = []
    for c in range(n_classes):
        # Correct, sorted by confidence desc
        ok = np.flatnonzero((pred == c) & (y_true == c))
        if len(ok) > 0:
            order = ok[np.argsort(-conf[ok])]
            out.extend(order[:k_correct].tolist())
        # Wrong: predicted as ``c`` but truth ≠ c — confident error
        wrong = np.flatnonzero((pred == c) & (y_true != c))
        if len(wrong) > 0:
            order = wrong[np.argsort(-conf[wrong])]
            out.extend(order[:k_wrong].tolist())
    # Deduplicate while preserving order
    seen: set[int] = set(); res: list[int] = []
    for i in out:
        if i not in seen:
            seen.add(i); res.append(int(i))
    return res

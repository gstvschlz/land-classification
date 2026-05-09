"""ResNet-18 builder for EuroSAT (10 classes), ImageNet-pretrained."""
from __future__ import annotations
import torch
from torch import nn
from torchvision import models


def build_resnet18(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    net = models.resnet18(weights=weights)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

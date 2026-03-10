"""Sanity-check demo for `neural-net/` models.

Creates synthetic data and runs a few epochs for regression and classification.

Run from repo root:
    python neural-net/demo.py
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from models import (
    build_large_classifier,
    build_large_regressor,
    build_small_classifier,
    build_small_regressor,
)
from trainer import TrainConfig, train_classification, train_regression


def _make_regression(n: int = 2048, d: int = 32):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(n, d, generator=g)
    w = torch.randn(d, 1, generator=g)
    y = x @ w + 0.1 * torch.randn(n, 1, generator=g)
    return x, y


def _make_classification(n: int = 2048, d: int = 32, k: int = 3):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(n, d, generator=g)
    W = torch.randn(d, k, generator=g)
    logits = x @ W + 0.25 * torch.randn(n, k, generator=g)
    y = logits.argmax(dim=-1)
    return x, y


def main() -> None:
    # --------------------- Regression ---------------------
    x, y = _make_regression()
    ds = TensorDataset(x, y)
    train_loader = DataLoader(ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(ds, batch_size=256)

    print("\n[Regression] small")
    model = build_small_regressor(input_dim=x.shape[1], num_targets=y.shape[1])
    train_regression(model, train_loader, val_loader, TrainConfig(epochs=2, lr=1e-3, log_every=20))

    print("\n[Regression] large")
    model = build_large_regressor(input_dim=x.shape[1], num_targets=y.shape[1])
    train_regression(model, train_loader, val_loader, TrainConfig(epochs=2, lr=5e-4, log_every=20))

    # ------------------- Classification -------------------
    x, y = _make_classification()
    ds = TensorDataset(x, y)
    train_loader = DataLoader(ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(ds, batch_size=256)
    num_classes = int(y.max().item() + 1)

    print("\n[Classification] small")
    model = build_small_classifier(input_dim=x.shape[1], num_classes=num_classes)
    train_classification(model, train_loader, val_loader, TrainConfig(epochs=2, lr=1e-3, log_every=20))

    print("\n[Classification] large")
    model = build_large_classifier(input_dim=x.shape[1], num_classes=num_classes)
    train_classification(model, train_loader, val_loader, TrainConfig(epochs=2, lr=5e-4, log_every=20))


if __name__ == "__main__":
    main()

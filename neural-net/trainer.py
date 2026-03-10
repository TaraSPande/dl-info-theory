"""Tiny training utilities for the MLP models.

These are intentionally lightweight and dataset-agnostic: you provide your own
PyTorch DataLoaders that yield either:

* regression: (x, y) with y float tensor shaped (batch, num_targets)
* classification: (x, y) with y long tensor shaped (batch,)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_


@dataclass
class TrainConfig:
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    device: Optional[str] = None  # e.g. "cuda" or "cpu"; None => auto
    log_every: int = 50


def _get_device(device: Optional[str]) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate_regression(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    mse = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        mse += torch.sum((pred - y) ** 2).item()
        n += y.numel()
    model.train()
    return mse / max(1, n)


@torch.no_grad()
def evaluate_classification(model: nn.Module, loader, device: torch.device) -> Dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    ce = nn.CrossEntropyLoss()
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss_sum += float(ce(logits, y))
        pred = logits.argmax(dim=-1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    model.train()
    return {
        "loss": loss_sum / max(1, len(loader)),
        "acc": correct / max(1, total),
    }


def train_regression(model: nn.Module, train_loader, val_loader, cfg: TrainConfig) -> None:
    device = _get_device(cfg.device)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    global_step = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            opt.zero_grad(set_to_none=True)
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            opt.step()

            global_step += 1
            if cfg.log_every > 0 and global_step % cfg.log_every == 0:
                print(f"step={global_step} loss={loss.item():.6f}")

        val_mse = evaluate_regression(model, val_loader, device)
        print(f"epoch={epoch} val_mse={val_mse:.6f}")


def train_classification(model: nn.Module, train_loader, val_loader, cfg: TrainConfig) -> None:
    device = _get_device(cfg.device)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    global_step = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            opt.step()

            global_step += 1
            if cfg.log_every > 0 and global_step % cfg.log_every == 0:
                print(f"step={global_step} loss={loss.item():.6f}")

        metrics = evaluate_classification(model, val_loader, device)
        print(f"epoch={epoch} val_loss={metrics['loss']:.6f} val_acc={metrics['acc']:.4f}")

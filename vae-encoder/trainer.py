"""Training utilities for the tabular VAE + downstream heads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from models import (
    ClassificationHead,
    EncoderWithHead,
    RegressionHead,
    TabularVAE,
    TabularVAEEncoder,
    vae_loss,
)


@dataclass
class TrainConfig:
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    device: Optional[str] = None
    log_every: int = 50


def _get_device(device: Optional[str]) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_vae(
    vae: TabularVAE,
    train_loader,
    val_loader,
    *,
    cfg: TrainConfig,
    beta: float = 1.0,
    recon: Literal["mse", "l1"] = "mse",
) -> None:
    device = _get_device(cfg.device)
    vae.to(device)
    opt = torch.optim.AdamW(vae.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    global_step = 0
    for epoch in range(1, cfg.epochs + 1):
        vae.train()
        for x in train_loader:
            # allow loader to yield x or (x, y)
            if isinstance(x, (tuple, list)):
                x = x[0]
            x = x.to(device)

            opt.zero_grad(set_to_none=True)
            out = vae(x)
            losses = vae_loss(x, out["x_hat"], out["mu"], out["logvar"], beta=beta, recon=recon)
            losses["loss"].backward()
            clip_grad_norm_(vae.parameters(), cfg.max_grad_norm)
            opt.step()

            global_step += 1
            if cfg.log_every > 0 and global_step % cfg.log_every == 0:
                print(
                    f"step={global_step} loss={losses['loss'].item():.6f} "
                    f"recon={losses['recon_loss'].item():.6f} kl={losses['kl_loss'].item():.6f}"
                )

        metrics = evaluate_vae(vae, val_loader, device=device, beta=beta, recon=recon)
        print(
            f"epoch={epoch} val_loss={metrics['loss']:.6f} val_recon={metrics['recon_loss']:.6f} val_kl={metrics['kl_loss']:.6f}"
        )


@torch.no_grad()
def evaluate_vae(
    vae: TabularVAE,
    loader,
    *,
    device: torch.device,
    beta: float = 1.0,
    recon: Literal["mse", "l1"] = "mse",
) -> Dict[str, float]:
    vae.eval()
    loss_sum = 0.0
    recon_sum = 0.0
    kl_sum = 0.0
    n = 0
    for x in loader:
        if isinstance(x, (tuple, list)):
            x = x[0]
        x = x.to(device)
        out = vae(x)
        losses = vae_loss(x, out["x_hat"], out["mu"], out["logvar"], beta=beta, recon=recon)
        b = x.shape[0]
        loss_sum += float(losses["loss"]) * b
        recon_sum += float(losses["recon_loss"]) * b
        kl_sum += float(losses["kl_loss"]) * b
        n += b
    vae.train()
    return {
        "loss": loss_sum / max(1, n),
        "recon_loss": recon_sum / max(1, n),
        "kl_loss": kl_sum / max(1, n),
    }


def train_regression_head(
    encoder: TabularVAEEncoder,
    *,
    num_targets: int,
    train_loader,
    val_loader,
    cfg: TrainConfig,
    head_hidden: Optional[list[int]] = None,
    freeze_encoder: bool = True,
) -> EncoderWithHead:
    device = _get_device(cfg.device)
    encoder.to(device)
    if freeze_encoder:
        for p in encoder.parameters():
            p.requires_grad = False

    head = RegressionHead(latent_dim=encoder.cfg.latent_dim, num_targets=num_targets, hidden=head_hidden)
    model = EncoderWithHead(encoder, head, use_mean=True).to(device)

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    global_step = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(x)["out"]
            loss = loss_fn(out, y)
            loss.backward()
            clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            opt.step()
            global_step += 1
            if cfg.log_every > 0 and global_step % cfg.log_every == 0:
                print(f"step={global_step} loss={loss.item():.6f}")
        metrics = evaluate_regression(model, val_loader, device)
        print(f"epoch={epoch} val_mse={metrics:.6f}")

    return model


@torch.no_grad()
def evaluate_regression(model: EncoderWithHead, loader, device: torch.device) -> float:
    model.eval()
    mse = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)["out"]
        mse += torch.sum((pred - y) ** 2).item()
        n += y.numel()
    model.train()
    return mse / max(1, n)


def train_classification_head(
    encoder: TabularVAEEncoder,
    *,
    num_classes: int,
    train_loader,
    val_loader,
    cfg: TrainConfig,
    head_hidden: Optional[list[int]] = None,
    freeze_encoder: bool = True,
) -> EncoderWithHead:
    device = _get_device(cfg.device)
    encoder.to(device)
    if freeze_encoder:
        for p in encoder.parameters():
            p.requires_grad = False

    head = ClassificationHead(latent_dim=encoder.cfg.latent_dim, num_classes=num_classes, hidden=head_hidden)
    model = EncoderWithHead(encoder, head, use_mean=True).to(device)

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    global_step = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)["out"]
            loss = loss_fn(logits, y)
            loss.backward()
            clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            opt.step()
            global_step += 1
            if cfg.log_every > 0 and global_step % cfg.log_every == 0:
                print(f"step={global_step} loss={loss.item():.6f}")
        metrics = evaluate_classification(model, val_loader, device)
        print(f"epoch={epoch} val_loss={metrics['loss']:.6f} val_acc={metrics['acc']:.4f}")

    return model


@torch.no_grad()
def evaluate_classification(model: EncoderWithHead, loader, device: torch.device) -> Dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    ce = nn.CrossEntropyLoss()
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)["out"]
        loss_sum += float(ce(logits, y))
        pred = logits.argmax(dim=-1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    model.train()
    return {
        "loss": loss_sum / max(1, len(loader)),
        "acc": correct / max(1, total),
    }

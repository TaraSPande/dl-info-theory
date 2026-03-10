"""End-to-end sanity check demo for the VAE encoder workflow.

1) Train VAE on synthetic x.
2) Discard decoder; keep encoder.
3) Attach a regression head and train it.
4) Attach a classification head and train it.

Run from repo root:
    python vae-encoder/demo.py
"""

from __future__ import annotations

import copy

import torch
from torch.utils.data import DataLoader, TensorDataset

from models import TabularVAE, VAEConfig
from trainer import TrainConfig, train_classification_head, train_regression_head, train_vae


def make_unlabeled(n: int = 4096, d: int = 32) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    means = torch.tensor([[2.0] * d, [-2.0] * d])
    comps = torch.randint(0, 2, (n,), generator=g)
    x = means[comps] + 0.5 * torch.randn(n, d, generator=g)
    return x


def make_regression_labels(x: torch.Tensor) -> torch.Tensor:
    g = torch.Generator().manual_seed(1)
    y = x[:, :5].sum(dim=-1, keepdim=True) + 0.1 * torch.randn(x.shape[0], 1, generator=g)
    return y


def make_classification_labels(x: torch.Tensor, k: int = 3) -> torch.Tensor:
    g = torch.Generator().manual_seed(2)
    W = torch.randn(x.shape[1], k, generator=g)
    logits = x @ W
    return logits.argmax(dim=-1)


def main() -> None:
    x = make_unlabeled()
    ds_unlab = TensorDataset(x)
    train_loader = DataLoader(ds_unlab, batch_size=128, shuffle=True)
    val_loader = DataLoader(ds_unlab, batch_size=512)

    cfg = VAEConfig(
        input_dim=x.shape[1],
        latent_dim=16,
        enc_hidden=[256, 128],
        dec_hidden=[128, 256],
        dropout=0.0,
    )
    vae = TabularVAE(cfg)

    print("\n[1] Training VAE")
    train_vae(vae, train_loader, val_loader, cfg=TrainConfig(epochs=3, lr=1e-3, log_every=50), beta=1.0)

    # Keep only the encoder
    encoder = copy.deepcopy(vae.encoder).cpu()

    print("\n[2] Downstream regression head (encoder frozen)")
    y_reg = make_regression_labels(x)
    ds_reg = TensorDataset(x, y_reg)
    reg_train = DataLoader(ds_reg, batch_size=128, shuffle=True)
    reg_val = DataLoader(ds_reg, batch_size=512)
    train_regression_head(
        encoder,
        num_targets=1,
        train_loader=reg_train,
        val_loader=reg_val,
        cfg=TrainConfig(epochs=3, lr=1e-3, log_every=50),
        freeze_encoder=True,
    )

    print("\n[3] Downstream classification head (encoder frozen)")
    y_cls = make_classification_labels(x, k=3)
    ds_cls = TensorDataset(x, y_cls)
    cls_train = DataLoader(ds_cls, batch_size=128, shuffle=True)
    cls_val = DataLoader(ds_cls, batch_size=512)
    train_classification_head(
        encoder,
        num_classes=3,
        train_loader=cls_train,
        val_loader=cls_val,
        cfg=TrainConfig(epochs=3, lr=1e-3, log_every=50),
        freeze_encoder=True,
    )


if __name__ == "__main__":
    main()

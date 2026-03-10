"""VAE encoder/decoder modules for float tabular data.

Design goal:
- Train a full VAE (encoder+decoder) with reconstruction+KL loss.
- After training, discard the decoder and reuse the encoder as a feature extractor.
- Attach a downstream head for regression or classification.

This folder name contains a hyphen (`vae-encoder/`), so it is not intended to be
imported as a normal Python package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


ActivationName = Literal["relu", "gelu", "tanh", "silu"]


def _act(name: ActivationName) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unknown activation: {name}")


def _mlp(
    *,
    in_dim: int,
    hidden_dims: List[int],
    out_dim: int,
    activation: ActivationName = "relu",
    dropout: float = 0.0,
    use_layernorm: bool = False,
) -> nn.Sequential:
    layers: List[nn.Module] = []
    d = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(d, h))
        if use_layernorm:
            layers.append(nn.LayerNorm(h))
        layers.append(_act(activation))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


@dataclass(frozen=True)
class VAEConfig:
    input_dim: int
    latent_dim: int
    enc_hidden: List[int] = (256, 128)  # type: ignore[assignment]
    dec_hidden: List[int] = (128, 256)  # type: ignore[assignment]
    activation: ActivationName = "relu"
    dropout: float = 0.0
    use_layernorm: bool = True


class TabularVAEEncoder(nn.Module):
    """Encoder that maps x -> (mu, logvar) and can sample z via reparameterization."""

    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg
        # shared trunk
        trunk_out = cfg.enc_hidden[-1] if len(cfg.enc_hidden) > 0 else cfg.input_dim
        self.trunk = _mlp(
            in_dim=cfg.input_dim,
            hidden_dims=list(cfg.enc_hidden[:-1]) if len(cfg.enc_hidden) > 1 else [],
            out_dim=trunk_out,
            activation=cfg.activation,
            dropout=cfg.dropout,
            use_layernorm=cfg.use_layernorm,
        )
        self.mu = nn.Linear(trunk_out, cfg.latent_dim)
        self.logvar = nn.Linear(trunk_out, cfg.latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 2:
            raise ValueError(f"Expected x with shape (batch, input_dim) but got {tuple(x.shape)}")
        h = self.trunk(x)
        return self.mu(h), self.logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # z = mu + sigma * eps; sigma = exp(0.5*logvar)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class TabularVAEDecoder(nn.Module):
    """Decoder that maps z -> reconstruction x_hat."""

    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg
        self.net = _mlp(
            in_dim=cfg.latent_dim,
            hidden_dims=list(cfg.dec_hidden),
            out_dim=cfg.input_dim,
            activation=cfg.activation,
            dropout=cfg.dropout,
            use_layernorm=cfg.use_layernorm,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2:
            raise ValueError(f"Expected z with shape (batch, latent_dim) but got {tuple(z.shape)}")
        return self.net(z)


class TabularVAE(nn.Module):
    """Full VAE: encoder + decoder."""

    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = TabularVAEEncoder(cfg)
        self.decoder = TabularVAEDecoder(cfg)

    def forward(self, x: torch.Tensor) -> dict:
        mu, logvar = self.encoder(x)
        z = self.encoder.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return {"x_hat": x_hat, "mu": mu, "logvar": logvar, "z": z}


def vae_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    *,
    beta: float = 1.0,
    recon: Literal["mse", "l1"] = "mse",
) -> dict:
    """Compute beta-VAE loss terms.

    Returns dict with keys: loss, recon_loss, kl_loss
    """
    if recon == "mse":
        recon_loss = F.mse_loss(x_hat, x, reduction="mean")
    elif recon == "l1":
        recon_loss = F.l1_loss(x_hat, x, reduction="mean")
    else:
        raise ValueError(f"Unknown recon loss: {recon}")

    # KL(q(z|x) || N(0,I)) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl = -0.5 * torch.mean(torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1))
    loss = recon_loss + beta * kl
    return {"loss": loss, "recon_loss": recon_loss, "kl_loss": kl}


class DownstreamHead(nn.Module):
    """Base class marker for downstream heads."""


class RegressionHead(DownstreamHead):
    def __init__(self, latent_dim: int, num_targets: int = 1, hidden: Optional[List[int]] = None):
        super().__init__()
        hidden = hidden or []
        self.net = _mlp(in_dim=latent_dim, hidden_dims=hidden, out_dim=num_targets, activation="relu")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class ClassificationHead(DownstreamHead):
    def __init__(self, latent_dim: int, num_classes: int, hidden: Optional[List[int]] = None):
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        hidden = hidden or []
        self.net = _mlp(in_dim=latent_dim, hidden_dims=hidden, out_dim=num_classes, activation="relu")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class EncoderWithHead(nn.Module):
    """Convenience module: encoder -> z (sample or mean) -> head."""

    def __init__(
        self,
        encoder: TabularVAEEncoder,
        head: DownstreamHead,
        *,
        use_mean: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.use_mean = use_mean

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = mu if self.use_mean else self.encoder.reparameterize(mu, logvar)
        return z, mu, logvar

    def forward(self, x: torch.Tensor) -> dict:
        z, mu, logvar = self.encode(x)
        out = self.head(z)
        return {"out": out, "z": z, "mu": mu, "logvar": logvar}

"""Simple feed-forward neural nets for tabular/vector inputs.

This folder name contains a hyphen (`neural-net/`), so it is not intended to be
imported as a Python package name. Run scripts from within this repo like:

    python neural-net/demo.py

or import by adding this directory to PYTHONPATH.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal

import torch
import torch.nn as nn


ActivationName = Literal["relu", "gelu", "tanh", "silu"]


def _make_activation(name: ActivationName) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unknown activation: {name}")


@dataclass(frozen=True)
class MLPConfig:
    input_dim: int
    hidden_dims: List[int]
    dropout: float = 0.0
    activation: ActivationName = "relu"
    use_layernorm: bool = False


class MLPBackbone(nn.Module):
    """A generic MLP feature extractor.

    Accepts inputs of shape (batch, input_dim) and outputs (batch, hidden_dims[-1]).
    """

    def __init__(self, cfg: MLPConfig):
        super().__init__()
        self.cfg = cfg

        layers: List[nn.Module] = []
        in_dim = cfg.input_dim
        for i, h in enumerate(cfg.hidden_dims):
            layers.append(nn.Linear(in_dim, h))
            if cfg.use_layernorm:
                layers.append(nn.LayerNorm(h))
            layers.append(_make_activation(cfg.activation))
            if cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
            in_dim = h

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected x with shape (batch, features) but got {tuple(x.shape)}")
        return self.net(x)


class MLPRegressor(nn.Module):
    """MLP for regression.

    Output: (batch, num_targets)
    """

    def __init__(self, backbone: MLPBackbone, num_targets: int = 1):
        super().__init__()
        self.backbone = backbone
        out_dim = backbone.cfg.hidden_dims[-1] if backbone.cfg.hidden_dims else backbone.cfg.input_dim
        self.head = nn.Linear(out_dim, num_targets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        return self.head(h)


class MLPClassifier(nn.Module):
    """MLP for classification.

    Output logits: (batch, num_classes)
    """

    def __init__(self, backbone: MLPBackbone, num_classes: int):
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        self.backbone = backbone
        out_dim = backbone.cfg.hidden_dims[-1] if backbone.cfg.hidden_dims else backbone.cfg.input_dim
        self.head = nn.Linear(out_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        return self.head(h)


def build_small_regressor(
    *,
    input_dim: int,
    num_targets: int = 1,
    dropout: float = 0.0,
    activation: ActivationName = "relu",
    use_layernorm: bool = False,
) -> MLPRegressor:
    cfg = MLPConfig(
        input_dim=input_dim,
        hidden_dims=[64, 64],
        dropout=dropout,
        activation=activation,
        use_layernorm=use_layernorm,
    )
    return MLPRegressor(MLPBackbone(cfg), num_targets=num_targets)


def build_large_regressor(
    *,
    input_dim: int,
    num_targets: int = 1,
    dropout: float = 0.1,
    activation: ActivationName = "gelu",
    use_layernorm: bool = True,
) -> MLPRegressor:
    cfg = MLPConfig(
        input_dim=input_dim,
        hidden_dims=[512, 256, 256, 128],
        dropout=dropout,
        activation=activation,
        use_layernorm=use_layernorm,
    )
    return MLPRegressor(MLPBackbone(cfg), num_targets=num_targets)


def build_small_classifier(
    *,
    input_dim: int,
    num_classes: int,
    dropout: float = 0.0,
    activation: ActivationName = "relu",
    use_layernorm: bool = False,
) -> MLPClassifier:
    cfg = MLPConfig(
        input_dim=input_dim,
        hidden_dims=[64, 64],
        dropout=dropout,
        activation=activation,
        use_layernorm=use_layernorm,
    )
    return MLPClassifier(MLPBackbone(cfg), num_classes=num_classes)


def build_large_classifier(
    *,
    input_dim: int,
    num_classes: int,
    dropout: float = 0.1,
    activation: ActivationName = "gelu",
    use_layernorm: bool = True,
) -> MLPClassifier:
    cfg = MLPConfig(
        input_dim=input_dim,
        hidden_dims=[512, 256, 256, 128],
        dropout=dropout,
        activation=activation,
        use_layernorm=use_layernorm,
    )
    return MLPClassifier(MLPBackbone(cfg), num_classes=num_classes)

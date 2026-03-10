from dataclasses import dataclass
from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer.layers import LearnedPositionalEmbedding, Encoder, Decoder
from transformer.config import TransformerConfig

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------

class EncoderClassifier(nn.Module):
    """Encoder-only classifier (GLUE/SuperGLUE) using mean pooling."""
    def __init__(self, cfg: TransformerConfig, num_labels: int, pad_token_id: int = 0):
        super().__init__()
        self.cfg = cfg
        self.pad_token_id = pad_token_id

        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=pad_token_id)
        self.pos = LearnedPositionalEmbedding(cfg.max_src_len, cfg.d_model)
        # Store max position for clamping
        self._max_pos = cfg.max_src_len
        # Build encoder-only stack: reuse Encoder
        self.encoder = Encoder(cfg)
        self.ln_f = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
        self.drop = nn.Dropout(cfg.dropout)
        self.classifier = nn.Linear(cfg.d_model, num_labels)

    def _positions(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        pos = torch.arange(0, T, device=x.device).unsqueeze(0).expand(B, T)
        # Clamp positions to prevent out-of-bounds access to positional embeddings
        max_pos = getattr(self, '_max_pos', 256)  # Default fallback
        pos = torch.clamp(pos, max=max_pos - 1)
        return pos

    def _kpm(self, input_ids: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if attn_mask is not None:
            return attn_mask
        return (input_ids != self.pad_token_id).long()

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None):
        pos = self._positions(input_ids)
        x = self.tok(input_ids) + self.pos(pos)
        x = self.drop(x)
        kpm = self._kpm(input_ids, attention_mask)
        x = self.encoder(x, kpm)
        x = self.ln_f(x)
        # mean-pool over non-pad tokens
        mask = kpm.unsqueeze(-1).float()
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        logits = self.classifier(pooled)
        out = {"logits": logits}
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            out["loss"] = loss
        return out


class EncoderRegressor(nn.Module):
    """Encoder-only regressor using mean pooling over token embeddings.

    Predicts one or more continuous targets with an MSE loss.
    """
    def __init__(self, cfg: TransformerConfig, num_targets: int, pad_token_id: int = 0):
        super().__init__()
        self.cfg = cfg
        self.pad_token_id = pad_token_id
        self.num_targets = int(num_targets)

        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=pad_token_id)
        self.pos = LearnedPositionalEmbedding(cfg.max_src_len, cfg.d_model)
        # Store max position for clamping
        self._max_pos = cfg.max_src_len
        # Encoder stack
        self.encoder = Encoder(cfg)
        self.ln_f = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
        self.drop = nn.Dropout(cfg.dropout)
        self.regressor = nn.Linear(cfg.d_model, num_targets)

    def _positions(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        pos = torch.arange(0, T, device=x.device).unsqueeze(0).expand(B, T)
        # Clamp positions to prevent out-of-bounds access to positional embeddings
        max_pos = getattr(self, '_max_pos', 256)  # Default fallback
        pos = torch.clamp(pos, max=max_pos - 1)
        return pos

    def _kpm(self, input_ids: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if attn_mask is not None:
            return attn_mask
        return (input_ids != self.pad_token_id).long()

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None):
        pos = self._positions(input_ids)
        x = self.tok(input_ids) + self.pos(pos)
        x = self.drop(x)
        kpm = self._kpm(input_ids, attention_mask)
        x = self.encoder(x, kpm)
        x = self.ln_f(x)
        # mean-pool over non-pad tokens
        mask = kpm.unsqueeze(-1).float()
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        preds = self.regressor(pooled)
        out = {"logits": preds}
        if labels is not None:
            # Masked MSE that ignores NaNs in labels
            labels_f = labels.to(preds.dtype)
            mask = torch.isfinite(labels_f)
            labels_zn = torch.nan_to_num(labels_f, nan=0.0, posinf=0.0, neginf=0.0)
            se = (preds - labels_zn) ** 2
            denom = mask.sum().clamp(min=1)
            loss = (se * mask.float()).sum() / denom
            out["loss"] = loss
        return out

# -----------------------------------------------------------------------------
# Factory helpers (optional)
# -----------------------------------------------------------------------------

def build_classifier(vocab_size: int, pad_token_id: int, num_labels: int, **kwargs) -> EncoderClassifier:
    cfg = TransformerConfig(vocab_size=vocab_size, **kwargs)
    return EncoderClassifier(cfg, num_labels=num_labels, pad_token_id=pad_token_id)

def build_regressor(vocab_size: int, pad_token_id: int, num_targets: int, **kwargs) -> EncoderRegressor:
    cfg = TransformerConfig(vocab_size=vocab_size, **kwargs)
    return EncoderRegressor(cfg, num_targets=num_targets, pad_token_id=pad_token_id)

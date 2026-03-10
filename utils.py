"""Small shared helpers used across modules.

Historically, some utilities lived under transformer-encoder/utils.py.
This root-level module exists so `from utils import ...` works when running
from the repo root.
"""

from __future__ import annotations

from transformer_encoder_shim import ensure_special_tokens, set_seed  # noqa: F401

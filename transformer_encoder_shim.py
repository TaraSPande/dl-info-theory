"""Shim module to expose transformer-encoder utilities at repo root.

The directory `transformer-encoder/` is not importable as a Python package name
because it contains a hyphen. Some code expects `from utils import ...`.

We keep the canonical implementations in `transformer-encoder/utils.py` and
re-export them here so other modules can import without manipulating sys.path.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


_ROOT = Path(__file__).resolve().parent
_TX_UTILS = _ROOT / "transformer-encoder" / "utils.py"

spec = importlib.util.spec_from_file_location("_tx_utils", str(_TX_UTILS))
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load transformer utils from {_TX_UTILS}")
_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_mod)  # type: ignore[call-arg]


set_seed = getattr(_mod, "set_seed")
ensure_special_tokens = getattr(_mod, "ensure_special_tokens")

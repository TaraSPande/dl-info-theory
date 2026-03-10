"""Evaluation dispatcher for saved runs.

The unified training pipeline writes:

  runs/<run_slug_timestamp>/
    run.json                 # unified metadata
    train.log
    final/
      pytorch_model.bin
      run_config.json
      (tokenizer files if transformer)

Transformer runs save checkpoints with the existing transformer trainer, which
also uses the `final/` folder name, so evaluation always targets `final/`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import torch
import os

from preprocess import DataConfig, TaskType, build_dataset, build_tabular_loaders
from runlib import ensure_transformer_on_path


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _resolve_checkpoint_dir(run_dir: Path) -> Path:
    ckpt = run_dir / "final"
    if not ckpt.exists():
        raise FileNotFoundError(f"Expected checkpoint dir at: {ckpt}")
    if not (ckpt / "pytorch_model.bin").exists():
        raise FileNotFoundError(f"Missing pytorch_model.bin in: {ckpt}")
    if not (ckpt / "run_config.json").exists():
        raise FileNotFoundError(f"Missing run_config.json in: {ckpt}")
    return ckpt


def _parse_split_slice(split: str) -> tuple[str, int | None]:
    """Parse HF-style split slice syntax like 'test[:2000]'.

    Returns: (base_split, limit)
    """
    s = split.strip()
    if ":" not in s:
        return s, None
    # Expect 'name[:N]'
    if not s.endswith("]") or "[:" not in s:
        raise ValueError(f"Unsupported split slice syntax: {split}")
    name, rest = s.split("[:", 1)
    n_str = rest[:-1]
    n = int(n_str)
    if n <= 0:
        raise ValueError("Slice size must be > 0")
    return name, n


def _load_text_dataset(data_cfg0: Dict[str, Any], split: str) -> Any:
    """Load an evaluation dataset for Transformer runs.

    Supports both HF datasets and CSV datasets (including our CSV->text adapter).
    """
    cfg = DataConfig(**data_cfg0)
    ds = build_dataset(cfg)
    split_name, limit = _parse_split_slice(split)

    # map common names to actual configured split keys
    if split_name == "train":
        key = cfg.split_train
    elif split_name in ("validation", "val"):
        key = cfg.split_val
    elif split_name == "test":
        key = cfg.split_test or "test"
    else:
        key = split_name

    if key not in ds:
        # fallback to validation if test missing
        if key == (cfg.split_test or "test") and cfg.split_val in ds:
            key = cfg.split_val
        else:
            raise ValueError(f"Split '{key}' not found. Available: {list(ds.keys())}")

    out = ds[key]
    if limit is not None and len(out) > limit:
        out = out.select(range(limit))
    return out


# ---------------------------------------------------------------------------
# Transformer evaluation (text)
# ---------------------------------------------------------------------------


def _detect_transformer_config(state_dict: Dict[str, torch.Tensor], run_cfg: Dict[str, Any], data_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Infer TransformerConfig args from weights with fallbacks."""
    import re

    # d_model / max_len
    d_model = None
    max_src_len = None
    for k, v in state_dict.items():
        if k.endswith("pos.weight.weight") or k.endswith("src_pos.weight.weight"):
            max_src_len, d_model = int(v.shape[0]), int(v.shape[1])
            break

    if d_model is None:
        for key in ("tok.weight", "src_tok.weight", "tgt_tok.weight"):
            if key in state_dict:
                d_model = int(state_dict[key].shape[1])
                break
    if d_model is None:
        d_model = int(run_cfg.get("d_model", 512))

    if max_src_len is None:
        max_src_len = int(data_cfg.get("max_length", run_cfg.get("max_len", 512)))

    # vocab size
    vocab_size = None
    for key in ("tok.weight", "src_tok.weight", "tgt_tok.weight"):
        if key in state_dict:
            vocab_size = int(state_dict[key].shape[0])
            break

    # d_ff
    d_ff = 4 * int(d_model)
    for k, v in state_dict.items():
        if k.endswith("encoder.layers.0.ff.net.0.weight"):
            d_ff = int(v.shape[0])
            break

    # layers
    max_layer_enc = -1
    for k in state_dict.keys():
        m = re.match(r"encoder\.layers\.(\d+)\.", k)
        if m:
            max_layer_enc = max(max_layer_enc, int(m.group(1)))
    n_layers_enc = (max_layer_enc + 1) if max_layer_enc >= 0 else int(run_cfg.get("layers_enc", 6))

    # heads
    n_heads_det = None
    for key in ("encoder.layers.0.self_attn.gate", "encoder.layers.0.self_attn.jto_gate"):
        if key in state_dict:
            try:
                n_heads_det = int(state_dict[key].numel())
                break
            except Exception:
                pass
    if n_heads_det is None and "encoder.layers.0.self_attn.rand_logits" in state_dict:
        try:
            n_heads_det = int(state_dict["encoder.layers.0.self_attn.rand_logits"].shape[0])
        except Exception:
            n_heads_det = None
    n_heads = int(run_cfg.get("heads", n_heads_det if n_heads_det is not None else 8))

    def _has(prefix: str) -> bool:
        base = "encoder.layers.0.self_attn"
        return any(k.startswith(f"{base}.{prefix}") for k in state_dict.keys())

    if _has("jto_gate"):
        attn_self_enc = "jto"
    elif _has("rand_logits"):
        attn_self_enc = "hybrid_random" if _has("gate") else "synth_random"
    elif _has("synth"):
        attn_self_enc = "hybrid_dense" if _has("gate") else "synth_dense"
    else:
        attn_self_enc = "vanilla"

    synth_hidden = int(run_cfg.get("synth_hidden", 0))
    for key in (
        "encoder.layers.0.self_attn.synth.0.weight",
        "decoder.layers.0.self_attn.synth.0.weight",
        "decoder.layers.0.cross_attn.synth.0.weight",
    ):
        if key in state_dict:
            synth_hidden = int(state_dict[key].shape[0])
            break

    return {
        "vocab_size": int(vocab_size) if vocab_size is not None else None,
        "d_model": int(d_model),
        "n_heads": int(n_heads),
        "d_ff": int(d_ff),
        "n_layers_enc": int(n_layers_enc),
        "max_src_len": int(max_src_len),
        "attn_mode_self_enc": attn_self_enc,
        "synth_hidden": int(synth_hidden),
        "synth_fixed_random": bool(run_cfg.get("synth_fixed_random", False)),
        "gate_init": float(run_cfg.get("gate_init", 0.5)),
    }


@torch.no_grad()
def _eval_transformer_classification(run_dir: Path, *, split: str, run_cfg: Dict[str, Any], data_cfg0: Dict[str, Any]) -> Dict[str, float]:
    from transformers import AutoTokenizer

    ckpt_dir = _resolve_checkpoint_dir(run_dir)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, use_fast=True)
    state_dict = torch.load(ckpt_dir / "pytorch_model.bin", map_location="cpu")

    ensure_transformer_on_path()
    from transformer.config import TransformerConfig
    from transformer.models import EncoderClassifier

    det = _detect_transformer_config(state_dict, run_cfg, data_cfg0)
    vocab_size = det["vocab_size"] or len(tokenizer)

    # Load eval split via build_dataset() so CSV runs work.
    ds = _load_text_dataset(data_cfg0, split)

    num_labels = None
    if "classifier.weight" in state_dict:
        num_labels = int(state_dict["classifier.weight"].shape[0])
    if num_labels is None:
        # infer from dataset features
        num_labels = int(ds.features[data_cfg0["label_field"]].num_classes)  # type: ignore[index]

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id or 0
    if not (0 <= int(pad_id) < len(tokenizer)):
        pad_id = 0

    cfg = TransformerConfig(
        vocab_size=int(vocab_size),
        d_model=int(det["d_model"]),
        n_heads=int(det["n_heads"]),
        d_ff=int(det["d_ff"]),
        n_layers_enc=int(det["n_layers_enc"]),
        n_layers_dec=0,
        dropout=float(run_cfg.get("dropout", 0.1)),
        max_src_len=int(det["max_src_len"]),
        max_tgt_len=1,
        tie_tgt_embeddings=True,
        layer_norm_eps=1e-5,
        attn_mode_self_enc=str(det["attn_mode_self_enc"]),
        attn_mode_self_dec="vanilla",
        attn_mode_cross="vanilla",
        synth_hidden=int(det["synth_hidden"]),
        synth_fixed_random=bool(det["synth_fixed_random"]),
        gate_init=float(det["gate_init"]),
    )

    model = EncoderClassifier(cfg, num_labels=int(num_labels), pad_token_id=int(pad_id))
    model.load_state_dict(state_dict, strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # Restrict to fields needed
    text_fields = tuple(data_cfg0.get("text_fields") or ["text"])
    label_field = str(data_cfg0["label_field"])
    batch_size = int(run_cfg.get("batch_size", 64))
    max_len = int(min(int(run_cfg.get("max_len", 512)), int(det["max_src_len"])))

    correct = 0
    total = 0
    for i in range(0, len(ds), batch_size):
        batch = ds[i : i + batch_size]
        if len(text_fields) == 1:
            texts = batch[text_fields[0]]
            enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        else:
            t1 = batch[text_fields[0]]
            t2 = batch[text_fields[1]]
            enc = tokenizer(t1, t2, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        labels = torch.tensor(batch[label_field], dtype=torch.long).to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask)["logits"]
        preds = logits.argmax(dim=-1)
        correct += int((preds == labels).sum().item())
        total += int(labels.numel())

    return {"acc": correct / max(1, total), "n": float(total)}


@torch.no_grad()
def _eval_transformer_regression(run_dir: Path, *, split: str, run_cfg: Dict[str, Any], data_cfg0: Dict[str, Any]) -> Dict[str, float]:
    from transformers import AutoTokenizer

    ckpt_dir = _resolve_checkpoint_dir(run_dir)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, use_fast=True)
    state_dict = torch.load(ckpt_dir / "pytorch_model.bin", map_location="cpu")

    ensure_transformer_on_path()
    from transformer.config import TransformerConfig
    from transformer.models import EncoderRegressor

    det = _detect_transformer_config(state_dict, run_cfg, data_cfg0)
    vocab_size = det["vocab_size"] or len(tokenizer)

    # targets
    num_targets = 1
    if "regressor.weight" in state_dict:
        num_targets = int(state_dict["regressor.weight"].shape[0])
    label_fields = data_cfg0.get("label_fields")
    if not label_fields:
        lf = data_cfg0.get("label_field")
        label_fields = [lf] if lf else ["label"]
    label_fields = list(label_fields)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id or 0
    if not (0 <= int(pad_id) < len(tokenizer)):
        pad_id = 0

    cfg = TransformerConfig(
        vocab_size=int(vocab_size),
        d_model=int(det["d_model"]),
        n_heads=int(det["n_heads"]),
        d_ff=int(det["d_ff"]),
        n_layers_enc=int(det["n_layers_enc"]),
        n_layers_dec=0,
        dropout=float(run_cfg.get("dropout", 0.1)),
        max_src_len=int(det["max_src_len"]),
        max_tgt_len=1,
        tie_tgt_embeddings=True,
        layer_norm_eps=1e-5,
        attn_mode_self_enc=str(det["attn_mode_self_enc"]),
        attn_mode_self_dec="vanilla",
        attn_mode_cross="vanilla",
        synth_hidden=int(det["synth_hidden"]),
        synth_fixed_random=bool(det["synth_fixed_random"]),
        gate_init=float(det["gate_init"]),
    )

    model = EncoderRegressor(cfg, num_targets=int(num_targets), pad_token_id=int(pad_id))
    model.load_state_dict(state_dict, strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    ds = _load_text_dataset(data_cfg0, split)
    text_field = (data_cfg0.get("text_fields") or ["text"])[0]

    batch_size = int(run_cfg.get("batch_size", 64))
    max_len = int(min(int(run_cfg.get("max_len", 512)), int(det["max_src_len"])))

    se_sum = 0.0
    ae_sum = 0.0
    n = 0

    def _safe_float(v: Any) -> float:
        try:
            return float(v)
        except Exception:
            return float("nan")

    for i in range(0, len(ds), batch_size):
        batch = ds[i : i + batch_size]
        texts = batch[text_field]
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # Prefer the unified regression format from build_dataset(): 'labels' is a list[float] per row
        if "labels" in batch:
            labels = torch.tensor(batch["labels"], dtype=torch.float32, device=device)
        else:
            labels = torch.tensor(
                [[_safe_float(batch[f][j]) for f in label_fields] for j in range(len(texts))],
                dtype=torch.float32,
                device=device,
            )
        preds = model(input_ids=input_ids, attention_mask=attention_mask)["logits"]

        mask = torch.isfinite(labels)
        labels_zn = torch.nan_to_num(labels, nan=0.0, posinf=0.0, neginf=0.0)
        diff = preds - labels_zn
        se_sum += float(((diff**2) * mask).sum().item())
        ae_sum += float((diff.abs() * mask).sum().item())
        n += int(mask.sum().item())

    mse = se_sum / max(1, n)
    mae = ae_sum / max(1, n)
    return {"mse": mse, "rmse": mse ** 0.5, "mae": mae, "n": float(n)}


# ---------------------------------------------------------------------------
# Tabular evaluation (MLP/VAE)
# ---------------------------------------------------------------------------


def _load_local_module(mod_name: str, file_path: Path):
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(mod_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module {mod_name} from {file_path}")
    m = importlib.util.module_from_spec(spec)
    # Required for dataclasses and other libs that consult sys.modules.
    sys.modules[mod_name] = m
    spec.loader.exec_module(m)  # type: ignore[call-arg]
    return m


@torch.no_grad()
def _eval_tabular_mlp(run_dir: Path, split: str, *, max_eval_samples: int | None = None) -> Dict[str, float]:
    ckpt_dir = _resolve_checkpoint_dir(run_dir)
    meta = _read_json(ckpt_dir / "run_config.json")
    run_cfg = meta.get("run_config", {})
    data_cfg0 = meta.get("data_config", {})
    input_dim = int(meta.get("input_dim", 0) or 0)
    label_names = meta.get("label_names")

    # Apply eval subsampling by capping the requested split
    if max_eval_samples is not None:
        tmp = dict(data_cfg0)
        if split == "train":
            tmp["max_train_samples"] = int(max_eval_samples)
        elif split in ("validation", "val"):
            tmp["max_val_samples"] = int(max_eval_samples)
        else:
            tmp["max_test_samples"] = int(max_eval_samples)
        data_cfg0 = tmp

    data_cfg = DataConfig(**data_cfg0)
    ds = build_dataset(data_cfg)

    # pick split; fall back if missing
    split_key = split
    if split_key not in ds:
        split_key = data_cfg.split_val if data_cfg.split_val in ds else data_cfg.split_train
    # loaders
    train_loader, val_loader, test_loader, input_dim2, label_names2 = build_tabular_loaders(ds, data_cfg, batch_size=int(run_cfg.get("batch_size", 256)))
    loader = {data_cfg.split_train: train_loader, data_cfg.split_val: val_loader}.get(split_key)
    if loader is None:
        loader = test_loader
    if loader is None:
        loader = val_loader

    proj = Path(__file__).resolve().parents[1]
    nn_models = _load_local_module("nn_models", proj / "neural-net" / "models.py")
    nn_trainer = _load_local_module("nn_trainer", proj / "neural-net" / "trainer.py")

    num_classes = None
    if data_cfg.task_type == TaskType.CLASSIFICATION:
        if label_names2 is not None:
            num_classes = len(label_names2)
        elif label_names is not None:
            num_classes = len(label_names)
        else:
            # derive from encoded labels
            num_classes = int(torch.max(ds[data_cfg.split_train]["label"]).item()) + 1

    # rebuild model
    fam = str(run_cfg.get("model_family"))
    if fam == "mlp_small":
        if data_cfg.task_type == TaskType.CLASSIFICATION:
            model = nn_models.build_small_classifier(input_dim=input_dim2, num_classes=int(num_classes))
        else:
            model = nn_models.build_small_regressor(input_dim=input_dim2, num_targets=1)
    else:
        if data_cfg.task_type == TaskType.CLASSIFICATION:
            model = nn_models.build_large_classifier(input_dim=input_dim2, num_classes=int(num_classes))
        else:
            model = nn_models.build_large_regressor(input_dim=input_dim2, num_targets=1)

    state = torch.load(ckpt_dir / "pytorch_model.bin", map_location="cpu")
    model.load_state_dict(state, strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if data_cfg.task_type == TaskType.CLASSIFICATION:
        metrics = nn_trainer.evaluate_classification(model, loader, device)
        return {"acc": float(metrics["acc"]), "loss": float(metrics["loss"])}
    else:
        mse = nn_trainer.evaluate_regression(model, loader, device)
        return {"mse": float(mse), "rmse": float(mse ** 0.5)}


@torch.no_grad()
def _eval_tabular_vae(run_dir: Path, split: str, *, max_eval_samples: int | None = None) -> Dict[str, float]:
    ckpt_dir = _resolve_checkpoint_dir(run_dir)
    meta = _read_json(ckpt_dir / "run_config.json")
    run_cfg = meta.get("run_config", {})
    data_cfg0 = meta.get("data_config", {})

    if max_eval_samples is not None:
        tmp = dict(data_cfg0)
        if split == "train":
            tmp["max_train_samples"] = int(max_eval_samples)
        elif split in ("validation", "val"):
            tmp["max_val_samples"] = int(max_eval_samples)
        else:
            tmp["max_test_samples"] = int(max_eval_samples)
        data_cfg0 = tmp

    data_cfg = DataConfig(**data_cfg0)
    ds = build_dataset(data_cfg)

    split_key = split
    if split_key not in ds:
        split_key = data_cfg.split_val if data_cfg.split_val in ds else data_cfg.split_train

    train_loader, val_loader, test_loader, input_dim, label_names = build_tabular_loaders(ds, data_cfg, batch_size=int(run_cfg.get("batch_size", 256)))
    loader = {data_cfg.split_train: train_loader, data_cfg.split_val: val_loader}.get(split_key)
    if loader is None:
        loader = test_loader
    if loader is None:
        loader = val_loader

    proj = Path(__file__).resolve().parents[1]
    # VAE trainer uses `from models import ...` so we must load its models module as `models`.
    vae_models = _load_local_module("models", proj / "vae-encoder" / "models.py")
    vae_trainer = _load_local_module("vae_trainer", proj / "vae-encoder" / "trainer.py")

    latent_dim = int(run_cfg.get("vae_latent_dim", 32))
    enc = vae_models.TabularVAEEncoder(vae_models.VAEConfig(input_dim=input_dim, latent_dim=latent_dim))

    if data_cfg.task_type == TaskType.CLASSIFICATION:
        num_classes = int(len(label_names) if label_names is not None else int(torch.max(ds[data_cfg.split_train]["label"]).item()) + 1)
        head = vae_models.ClassificationHead(latent_dim=latent_dim, num_classes=num_classes, hidden=None)
    else:
        head = vae_models.RegressionHead(latent_dim=latent_dim, num_targets=1, hidden=None)

    model = vae_models.EncoderWithHead(enc, head, use_mean=True)
    state = torch.load(ckpt_dir / "pytorch_model.bin", map_location="cpu")
    model.load_state_dict(state, strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if data_cfg.task_type == TaskType.CLASSIFICATION:
        metrics = vae_trainer.evaluate_classification(model, loader, device)
        return {"acc": float(metrics["acc"]), "loss": float(metrics["loss"])}
    else:
        mse = vae_trainer.evaluate_regression(model, loader, device)
        return {"mse": float(mse), "rmse": float(mse ** 0.5)}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate_run(run_dir: str, *, split: str = "test", max_eval_samples: int | None = None) -> Dict[str, Any]:
    """Evaluate a run directory.

    split: "test" / "validation" / "train".
    For HF datasets you can pass dataset slicing syntax (e.g. "test[:2000]").
    """

    run_path = Path(run_dir)
    if not run_path.exists():
        raise FileNotFoundError(run_dir)
    run_json = run_path / "run.json"
    if not run_json.exists():
        raise FileNotFoundError(f"Missing run.json in {run_path}")

    meta = _read_json(run_json)
    run_cfg = meta.get("run_config", {})
    data_cfg = meta.get("data_config", {})
    model_family = str(run_cfg.get("model_family"))
    task = str(run_cfg.get("task"))

    # Note: split parsing for tabular runs: allow "test"/"validation"/"train" only.
    tab_split = split.split(":")[0] if ":" in split else split

    if model_family == "transformer":
        if max_eval_samples is not None:
            # For transformer eval we can use HF split slicing syntax.
            if ":" in split:
                raise ValueError("max_eval_samples cannot be combined with HF slicing syntax in --split")
            split = f"{split}[:{int(max_eval_samples)}]"
        if task == TaskType.CLASSIFICATION.value:
            metrics = _eval_transformer_classification(run_path, split=split, run_cfg=run_cfg, data_cfg0=data_cfg)
        else:
            metrics = _eval_transformer_regression(run_path, split=split, run_cfg=run_cfg, data_cfg0=data_cfg)
    elif model_family in ("mlp_small", "mlp_large"):
        metrics = _eval_tabular_mlp(run_path, tab_split, max_eval_samples=max_eval_samples)
    elif model_family == "vae":
        metrics = _eval_tabular_vae(run_path, tab_split, max_eval_samples=max_eval_samples)
    else:
        raise ValueError(f"Unknown model_family in run.json: {model_family}")

    results = {
        "run_dir": str(run_path),
        "model_family": model_family,
        "task": task,
        "split": split,
        "metrics": metrics,
    }
    append_to_json(f"results/{model_family}_{task}.json", results)

    return results

def append_to_json(filename, new_data):
    # 1. Read existing data (or initialize if the file is empty/doesn't exist)
    data = []
    if os.path.exists(filename) and os.stat(filename).st_size != 0:
        with open(filename, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                # Handle case where file is corrupted or not a valid JSON list initially
                data = []

    # Ensure the loaded data is a list to support appending
    if not isinstance(data, list):
        # You might want to handle this case differently, e.g., wrap the existing data in a list
        raise TypeError(f"JSON file does not contain a list. Found type: {type(data)}")

    # 2. Append the new data to the Python object
    data.append(new_data)

    # 3. Write the entire updated data back to the file
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

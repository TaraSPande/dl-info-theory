"""Unified training/evaluation entrypoints.

This repo has three model families:
  - Transformer encoder (text -> token ids) for classification/regression.
  - MLP (tabular -> float features) for classification/regression.
  - VAE (tabular -> float features) trained as a VAE then used with a head.

This module provides a succinct dispatcher that:
  1) loads data from HuggingFace or CSV
  2) trains the selected model family for the selected task
  3) saves artifacts into runs/<run_name>/{run.json, train.log, final/}
  4) can call evaluation functions for the run.
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import torch

from preprocess import (
    DATA_REGISTRY,
    DataConfig,
    DataSource,
    InputType,
    TaskType,
    TokenizerConfig,
    build_dataset,
    build_tabular_loaders,
    build_tokenizer,
)


ModelFamily = Literal["transformer", "mlp_small", "mlp_large", "vae"]


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def ensure_transformer_on_path() -> Path:
    """Add ./transformer-encoder to sys.path so `import transformer` works."""
    tx_root = _project_root() / "transformer-encoder"
    if str(tx_root) not in sys.path:
        sys.path.insert(0, str(tx_root))
    return tx_root


@dataclass
class RunConfig:
    """Serializable run configuration.

    Notes:
      - For HF data, set data_key to a key in preprocess.DATA_REGISTRY.
      - For CSV/tabular, set data_source=csv and specify csv_path + feature_fields + label_field.
    """

    name: str
    task: TaskType
    model_family: ModelFamily

    # data
    # Option A (recommended): use a registry key in preprocess.DATA_REGISTRY
    data_key: Optional[str] = None

    # Option B: specify HF dataset fields directly
    hf_dataset_id: Optional[str] = None
    hf_dataset_config: Optional[str] = None
    hf_text_fields: Optional[list[str]] = None

    data_source: DataSource = DataSource.HF

    # CSV/tabular
    csv_path: Optional[str] = None
    feature_fields: Optional[list[str]] = None
    label_field: Optional[str] = None
    # CSV split sizes (only used when config.data_source==csv and only a single CSV is provided).
    val_size: float = 0.1
    test_size: float = 0.1

    # Optional subsampling caps (applied after split creation)
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None
    max_test_samples: Optional[int] = None

    # If classification and labels are strings, class encoding will handle it.
    # If labels are already ints, that's fine too.

    # tokenizer (transformer only)
    tokenizer: str = "gpt2"
    max_len: int = 256

    # training
    epochs: int = 10
    batch_size: int = 128
    lr: float = 5e-4
    fp16: bool = False
    warmup_steps: int = 4000
    grad_accum_steps: int = 1
    seed: int = 42

    # transformer architecture
    d_model: int = 512
    heads: int = 8
    layers_enc: int = 6
    d_ff_scale: int = 4
    dropout: float = 0.1
    attn_self_enc: str = "vanilla"
    synth_hidden: int = 512
    synth_fixed_random: bool = False
    gate_init: float = 0.5

    # VAE specific
    vae_latent_dim: int = 32
    vae_beta: float = 1.0
    vae_recon: Literal["mse", "l1"] = "mse"
    vae_freeze_encoder: bool = True

    out_root: str = "./runs"

    def slug(self) -> str:
        parts = [self.task.value, self.model_family]
        if self.data_key:
            parts.append(self.data_key)
        elif self.csv_path:
            parts.append(Path(self.csv_path).stem)
        if self.model_family == "transformer":
            parts.append(f"enc{self.layers_enc}d{self.d_model}h{self.heads}")
            parts.append(self.attn_self_enc)
        return "-".join(parts)


@contextmanager
def tee_to_file(log_path: str):
    """Mirror stdout/stderr to a logfile while also keeping console output."""

    class Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()

        def flush(self):
            for s in self.streams:
                s.flush()

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", buffering=1) as f:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = Tee(sys.stdout, f)
        sys.stderr = Tee(sys.stderr, f)
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_out, old_err


def unique_run_dir(root: str, slug: str, *, suffix: Optional[str] = None) -> str:
    """Create a unique run directory.

    If suffix is provided, it is used instead of a timestamp.
    """
    ts_or_suffix = suffix if suffix else datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(root, f"{slug}_{ts_or_suffix}")
    path = base
    n = 1
    while os.path.exists(path):
        n += 1
        path = f"{base}_{n}"
    os.makedirs(path, exist_ok=False)
    return path


def _make_data_config(cfg: RunConfig) -> DataConfig:
    """Resolve a DataConfig from a run config."""
    if cfg.data_source == DataSource.HF:
        if cfg.data_key:
            dc = DATA_REGISTRY[cfg.data_key]
            # Override task type (lets you run the same dataset under different heads if desired)
            # but keep input_type.
            dc = DataConfig(**{**asdict(dc), "task_type": cfg.task})  # type: ignore[arg-type]
            return DataConfig(**{**asdict(dc), "max_length": int(cfg.max_len)})  # type: ignore[arg-type]

        # Direct HF specification
        if not cfg.hf_dataset_id:
            raise ValueError("For HF data_source you must set either data_key or hf_dataset_id")
        if not cfg.label_field:
            raise ValueError("label_field is required for HF runs (classification label or regression target)")
        tf = tuple(cfg.hf_text_fields or ["text"])
        return DataConfig(
            name=cfg.name,
            task_type=cfg.task,
            data_source=DataSource.HF,
            input_type=InputType.TEXT,
            dataset_id=str(cfg.hf_dataset_id),
            dataset_config=cfg.hf_dataset_config,
            text_fields=tf,
            label_field=str(cfg.label_field),
            split_train="train",
            split_val="validation",
            split_test="test",
            val_size=0.1,
            test_size=0.0,
            split_seed=int(cfg.seed),
            max_length=int(cfg.max_len),
        )

    # CSV
    if not cfg.csv_path:
        raise ValueError("csv_path required for csv data_source")
    if not cfg.feature_fields:
        raise ValueError("feature_fields required for csv/tabular")
    if not cfg.label_field:
        raise ValueError("label_field required for csv/tabular")

    # Transformer on CSV numeric features: stringify features into a synthetic text field.
    if cfg.model_family == "transformer":
        return DataConfig(
            name=cfg.name,
            task_type=cfg.task,
            data_source=DataSource.CSV,
            input_type=InputType.TEXT,
            dataset_id=str(cfg.csv_path),
            csv_files={"train": str(cfg.csv_path)},
            # Reuse feature_fields to build text; then read from the generated "text".
            feature_fields=tuple(cfg.feature_fields),
            text_fields=("text",),
            label_field=str(cfg.label_field),
            split_train="train",
            split_val="validation",
            split_test="test",
            val_size=float(cfg.val_size),
            test_size=float(cfg.test_size),
            split_seed=int(cfg.seed),
            max_train_samples=cfg.max_train_samples,
            max_val_samples=cfg.max_val_samples,
            max_test_samples=cfg.max_test_samples,
            max_length=int(cfg.max_len),
        )

    return DataConfig(
        name=cfg.name,
        task_type=cfg.task,
        data_source=DataSource.CSV,
        input_type=InputType.TABULAR,
        dataset_id=str(cfg.csv_path),
        csv_files={"train": str(cfg.csv_path)},
        feature_fields=tuple(cfg.feature_fields),
        label_field=str(cfg.label_field),
        split_train="train",
        split_val="validation",
        split_test="test",
        val_size=float(cfg.val_size),
        test_size=float(cfg.test_size),
        split_seed=int(cfg.seed),
        max_train_samples=cfg.max_train_samples,
        max_val_samples=cfg.max_val_samples,
        max_test_samples=cfg.max_test_samples,
    )


def _save_run_json(run_dir: str, run_cfg: RunConfig, data_cfg: DataConfig) -> None:
    payload = {
        "run_config": asdict(run_cfg),
        "data_config": asdict(data_cfg),
        "created_at": datetime.now().isoformat(),
    }
    with open(os.path.join(run_dir, "run.json"), "w") as f:
        json.dump(payload, f, indent=2)


def train_run(run_cfg: RunConfig) -> str:
    """Train a run and return the run_dir."""
    suffix = None
    if run_cfg.max_train_samples is not None:
        suffix = f"n{int(run_cfg.max_train_samples)}"
    run_dir = unique_run_dir(run_cfg.out_root, run_cfg.slug(), suffix=suffix)
    log_path = os.path.join(run_dir, "train.log")

    data_cfg = _make_data_config(run_cfg)
    _save_run_json(run_dir, run_cfg, data_cfg)

    with tee_to_file(log_path):
        print("=" * 80)
        print("Run:", run_cfg.name)
        print("Run dir:", run_dir)
        print("Config:", json.dumps(asdict(run_cfg), indent=2))
        print("=" * 80)

        if run_cfg.model_family == "transformer":
            if data_cfg.input_type != InputType.TEXT:
                raise ValueError("Transformer requires input_type=text")

            # make transformer package importable
            ensure_transformer_on_path()

            import importlib

            trainer_mod = importlib.import_module("trainer")
            Trainer = getattr(trainer_mod, "Trainer")
            TrainConfig = getattr(trainer_mod, "TrainConfig")

            from transformer.models import build_classifier, build_regressor

            tok = build_tokenizer(TokenizerConfig(name_or_path=run_cfg.tokenizer, max_length=run_cfg.max_len))
            vocab_size = len(tok)

            common_kwargs = dict(
                d_model=run_cfg.d_model,
                n_heads=run_cfg.heads,
                d_ff=run_cfg.d_ff_scale * run_cfg.d_model,
                dropout=run_cfg.dropout,
                synth_hidden=run_cfg.synth_hidden,
                synth_fixed_random=run_cfg.synth_fixed_random,
                gate_init=run_cfg.gate_init,
                n_layers_enc=run_cfg.layers_enc,
                max_src_len=run_cfg.max_len,
                attn_mode_self_enc=run_cfg.attn_self_enc,
            )

            if run_cfg.task == TaskType.CLASSIFICATION:
                # infer classes from dataset
                raw = build_dataset(data_cfg)
                num_labels = raw[data_cfg.split_train].features[data_cfg.label_field].num_classes  # type: ignore[index]
                model = build_classifier(vocab_size=vocab_size, pad_token_id=tok.pad_token_id, num_labels=num_labels, **common_kwargs)
            else:
                # Determine num_targets from data_cfg if multi-target; default 1
                if data_cfg.label_fields is not None:
                    num_targets = len(data_cfg.label_fields)
                else:
                    num_targets = 1
                model = build_regressor(vocab_size=vocab_size, pad_token_id=tok.pad_token_id, num_targets=num_targets, **common_kwargs)

            trainer = Trainer(
                model=model,
                tokenizer=tok,
                data_config=data_cfg,
                train_config=TrainConfig(
                    # Transformer trainer writes its own epoch*/best/final subfolders.
                    output_dir=run_dir,
                    epochs=run_cfg.epochs,
                    batch_size=run_cfg.batch_size,
                    lr=run_cfg.lr,
                    fp16=run_cfg.fp16,
                    grad_accum_steps=run_cfg.grad_accum_steps,
                    warmup_steps=run_cfg.warmup_steps,
                    seed=run_cfg.seed,
                ),
            )
            trainer.train()
            return run_dir

        # tabular (MLP / VAE)
        ds = build_dataset(data_cfg)
        train_loader, val_loader, test_loader, input_dim, label_names = build_tabular_loaders(
            ds, data_cfg, batch_size=run_cfg.batch_size
        )

        if run_cfg.model_family in ("mlp_small", "mlp_large"):
            # local imports from hyphenated dirs: use importlib tricks by path
            import importlib.util
            import sys

            def _load_module(mod_name: str, file_path: str):
                spec = importlib.util.spec_from_file_location(mod_name, file_path)
                if spec is None or spec.loader is None:
                    raise RuntimeError(f"Cannot load module {mod_name} from {file_path}")
                m = importlib.util.module_from_spec(spec)
                # Needed for dataclasses (and other libs) that look up the module in sys.modules.
                sys.modules[mod_name] = m
                spec.loader.exec_module(m)  # type: ignore[call-arg]
                return m

            nn_models = _load_module("nn_models", os.path.join(os.path.dirname(__file__), "neural-net", "models.py"))
            nn_trainer = _load_module("nn_trainer", os.path.join(os.path.dirname(__file__), "neural-net", "trainer.py"))

            if run_cfg.task == TaskType.CLASSIFICATION:
                # determine num classes from encoded labels
                num_classes = int(len(label_names) if label_names is not None else int(torch.max(ds[data_cfg.split_train]["label"]).item()) + 1)
                if run_cfg.model_family == "mlp_small":
                    model = nn_models.build_small_classifier(input_dim=input_dim, num_classes=num_classes)
                else:
                    model = nn_models.build_large_classifier(input_dim=input_dim, num_classes=num_classes)
                nn_trainer.train_classification(
                    model,
                    train_loader,
                    val_loader,
                    nn_trainer.TrainConfig(epochs=run_cfg.epochs, lr=run_cfg.lr, device=None, log_every=50),
                )
            else:
                if run_cfg.model_family == "mlp_small":
                    model = nn_models.build_small_regressor(input_dim=input_dim, num_targets=1)
                else:
                    model = nn_models.build_large_regressor(input_dim=input_dim, num_targets=1)
                nn_trainer.train_regression(
                    model,
                    train_loader,
                    val_loader,
                    nn_trainer.TrainConfig(epochs=run_cfg.epochs, lr=run_cfg.lr, device=None, log_every=50),
                )

            # Save final checkpoint
            ckpt_dir = os.path.join(run_dir, "final")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "pytorch_model.bin"))
            with open(os.path.join(ckpt_dir, "run_config.json"), "w") as f:
                json.dump(
                    {
                        "run_config": asdict(run_cfg),
                        "data_config": asdict(data_cfg),
                        "model_class": model.__class__.__name__,
                        "label_names": label_names,
                        "input_dim": input_dim,
                    },
                    f,
                    indent=2,
                )
            return run_dir

        if run_cfg.model_family == "vae":
            import importlib.util
            import sys

            def _load_module(mod_name: str, file_path: str):
                spec = importlib.util.spec_from_file_location(mod_name, file_path)
                if spec is None or spec.loader is None:
                    raise RuntimeError(f"Cannot load module {mod_name} from {file_path}")
                m = importlib.util.module_from_spec(spec)
                sys.modules[mod_name] = m
                spec.loader.exec_module(m)  # type: ignore[call-arg]
                return m

            # The VAE trainer uses `from models import ...`.
            # So we load vae-encoder/models.py as module name `models`.
            vae_models = _load_module("models", os.path.join(os.path.dirname(__file__), "vae-encoder", "models.py"))
            vae_trainer = _load_module("vae_trainer", os.path.join(os.path.dirname(__file__), "vae-encoder", "trainer.py"))

            # Train VAE on x only
            vae = vae_models.TabularVAE(vae_models.VAEConfig(input_dim=input_dim, latent_dim=run_cfg.vae_latent_dim))
            vae_trainer.train_vae(
                vae,
                train_loader,
                val_loader,
                cfg=vae_trainer.TrainConfig(epochs=run_cfg.epochs, lr=run_cfg.lr, device=None, log_every=50),
                beta=run_cfg.vae_beta,
                recon=run_cfg.vae_recon,
            )

            encoder = vae.encoder
            if run_cfg.task == TaskType.CLASSIFICATION:
                num_classes = int(len(label_names) if label_names is not None else int(torch.max(ds[data_cfg.split_train]["label"]).item()) + 1)
                model = vae_trainer.train_classification_head(
                    encoder,
                    num_classes=num_classes,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    cfg=vae_trainer.TrainConfig(epochs=run_cfg.epochs, lr=run_cfg.lr, device=None, log_every=50),
                    freeze_encoder=run_cfg.vae_freeze_encoder,
                )
            else:
                model = vae_trainer.train_regression_head(
                    encoder,
                    num_targets=1,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    cfg=vae_trainer.TrainConfig(epochs=run_cfg.epochs, lr=run_cfg.lr, device=None, log_every=50),
                    freeze_encoder=run_cfg.vae_freeze_encoder,
                )

            ckpt_dir = os.path.join(run_dir, "final")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "pytorch_model.bin"))
            with open(os.path.join(ckpt_dir, "run_config.json"), "w") as f:
                json.dump(
                    {
                        "run_config": asdict(run_cfg),
                        "data_config": asdict(data_cfg),
                        "model_class": model.__class__.__name__,
                        "label_names": label_names,
                        "input_dim": input_dim,
                    },
                    f,
                    indent=2,
                )
            return run_dir

        raise ValueError(f"Unknown model_family: {run_cfg.model_family}")


def load_run(run_dir: str) -> Dict[str, Any]:
    with open(os.path.join(run_dir, "run.json")) as f:
        return json.load(f)

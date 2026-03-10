from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple, Union, List

import os
import math

import torch
from torch.utils.data import DataLoader

from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from utils import ensure_special_tokens

try:
    import sentencepiece as spm
except Exception:
    spm = None

# ------------------------------------------------------------
# Task types & config
# ------------------------------------------------------------

class TaskType(str, Enum):
    """
    Two modes the trainer understands:
        - REGRESSION
        - CLASSIFICATION
    """
    CLASSIFICATION = "classification"
    REGRESSION = "regression"

    # Backwards-compatible task types used by some legacy transformer code.
    # (Not used by the unified run pipeline, but kept so imports don't break.)
    CAUSAL_LM = "causal_lm"
    SEQ2SEQ = "seq2seq"


class DataSource(str, Enum):
    """Where the dataset comes from."""

    HF = "hf"
    CSV = "csv"


class InputType(str, Enum):
    """What the *model* expects as input."""

    TEXT = "text"         # e.g. Transformer encoder over token ids
    TABULAR = "tabular"   # e.g. MLP / VAE over float feature vectors


@dataclass
class DataConfig:
    """Declarative “recipe” for the dataset."""
    name: str                      # human name (e.g., "wmt14_en_de")
    task_type: TaskType
    data_source: DataSource = DataSource.HF
    input_type: InputType = InputType.TEXT

    # HF datasets
    dataset_id: Union[str, List[str]] = ""           # datasets.load_dataset name
    dataset_config: Optional[str] = None             # e.g., "de-en", "en", "3.0.0"

    # CSV datasets
    # Mapping split -> filepath, e.g. {"train": "train.csv", "validation": "val.csv"}
    csv_files: Optional[Dict[str, str]] = None

    # Text inputs (Transformer)
    text_fields: Tuple[str, ...] = ("text",)         # which columns from dataset to tokenize

    # Tabular inputs (MLP/VAE)
    feature_fields: Optional[Tuple[str, ...]] = None  # numeric feature columns

    # Labels
    label_field: Optional[str] = None                # cls label column; also supports single-target regression
    label_fields: Optional[Tuple[str, ...]] = None   # regression: multi-target continuous labels

    source_lang: Optional[str] = None   # for translation
    target_lang: Optional[str] = None   # for translation
    split_train: str = "train"
    split_val: str = "validation"
    split_test: Optional[str] = None

    # If split_val or split_test are missing, create them by splitting split_train.
    split_seed: int = 42
    val_size: float = 0.1
    test_size: float = 0.0

    # Optional subsampling for faster experiments/debugging.
    # Applied after split creation.
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None
    max_test_samples: Optional[int] = None

    max_length: int = 512
    max_target_length: int = 128       # for seq2seq targets (e.g., summaries)

# ------------------------------------------------------------
# Dataset registry & preprocessors
# ------------------------------------------------------------

# Helper: trim very long texts for faster demo runs
def _truncate_text(text: str, max_chars: int = 4000) -> str:
    return text if len(text) <= max_chars else text[:max_chars]


def _safe_float(v: Any) -> float:
    if v is None:
        return float("nan")
    if isinstance(v, str):
        s = v.strip()
        if s == "" or s.lower() in ("nan", "none", "null"):
            return float("nan")
    try:
        return float(v)
    except Exception:
        return float("nan")


def _safe_float0(v: Any) -> float:
    """Float cast with a 0.0 fallback (useful for features)."""
    out = _safe_float(v)
    return 0.0 if not math.isfinite(out) else out


def _load_raw_dataset(config: DataConfig) -> DatasetDict:
    """Load a raw DatasetDict from HF or CSV based on config."""
    if config.data_source == DataSource.CSV:
        if not config.csv_files:
            # Allow passing a single CSV path via dataset_id for convenience.
            if isinstance(config.dataset_id, str) and config.dataset_id:
                data_files = {"train": config.dataset_id}
            else:
                raise ValueError("CSV data_source requires csv_files or dataset_id=path/to.csv")
        else:
            data_files = config.csv_files
        return load_dataset("csv", data_files=data_files)

    # Default: HF
    if isinstance(config.dataset_id, list):
        all_splits: Dict[str, List[Dataset]] = {}
        for ds_id in config.dataset_id:
            part = load_dataset(ds_id, config.dataset_config)
            for split in part.keys():
                all_splits.setdefault(split, []).append(part[split])
        return DatasetDict({split: concatenate_datasets(parts) for split, parts in all_splits.items()})

    return load_dataset(config.dataset_id, config.dataset_config)


def _ensure_splits(ds: DatasetDict, config: DataConfig) -> DatasetDict:
    """Ensure config.split_val / split_test exist; create by splitting train if needed."""
    if config.split_train not in ds:
        raise ValueError(f"Train split '{config.split_train}' not found. Available: {list(ds.keys())}")

    # Build validation split if missing
    if config.split_val not in ds:
        if not (0.0 < float(config.val_size) < 1.0):
            raise ValueError("val_size must be in (0,1) when validation split is missing")
        tmp = ds[config.split_train].train_test_split(test_size=float(config.val_size), seed=int(config.split_seed))
        ds = DatasetDict({
            config.split_train: tmp["train"],
            config.split_val: tmp["test"],
            **{k: v for k, v in ds.items() if k not in (config.split_train, config.split_val)},
        })

    # Build test split if requested and missing
    if config.split_test and config.split_test not in ds:
        if not (0.0 < float(config.test_size) < 1.0):
            raise ValueError("test_size must be in (0,1) when test split is missing")
        tmp = ds[config.split_train].train_test_split(test_size=float(config.test_size), seed=int(config.split_seed))
        ds = DatasetDict({
            config.split_train: tmp["train"],
            config.split_test: tmp["test"],
            **{k: v for k, v in ds.items() if k not in (config.split_train, config.split_test)},
        })

    return ds


def _apply_max_samples(ds: DatasetDict, config: DataConfig) -> DatasetDict:
    """Optionally subsample each split for quick runs."""

    def _cap(split: str, n: Optional[int]) -> None:
        if n is None:
            return
        if split not in ds:
            return
        n = int(n)
        if n <= 0:
            raise ValueError(f"max_samples for split '{split}' must be > 0")
        if len(ds[split]) > n:
            ds[split] = ds[split].select(range(n))

    _cap(config.split_train, config.max_train_samples)
    _cap(config.split_val, config.max_val_samples)
    if config.split_test:
        _cap(config.split_test, config.max_test_samples)
    return ds


def build_dataset(config: DataConfig) -> DatasetDict:
    ds = _ensure_splits(_load_raw_dataset(config), config)
    ds = _apply_max_samples(ds, config)

    # -------------------------------------------------
    # CSV -> TEXT adapter (for Transformer on numeric features)
    # -------------------------------------------------
    # If you want to run a text model (Transformer) on a CSV that contains numeric
    # feature columns, we can stringify the features into a single "text" field.
    #
    # Example produced text:
    #   "MolWt=312.4 MolLogP=1.23 TPSA=77.1"
    if config.data_source == DataSource.CSV and config.input_type == InputType.TEXT and config.feature_fields:
        feat_fields = tuple(config.feature_fields)

        def to_text(batch):
            n = len(batch[feat_fields[0]])
            texts: list[str] = []
            for i in range(n):
                parts = [f"{f}={_safe_float0(batch[f][i]):.6g}" for f in feat_fields]
                texts.append(" ".join(parts))
            return {"text": texts}

        # Drop the numeric feature columns after we convert them to text.
        remove_cols = [f for f in feat_fields if f in ds[config.split_train].column_names]
        ds = ds.map(to_text, batched=True, remove_columns=remove_cols)

        # If user didn't explicitly set text_fields, default to the newly created "text".
        # (We can't mutate config here; callers should set text_fields=("text",) in config.)

    # ----------------------------
    # TABULAR (MLP/VAE)
    # ----------------------------
    if config.input_type == InputType.TABULAR:
        if not config.feature_fields:
            raise ValueError("feature_fields required for tabular inputs")
        if not config.label_field:
            raise ValueError("label_field required for tabular tasks")

        feat_fields = tuple(config.feature_fields)
        label_field = str(config.label_field)

        def map_tab(batch):
            n = len(batch[feat_fields[0]])
            xs = [[_safe_float0(batch[f][i]) for f in feat_fields] for i in range(n)]
            ys = batch[label_field]
            if config.task_type == TaskType.REGRESSION:
                ys = [_safe_float(y) for y in ys]
            return {"x": xs, "label": ys}

        # remove all original columns
        base_cols = ds[config.split_train].column_names
        ds = ds.map(map_tab, batched=True, remove_columns=base_cols)

        # Encode labels for classification into a contiguous int space.
        if config.task_type == TaskType.CLASSIFICATION:
            # Works across DatasetDict splits.
            ds = ds.class_encode_column("label")

        return ds

    # ----------------------------
    # TEXT (Transformer)
    # ----------------------------
    if config.task_type == TaskType.REGRESSION:
        # Support either label_fields (multi-target) or label_field (single-target)
        if config.label_fields is not None:
            label_fields = tuple(config.label_fields)
        elif config.label_field is not None:
            label_fields = (str(config.label_field),)
        else:
            raise ValueError("label_field or label_fields required for regression tasks")

        def map_fn(batch):
            texts = batch[config.text_fields[0]]
            labels = [[_safe_float(batch[f][i]) for f in label_fields] for i in range(len(texts))]
            return {"text": texts, "labels": labels}

        train_cols = ds[config.split_train].column_names
        ds = ds.map(map_fn, batched=True, remove_columns=train_cols)
        return ds

    if config.task_type == TaskType.CLASSIFICATION:
        if config.label_field is None:
            raise ValueError("label_field required for classification tasks")

        # For CSV classification, encode labels so downstream code can infer num_classes.
        if config.data_source == DataSource.CSV:
            try:
                ds = ds.class_encode_column(str(config.label_field))
            except Exception:
                # If the label is already integer-like, class_encode_column may fail; that's OK.
                pass
        return ds

    raise ValueError(f"Unknown task type: {config.task_type}")


# Predefined DataConfigs for requested datasets
DATA_REGISTRY: Dict[str, DataConfig] = {
    "smiles_properties": DataConfig(
        name="smiles_properties",
        task_type=TaskType.REGRESSION,
        data_source=DataSource.HF,
        input_type=InputType.TEXT,
        dataset_id="maykcaldas/smiles-transformers",
        dataset_config=None,
        text_fields=("text",),  # only SMILES goes here
        label_fields=(
            "NumHDonors",
            "NumHAcceptors",
            "MolLogP",
            "NumHeteroatoms",
            "RingCount",
            "NumRotatableBonds",
            "NumAromaticBonds",
            "NumAcidGroups",
            "NumBasicGroups",
            "Apol",
        ),
        split_train="train",
        split_val="validation",
        split_test="test"
    ),
    "smiles_selfies": DataConfig(
        name="smiles_selfies",
        task_type=TaskType.CLASSIFICATION,
        data_source=DataSource.HF,
        input_type=InputType.TEXT,
        dataset_id="mikemayuare/PubChem10M_SMILES_SELFIES",
        dataset_config=None,
        text_fields=("SMILES", "SELFIES"),
        split_train="train",
        split_val="train"
    ),
}

@dataclass
class TokenizerConfig:
    name_or_path: str = "gpt2"
    use_fast: bool = True
    max_length: int = 512
    max_target_length: int = 128


def build_tokenizer(cfg: TokenizerConfig) -> PreTrainedTokenizerBase:
    tok = AutoTokenizer.from_pretrained(cfg.name_or_path, use_fast=cfg.use_fast)
    tok = ensure_special_tokens(tok)
    return tok


# ------------------------------------------------------------
# Offline tokenization (speeds up training by avoiding per-batch tokenization)
# ------------------------------------------------------------
def tokenize_dataset(ds: DatasetDict, tokenizer: PreTrainedTokenizerBase, config: DataConfig) -> DatasetDict:
    """
    Pre-tokenize dataset examples so the collator only pads/batches.

    - SEQ2SEQ (e.g., WMT14): produce "src_ids" and "tgt_ids".
    - CAUSAL_LM: produce "input_ids" from "text".
    - CLASSIFICATION: produce "input_ids" from text_fields (single or pair).

    All sequences are kept variable-length; padding happens in the collator.
    """
    if config.task_type == TaskType.REGRESSION:
        def tok_fn_reg(batch):
            enc = tokenizer(
                batch["text"],
                truncation=True,
                max_length=config.max_length,
                padding=False,
                add_special_tokens=True,
            )
            return {"input_ids": enc["input_ids"]}

        train_cols = ds[config.split_train].column_names
        remove_cols = [c for c in ("text",) if c in train_cols]
        ds = ds.map(tok_fn_reg, batched=True, remove_columns=remove_cols)
        return ds

    elif config.task_type == TaskType.CLASSIFICATION:
        fields = config.text_fields

        def tok_fn_cls(batch):
            if len(fields) == 1:
                enc = tokenizer(
                    batch[fields[0]],
                    truncation=True,
                    max_length=config.max_length,
                    padding=False,
                    add_special_tokens=True,
                )
            else:
                enc = tokenizer(
                    batch[fields[0]],
                    batch[fields[1]],
                    truncation=True,
                    max_length=config.max_length,
                    padding=False,
                    add_special_tokens=True,
                )
            return {"input_ids": enc["input_ids"]}

        train_cols = ds[config.split_train].column_names
        # remove only the text fields; keep label_field and any metadata
        remove_cols = [c for c in fields if c in train_cols]
        ds = ds.map(tok_fn_cls, batched=True, remove_columns=remove_cols)
        return ds

    # Default: no changes
    return ds


def build_tabular_loaders(
    ds: DatasetDict,
    config: DataConfig,
    *,
    batch_size: int,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, Optional[DataLoader], int, Optional[list[str]]]:
    """Build torch DataLoaders for TABULAR datasets.

    Expects build_dataset() to have produced columns:
      - x: float list length D
      - label: float (regression) or int class id (classification)

    Returns: (train_loader, val_loader, test_loader, input_dim, label_names)
    """
    if config.input_type != InputType.TABULAR:
        raise ValueError("build_tabular_loaders requires config.input_type == TABULAR")

    train = ds[config.split_train]
    val = ds[config.split_val]
    test = ds[config.split_test] if (config.split_test and config.split_test in ds) else None

    input_dim = len(train[0]["x"]) if len(train) > 0 else len(config.feature_fields or [])

    # label names (classification only)
    label_names = None
    if config.task_type == TaskType.CLASSIFICATION:
        try:
            feat = train.features["label"]
            if hasattr(feat, "names"):
                label_names = list(feat.names)  # type: ignore[attr-defined]
        except Exception:
            label_names = None

    cols = ["x", "label"]
    train = train.with_format("torch", columns=cols)
    val = val.with_format("torch", columns=cols)
    if test is not None:
        test = test.with_format("torch", columns=cols)

    def _collate(batch):
        # batch: list[dict]
        x = torch.stack([b["x"].to(torch.float32) for b in batch], dim=0)
        if config.task_type == TaskType.CLASSIFICATION:
            y = torch.stack([b["label"].to(torch.int64) for b in batch], dim=0)
        else:
            y = torch.stack([b["label"].to(torch.float32) for b in batch], dim=0).unsqueeze(-1)
        return x, y

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin, collate_fn=_collate)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin, collate_fn=_collate)
    test_loader = None
    if test is not None:
        test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin, collate_fn=_collate)

    return train_loader, val_loader, test_loader, int(input_dim), label_names
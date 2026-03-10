"""Experiment helpers.

Historically this file contained a transformer-only experiment grid.

Now it provides:
  - the legacy Experiment/EXPERIMENTS list (kept for compatibility)
  - a new `run_matrix()` helper that uses the unified pipeline in runlib.py
"""

from dataclasses import dataclass
from typing import List, Optional

from preprocess import DataSource, TaskType
from runlib import RunConfig, train_run

# ---------------------------------------------------------------------------
# Experiment config
# ---------------------------------------------------------------------------

@dataclass
class Experiment:
    name: str
    data_key: str
    out_root: str = "./runs"
    tokenizer: str = "gpt2"

    # training
    epochs: int = 10
    batch_size: int = 128
    lr: float = 5e-4
    fp16: bool = False #true when using new enough GPU + supporting dataset
    grad_accum_steps: int = 1
    warmup_steps: int = 4000
    spm32k: bool = False #sentencepiece 32k overrides tokenizer (use for seq2seq)

    # model
    model_dim: int = 512
    heads: int = 8
    layers: int = 6  # used for encoder and/or decoder unless overridden
    layers_enc: Optional[int] = None
    layers_dec: Optional[int] = None
    d_ff_scale: int = 4  # FF size = d_ff_scale * model_dim -> filter size
    dropout: float = 0.1

    # lengths
    max_len: int = 256         # encoder or decoder (LM)
    max_tgt_len: int = 128     # decoder (seq2seq only)

    # attention choices
    attn_self_enc: str = "vanilla"
    attn_self_dec: str = "vanilla"
    attn_cross: str = "vanilla"
    synth_hidden: int = 512
    synth_fixed_random: bool = False
    gate_init: float = 0.5

    def slug(self) -> str:
        enc_layers = self.layers_enc or self.layers
        dec_layers = self.layers_dec or self.layers
        return (
            f"{self.data_key}-enc{enc_layers}dec{dec_layers}-d{self.model_dim}h{self.heads}-"
            f"{self.attn_self_enc}.{self.attn_self_dec}.{self.attn_cross}"
        )


# Experiment Suite
EXPERIMENTS: List[Experiment] = [
    Experiment(
        name="smiles_iupac_vanilla",
        data_key="smiles_iupac_bootstrap",
        attn_self_enc="vanilla",
        attn_self_dec="vanilla",
        attn_cross="vanilla",
        spm32k=True, #tokenizer
        epochs=10,
        lr=0.5,
        warmup_steps=8000,
        max_len=512,
        max_tgt_len=512,
    ),
    Experiment(
        name="smiles_iupac_dense",
        data_key="smiles_iupac_bootstrap",
        attn_self_enc="synth_dense",
        attn_self_dec="synth_dense",
        attn_cross="synth_dense",
        spm32k=True, #tokenizer
        epochs=10,
        lr=0.5,
        warmup_steps=8000,
        max_len=512,
        max_tgt_len=512,
    ),
    Experiment(
        name="iupac_smiles_vanilla",
        data_key="iupac_smiles_bootstrap",
        attn_self_enc="vanilla",
        attn_self_dec="vanilla",
        attn_cross="vanilla",
        spm32k=True, #tokenizer
        epochs=10,
        lr=0.5,
        warmup_steps=8000,
        max_len=512,
        max_tgt_len=512,
    ),
    Experiment(
        name="iupac_smiles_dense",
        data_key="iupac_smiles_bootstrap",
        attn_self_enc="synth_dense",
        attn_self_dec="synth_dense",
        attn_cross="synth_dense",
        spm32k=True, #tokenizer
        epochs=10,
        lr=0.5,
        warmup_steps=8000,
        max_len=512,
        max_tgt_len=512,
    ),
    Experiment(
        name="smiles_properties_vanilla",
        data_key="smiles_properties",
        attn_self_enc="vanilla",
        attn_self_dec="vanilla",
        attn_cross="vanilla",
        spm32k=True, #tokenizer
        epochs=1,
        lr=5e-4,
        warmup_steps=100000,
        max_len=512,
    ),
    Experiment(
        name="smiles_properties_dense",
        data_key="smiles_properties",
        attn_self_enc="synth_dense",
        attn_self_dec="synth_dense",
        attn_cross="synth_dense",
        spm32k=True, #tokenizer
        epochs=1,
        lr=5e-4,
        warmup_steps=100000,
        max_len=512,
    ),
    Experiment(
        name="smiles_properties_denseQ",
        data_key="smiles_properties",
        attn_self_enc="denseQ",
        attn_self_dec="denseQ",
        attn_cross="denseQ",
        spm32k=True, #tokenizer
        epochs=1,
        lr=5e-4,
        warmup_steps=100000,
        max_len=512,
    ),
]


# ---------------------------------------------------------------------------
# Unified experiment runner
# ---------------------------------------------------------------------------


def run_matrix(
    *,
    tasks: list[TaskType],
    models: list[str],
    out_root: str = "./runs",
    csv_path: Optional[str] = None,
    feature_fields: Optional[list[str]] = None,
    label_field: Optional[str] = None,
    data_key: Optional[str] = None,
) -> list[str]:
    """Train a Cartesian product of tasks × models.

    Provide either:
      - data_key (HF registry key)
      - OR csv_path + feature_fields + label_field (tabular)

    Returns a list of run directories.
    """

    run_dirs: list[str] = []

    for task in tasks:
        for model in models:
            if csv_path is not None:
                rc = RunConfig(
                    name=f"{task.value}-{model}",
                    task=task,
                    model_family=model,  # type: ignore[arg-type]
                    data_source=DataSource.CSV,
                    csv_path=csv_path,
                    feature_fields=feature_fields,
                    label_field=label_field,
                    out_root=out_root,
                )
            else:
                rc = RunConfig(
                    name=f"{task.value}-{model}",
                    task=task,
                    model_family=model,  # type: ignore[arg-type]
                    data_source=DataSource.HF,
                    data_key=data_key,
                    out_root=out_root,
                )

            run_dir = train_run(rc)
            run_dirs.append(run_dir)

    return run_dirs

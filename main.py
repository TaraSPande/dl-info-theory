"""Repo entrypoint.

New unified CLI:

Train:
  python main.py train --task classification --model mlp_small --csv data.csv --features f1,f2 --label y
  python main.py train --task regression --model transformer --data-key smiles_properties

Evaluate:
  python main.py eval --run runs/<...>

This CLI is intentionally minimal; for grid runs, use experiments.py.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from preprocess import DataSource, TaskType
from runlib import RunConfig, train_run


def _parse_csv_list(s: str) -> list[str]:
    s = s.strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def cmd_train(args: argparse.Namespace) -> None:
    if args.csv is not None:
        data_source = DataSource.CSV
        data_key = None
        csv_path = args.csv
        feature_fields = _parse_csv_list(args.features)
        if not feature_fields:
            raise SystemExit("--features is required for CSV runs")
        label_field = args.label
        if not label_field:
            raise SystemExit("--label is required for CSV runs")
    else:
        data_source = DataSource.HF
        data_key = args.data_key
        csv_path = None
        feature_fields = None
        label_field = args.label

    cfg = RunConfig(
        name=args.name or "run",
        task=TaskType(args.task),
        model_family=args.model,
        data_source=data_source,
        data_key=data_key,
        csv_path=csv_path,
        feature_fields=feature_fields,
        label_field=label_field,
        val_size=args.val_size,
        test_size=args.test_size,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
        hf_dataset_id=args.hf_dataset_id,
        hf_dataset_config=args.hf_dataset_config,
        hf_text_fields=_parse_csv_list(args.hf_text_fields) if args.hf_text_fields else None,
        tokenizer=args.tokenizer,
        max_len=args.max_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        fp16=args.fp16,
        warmup_steps=args.warmup_steps,
        grad_accum_steps=args.grad_accum_steps,
        seed=args.seed,
        # transformer
        d_model=args.d_model,
        heads=args.heads,
        layers_enc=args.layers_enc,
        d_ff_scale=args.d_ff_scale,
        dropout=args.dropout,
        attn_self_enc=args.attn_self_enc,
        synth_hidden=args.synth_hidden,
        synth_fixed_random=args.synth_fixed_random,
        gate_init=args.gate_init,
        # vae
        vae_latent_dim=args.vae_latent_dim,
        vae_beta=args.vae_beta,
        vae_recon=args.vae_recon,
        vae_freeze_encoder=not args.vae_unfrozen,
        out_root=args.out_root,
    )

    run_dir = train_run(cfg)
    print("\nRun complete:")
    print(run_dir)


def cmd_eval(args: argparse.Namespace) -> None:
    # Lazy import to avoid importing torch/transformers at CLI start for non-eval.
    from evaluation.runner import evaluate_run

    metrics = evaluate_run(args.run, split=args.split, max_eval_samples=args.max_eval_samples)
    print(json.dumps(metrics, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unified training/evaluation CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    # ------------------------------------------------------------------
    # train
    # ------------------------------------------------------------------
    t = sub.add_parser("train", help="Train a model and save into runs/")
    t.add_argument("--name", type=str, default="", help="Human name (stored in run.json)")
    t.add_argument("--task", type=str, choices=[x.value for x in TaskType if x.value in ("classification", "regression")], required=True)
    t.add_argument("--model", type=str, choices=["transformer", "mlp_small", "mlp_large", "vae"], required=True)

    # data selection
    t.add_argument("--data-key", type=str, default=None, help="Key from preprocess.DATA_REGISTRY (HF only)")
    t.add_argument("--hf-dataset-id", type=str, default=None, help="HF dataset id (if not using --data-key)")
    t.add_argument("--hf-dataset-config", type=str, default=None)
    t.add_argument("--hf-text-fields", type=str, default=None, help="Comma-separated text fields (HF direct mode)")
    t.add_argument("--label", type=str, default=None, help="Label field (HF direct mode) or CSV label column")

    t.add_argument("--csv", type=str, default=None, help="CSV path (tabular); if set, uses CSV mode")
    t.add_argument("--features", type=str, default="", help="Comma-separated numeric feature columns (CSV mode)")

    # CSV splitting (only used when you pass a single CSV)
    t.add_argument("--val-size", type=float, default=0.05, help="Validation fraction (CSV mode)")
    t.add_argument("--test-size", type=float, default=0.05, help="Test fraction (CSV mode)")

    # Subsampling caps
    t.add_argument("--max-train-samples", type=int, default=None, help="Cap number of training samples")
    t.add_argument("--max-val-samples", type=int, default=None, help="Cap number of validation samples")
    t.add_argument("--max-test-samples", type=int, default=None, help="Cap number of test samples")

    # training
    t.add_argument("--out-root", type=str, default="./runs")
    t.add_argument("--epochs", type=int, default=1)
    t.add_argument("--batch-size", type=int, default=32)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--fp16", action="store_true")

    # transformer-only
    t.add_argument("--tokenizer", type=str, default="gpt2")
    t.add_argument("--max-len", type=int, default=256)
    t.add_argument("--warmup-steps", type=int, default=4000)
    t.add_argument("--grad-accum-steps", type=int, default=1)
    t.add_argument("--d-model", type=int, default=512)
    t.add_argument("--heads", type=int, default=8)
    t.add_argument("--layers-enc", type=int, default=6)
    t.add_argument("--d-ff-scale", type=int, default=4)
    t.add_argument("--dropout", type=float, default=0.1)
    t.add_argument("--attn-self-enc", type=str, default="vanilla")
    t.add_argument("--synth-hidden", type=int, default=512)
    t.add_argument("--synth-fixed-random", action="store_true")
    t.add_argument("--gate-init", type=float, default=0.5)

    # VAE-only
    t.add_argument("--vae-latent-dim", type=int, default=32)
    t.add_argument("--vae-beta", type=float, default=1.0)
    t.add_argument("--vae-recon", type=str, choices=["mse", "l1"], default="mse")
    t.add_argument("--vae-unfrozen", action="store_true", help="If set, do NOT freeze encoder when training head")

    t.set_defaults(func=cmd_train)

    # ------------------------------------------------------------------
    # eval
    # ------------------------------------------------------------------
    e = sub.add_parser("eval", help="Evaluate a saved run (uses final/)")
    e.add_argument("--run", type=str, required=True, help="Path to a run dir under runs/")
    e.add_argument("--split", type=str, default="test", help="Dataset split: test/validation/train")
    e.add_argument("--max-eval-samples", type=int, default=None, help="Cap number of examples evaluated")
    e.set_defaults(func=cmd_eval)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

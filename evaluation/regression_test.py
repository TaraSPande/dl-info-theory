"""Regression evaluation (run-based).

This module replaces the old hard-coded checkpoint script.

Usage:
  python -m evaluation.regression_test --run runs/<run_dir>
"""

from __future__ import annotations

import argparse
import json

from evaluation.runner import evaluate_run


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=str, required=True)
    p.add_argument("--split", type=str, default="test")
    args = p.parse_args()

    out = evaluate_run(args.run, split=args.split)
    if str(out.get("task")) != "regression":
        raise SystemExit(f"Run task is not regression: {out.get('task')}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

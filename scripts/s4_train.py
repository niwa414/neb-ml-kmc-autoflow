#!/usr/bin/env python3
"""s4_train — invoke train_migration_soap.py with species-parameterized flags and
evaluate the admission gate against metrics.json.

Not stdlib-free (runs the user's ML venv via subprocess). Idempotent: refuses to
overwrite an existing --output-dir unless --overwrite.

Gate contract: metrics.json test MAE must be ≤ --mae-threshold; per-bucket
reporting is printed for the user, but buckets with < --min-bucket samples get a
⚠️ warning rather than a gate verdict (avoids noise-driven deadlock).

Exit codes: 0 trained + gate evaluated (pass or warn), 1 gate failed (strict),
2 usage.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_TRAINER = Path(os.environ.get(
    "NEB_TRAINER", "./train_migration_soap.py")).expanduser()
DEFAULT_VENV = Path(os.environ.get(
    "NEB_TRAIN_PYTHON", sys.executable)).expanduser()


def find_test_metrics(metrics: dict) -> dict | None:
    def walk(d):
        if isinstance(d, dict):
            if set(("mae", "rmse")) <= set(d):
                return d
            for v in d.values():
                hit = walk(v)
                if hit:
                    return hit
        return None
    return walk(metrics)


def main() -> int:
    ap = argparse.ArgumentParser(description="Train SOAP barrier model + admission gate.")
    ap.add_argument("--trainer", type=Path, default=DEFAULT_TRAINER)
    ap.add_argument("--python-bin", type=Path, default=DEFAULT_VENV)
    ap.add_argument("--data-root", type=str, required=True,
                    help="e.g. soap-KMC (trainer resolves <data-root>/<datasets>)")
    ap.add_argument("--datasets", type=str, default="NEB_new_data",
                    help="comma list, e.g. 'NEB_new_data,NEB_SIA'")
    ap.add_argument("--pair-types", type=str, default="all")
    ap.add_argument("--species", type=str, default="Mo,Re")
    ap.add_argument("--type-symbols", type=str, default="1:Mo,2:Re")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--extra-args", type=str, default="",
                    help="verbatim extra flags for the trainer, e.g. "
                         "'--soap-sparse --neb-features --disp-features'")
    ap.add_argument("--mae-threshold", type=float, default=0.20)
    ap.add_argument("--min-bucket", type=int, default=10)
    ap.add_argument("--gate", choices=["strict", "warn"], default="strict")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.output_dir.exists() and (args.output_dir / "model.joblib").exists():
        if not args.overwrite:
            print(f"❌ {args.output_dir} already has a model; pass --overwrite to retrain",
                  file=sys.stderr)
            return 2
        print(f"⚠️  overwriting existing model in {args.output_dir}")

    cmd = [
        str(args.python_bin), str(args.trainer),
        "--data-root", args.data_root,
        "--datasets", args.datasets,
        "--pair-types", args.pair_types,
        "--species", args.species,
        "--type-symbols", args.type_symbols,
        "--output-dir", str(args.output_dir),
    ] + [tok for tok in args.extra_args.split() if tok]
    print("train cmd:", " ".join(cmd))

    proc = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(proc.stdout[-4000:])
    if proc.returncode != 0:
        print(f"❌ trainer exited {proc.returncode}:\n{proc.stderr[-2000:]}", file=sys.stderr)
        return 1

    metrics_path = args.output_dir / "metrics.json"
    if not metrics_path.exists():
        print(f"❌ trainer finished but {metrics_path} missing", file=sys.stderr)
        return 1
    metrics = json.loads(metrics_path.read_text())
    test = find_test_metrics(metrics)
    if test is None:
        print("❌ no test metrics found in metrics.json", file=sys.stderr)
        return 1
    mae = float(test["mae"])
    verdict = mae <= args.mae_threshold
    print(f"\ngate[{args.gate}]: test MAE {mae:.4f} eV "
          f"{'≤' if verdict else '>'} {args.mae_threshold:.4f} eV")

    report = {
        "mae": mae, "rmse": test.get("rmse"), "r2": test.get("r2"),
        "threshold": args.mae_threshold, "verdict": "pass" if verdict else "fail",
        "note": f"buckets with <{args.min_bucket} samples are ⚠️-only (see trainer stdout)",
    }
    (args.output_dir / "gate_report.json").write_text(json.dumps(report, indent=2))

    if not verdict:
        if args.gate == "strict":
            print("❌ strict gate failed — do NOT deploy this model.", file=sys.stderr)
            return 1
        print("⚠️  warn mode: gate failed but deploy may proceed with ⚠️ tag.")
    print("✅ s4_train complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

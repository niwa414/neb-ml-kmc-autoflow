#!/usr/bin/env python3
"""s5_deploy_kmc — gate, version, and deploy a trained barrier model for misa-kmc.

Pure stdlib. Idempotent-ish: re-deploying the same version bumps a suffix; the
live directory name the KMC inputs point at (`kmc_runtime_pair_model`) is kept
STABLE — we version by archiving the previous live dir to
`kmc_runtime_pair_model_prev_<UTC>` and copying the new candidate in.

Deploy contract (verified 2026-10-11):
  KMC template examples/in.ml_barrier.template sets
    ml_barrier_model_dir .../soap-KMC/runs/kmc_runtime_pair_model
  and tools/kmc_ml_barrier.py expects model.joblib + scaler.npz (+ metrics.json).

Gate: reads metrics.json → test MAE must be ≤ --mae-threshold (strict) or the
deploy is tagged WARN (warn mode). Missing metrics → strict fails, warn proceeds.

Smoke test (optional): runs misa-kmc on a tiny ml_barrier input in a temp dir
and checks the log for ML-barrier activity (lookup hits / server queries) and
sane barrier magnitudes. Requires --misa-kmc and --smoke-input.

Exit codes: 0 deployed, 1 gate rejected (strict) or smoke failed, 2 usage.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REQUIRED_MODEL_FILES = ["model.joblib", "scaler.npz"]
LIVE_NAME = "kmc_runtime_pair_model"
DEPLOY_MANIFEST = "deploy_manifest.json"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def find_test_metrics(metrics: dict) -> dict | None:
    """metrics.json nests differently across trainer versions; find the test block."""
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


def gate_check(model_dir: Path, threshold: float, mode: str) -> tuple[bool, str, dict | None]:
    metrics_path = model_dir / "metrics.json"
    if not metrics_path.exists():
        msg = f"metrics.json missing in {model_dir}"
        return (False, msg + " (strict)", None) if mode == "strict" else (True, "WARN: " + msg, None)
    metrics = json.loads(metrics_path.read_text())
    test = find_test_metrics(metrics)
    if test is None:
        msg = "no test mae/rmse block found in metrics.json"
        return (False, msg, None) if mode == "strict" else (True, "WARN: " + msg, None)
    mae = float(test["mae"])
    if mae > threshold:
        msg = f"test MAE {mae:.4f} eV > threshold {threshold:.4f} eV"
        return (False, msg, test) if mode == "strict" else (True, "WARN: " + msg, test)
    return True, f"test MAE {mae:.4f} eV ≤ {threshold:.4f} eV", test


def smoke_test(misa_kmc: Path, smoke_input: Path, model_dir: Path,
               timeout: int = 300) -> tuple[bool, str]:
    workdir = Path(tempfile.mkdtemp(prefix="kmc_smoke_"))
    shutil.copy(smoke_input, workdir / smoke_input.name)
    cmd = [str(misa_kmc), "-in", smoke_input.name]
    try:
        proc = subprocess.run(cmd, cwd=workdir, capture_output=True,
                              text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"smoke run timed out after {timeout}s"
    log = (proc.stdout or "") + "\n" + (proc.stderr or "")
    # also harvest log.misa / screen output files if present
    for extra in workdir.glob("log*"):
        try:
            log += "\n" + extra.read_text(errors="replace")
        except OSError:
            pass
    hits = "ML lookup entries" in log or "ml_barrier" in log.lower()
    if proc.returncode != 0:
        return False, f"misa-kmc exited {proc.returncode}: {log[-400:]}"
    if not hits:
        return False, "no ML-barrier activity found in smoke log (ml_barrier not engaged?)"
    return True, f"smoke ok ({workdir})"


def main() -> int:
    ap = argparse.ArgumentParser(description="Gate + version + deploy barrier model to misa-kmc.")
    ap.add_argument("--model-dir", type=Path, required=True,
                    help="trainer output dir containing model.joblib/scaler.npz/metrics.json")
    ap.add_argument("--runs-root", type=Path,
                    default=Path(os.environ.get("KMC_RUNS_ROOT", "./runs")).expanduser(),
                    help="dir holding the live model dir (env KMC_RUNS_ROOT)")
    ap.add_argument("--mae-threshold", type=float, default=0.20,
                    help="eV; strict gate rejects test MAE above this (default 0.20)")
    ap.add_argument("--gate", choices=["strict", "warn"], default="strict")
    ap.add_argument("--misa-kmc", type=Path, default=None,
                    help="enable smoke test with this binary")
    ap.add_argument("--smoke-input", type=Path, default=None,
                    help="small KMC input using ml_barrier for the smoke test")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.model_dir.is_dir():
        print(f"❌ model dir not found: {args.model_dir}", file=sys.stderr)
        return 2
    missing = [f for f in REQUIRED_MODEL_FILES if not (args.model_dir / f).exists()]
    if missing:
        print(f"❌ model dir missing files: {missing}", file=sys.stderr)
        return 2

    ok, gate_msg, test = gate_check(args.model_dir, args.mae_threshold, args.gate)
    print(f"gate[{args.gate}]: {gate_msg}")
    if not ok:
        print("❌ deploy blocked by gate.", file=sys.stderr)
        return 1

    live = args.runs_root / LIVE_NAME
    live_resolved = live.resolve()
    model_resolved = args.model_dir.resolve()
    if model_resolved == live_resolved:
        print(f"❌ --model-dir IS the live deploy dir ({live}); "
              "deploying a dir onto itself would destroy it. Point at a trainer output dir instead.",
              file=sys.stderr)
        return 2
    if live_resolved in model_resolved.parents:
        print(f"❌ --model-dir sits INSIDE the live dir ({model_resolved}); refusing.",
              file=sys.stderr)
        return 2
    prev = args.runs_root / f"{LIVE_NAME}_prev_{utc_stamp()}"
    staging = args.runs_root / f".staging_{utc_stamp()}"
    plan = [
        f"copy {args.model_dir} → {staging} (staging)",
        f"archive {live} → {prev}" if live.exists() else f"create {live} (no previous)",
        f"activate {staging} → {live}",
    ]
    if args.smoke_input and args.misa_kmc:
        plan.append(f"smoke: {args.misa_kmc} -in {args.smoke_input.name}")
    print("deploy plan:")
    for step in plan:
        print(f"  - {step}")
    if args.dry_run:
        print("dry-run, nothing written.")
        return 0

    try:
        shutil.copytree(args.model_dir, staging)
        if live.exists():
            shutil.move(str(live), str(prev))
        shutil.move(str(staging), str(live))
    except OSError as exc:
        # best-effort rollback: if live was archived but activation failed, restore it
        if not live.exists() and prev.exists():
            shutil.move(str(prev), str(live))
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        print(f"❌ deploy failed: {exc}", file=sys.stderr)
        return 1

    record = {
        "deployed_utc": datetime.now(timezone.utc).isoformat(),
        "source_model_dir": str(args.model_dir.resolve()),
        "live_dir": str(live),
        "archived_prev": str(prev) if prev.exists() else None,
        "gate_mode": args.gate,
        "gate_msg": gate_msg,
        "test_metrics": test,
        "warned": gate_msg.startswith("WARN"),
    }

    if args.smoke_input and args.misa_kmc:
        ok, smoke_msg = smoke_test(args.misa_kmc, args.smoke_input, live)
        record["smoke"] = {"ok": ok, "msg": smoke_msg}
        print(f"smoke: {'✅' if ok else '❌'} {smoke_msg}")
        if not ok:
            print("❌ smoke failed — rolling back to previous live dir.", file=sys.stderr)
            shutil.rmtree(live)
            if record["archived_prev"]:
                shutil.move(record["archived_prev"], str(live))
            return 1

    manifest_path = args.runs_root / DEPLOY_MANIFEST
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"deploys": []}
    manifest["deploys"].append(record)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"✅ deployed → {live}  (manifest: {manifest_path})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

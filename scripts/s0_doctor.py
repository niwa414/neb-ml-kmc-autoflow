#!/usr/bin/env python3
"""s0_doctor — preflight checks for the neb-ml-kmc-autoflow pipeline.

Pure stdlib. Read-only. Idempotent. Never guesses: every missing item is
reported explicitly with the flag/env var needed to fix it.

Checks (each reports ok / missing / warn):
  1. repo roots          --kmc-ml-root, --mietclaw-root
  2. LAMMPS executable   --lmp-path or $LMP_BIN or PATH (lmp/lmp_serial/lammps)
  3. misa-kmc binaries   <root>/crystalkmc-fix-diffusion-coef/build/bin/misa-kmc
  4. ML venv python      --python-bin (default kmc-ml/.venv/bin/python)
                         + import probe for dscribe/sklearn/joblib/numpy
  5. potential file      --potential
  6. NEB data dirs       NEB_new_data/, NEB_SIA/ under soap-KMC/

Exit codes: 0 = all ok, 1 = at least one missing, 2 = usage error.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Defaults resolve via env vars; everything is overridable by CLI flags.
DEFAULT_KMC_ML_ROOT = Path(os.environ.get("KMC_ML_ROOT", "./kmc-ml")).expanduser()
DEFAULT_MIETCLAW_ROOT = Path(os.environ.get("MIETCLAW_ROOT", "./miet-claw")).expanduser()
DEFAULT_POTENTIAL_NAME = "MoRe.eam.fs"

IMPORT_PROBE = "import dscribe, sklearn, joblib, numpy; print('imports-ok')"
LMP_CANDIDATES = ("lmp", "lmp_serial", "lmp_mpi", "lammps")


def check_root(name: str, root: Path) -> dict:
    markers = {
        "kmc-ml": ["soap-KMC/train_migration_soap.py", "crystalkmc-fix-diffusion-coef"],
        "miet-claw": ["src/miet_claw/neb_runtime.py", "src/miet_claw/bridge.py"],
    }[name]
    if not root.is_dir():
        return {"name": f"{name} root", "status": "missing",
                "detail": f"directory not found: {root}",
                "fix": f"--{name.replace('_', '-')}-root <path>"}
    absent = [m for m in markers if not (root / m).exists()]
    if absent:
        return {"name": f"{name} root", "status": "missing",
                "detail": f"directory exists but markers absent: {absent}",
                "fix": f"verify this is the real {name} checkout"}
    return {"name": f"{name} root", "status": "ok", "detail": str(root)}


def check_lmp(explicit: str | None) -> dict:
    if explicit:
        p = Path(explicit).expanduser()
        if p.is_file() and os.access(p, os.X_OK):
            return {"name": "lammps executable", "status": "ok", "detail": str(p)}
        return {"name": "lammps executable", "status": "missing",
                "detail": f"--lmp-path given but not executable: {p}",
                "fix": "rebuild LAMMPS or point --lmp-path at the real binary"}
    env = os.environ.get("LMP_BIN")
    if env:
        p = Path(env).expanduser()
        if p.is_file() and os.access(p, os.X_OK):
            return {"name": "lammps executable", "status": "ok",
                    "detail": str(p), "source": "$LMP_BIN"}
        return {"name": "lammps executable", "status": "missing",
                "detail": f"$LMP_BIN set but not executable: {p}",
                "fix": "export LMP_BIN=<real lmp path>"}
    for cand in LMP_CANDIDATES:
        found = shutil.which(cand)
        if found:
            return {"name": "lammps executable", "status": "ok",
                    "detail": found, "source": "PATH"}
    return {"name": "lammps executable", "status": "missing",
            "detail": "not on PATH and no --lmp-path / $LMP_BIN",
            "fix": "--lmp-path <path> (e.g. conda env miet-stack lmp binary)"}


def check_misa_kmc(kmc_ml_root: Path, mietclaw_root: Path) -> dict:
    results = []
    for root, tag in ((kmc_ml_root, "kmc-ml"), (mietclaw_root, "miet-claw")):
        p = root / "crystalkmc-fix-diffusion-coef" / "build" / "bin" / "misa-kmc"
        results.append((tag, p, p.is_file() and os.access(p, os.X_OK)))
    ok = [r for r in results if r[2]]
    if not ok:
        return {"name": "misa-kmc binary", "status": "missing",
                "detail": "no executable misa-kmc in either repo "
                          + "; ".join(f"{t}:{p}" for t, p, _ in results),
                "fix": "build misa-kmc (see repo MAKE/ docs) in at least one repo"}
    detail = "; ".join(f"{t}:{p}" for t, p, good in results if good)
    missing = [t for t, _, good in results if not good]
    out = {"name": "misa-kmc binary", "status": "ok", "detail": detail}
    if missing:
        out["status"] = "warn"
        out["detail"] += f" (absent in: {','.join(missing)} — fine if you only run the other)"
    return out


def check_neb_mpi(lmp_path_str: str | None) -> dict:
    """Multi-replica NEB needs lmp_mpi + mpiexec next to lmp (s2 hard requirement)."""
    if not lmp_path_str:
        env = os.environ.get("LMP_BIN")
        lmp_path_str = env or shutil.which("lmp")
    if not lmp_path_str:
        return {"name": "neb mpi toolchain", "status": "missing",
                "detail": "no lmp path to derive lmp_mpi/mpiexec from",
                "fix": "--lmp-path first"}
    parent = Path(lmp_path_str).expanduser().parent
    lmp_mpi = parent / "lmp_mpi"
    mpiexec = parent / "mpiexec"
    absent = [str(p) for p in (lmp_mpi, mpiexec) if not p.is_file()]
    if absent:
        return {"name": "neb mpi toolchain", "status": "missing",
                "detail": f"absent: {absent} (multi-replica NEB will not run)",
                "fix": "install an MPI LAMMPS build with REPLICA package"}
    return {"name": "neb mpi toolchain", "status": "ok",
            "detail": f"{lmp_mpi} + {mpiexec}"}


def check_python_bin(python_bin: Path, kmc_ml_root: Path) -> dict:
    candidate = python_bin if python_bin else kmc_ml_root / ".venv" / "bin" / "python"
    if not candidate.is_file():
        return {"name": "ml venv python", "status": "missing",
                "detail": f"not found: {candidate}",
                "fix": "create kmc-ml/.venv with dscribe scikit-learn joblib numpy"}
    try:
        proc = subprocess.run(
            [str(candidate), "-c", IMPORT_PROBE],
            capture_output=True, text=True, timeout=60,
        )
    except subprocess.TimeoutExpired:
        return {"name": "ml venv python", "status": "missing",
                "detail": f"import probe timed out: {candidate}",
                "fix": "check the venv is functional"}
    if proc.returncode != 0:
        return {"name": "ml venv python", "status": "missing",
                "detail": f"import probe failed in {candidate}: {proc.stderr.strip()[:300]}",
                "fix": f"{candidate} -m pip install dscribe scikit-learn joblib numpy"}
    return {"name": "ml venv python", "status": "ok", "detail": str(candidate)}


def check_potential(path: Path) -> dict:
    if path.is_file():
        return {"name": "potential file", "status": "ok", "detail": str(path)}
    return {"name": "potential file", "status": "missing",
            "detail": f"not found: {path}",
            "fix": "--potential <path> (for MoReX pass the ternary EAM file)"}


def check_data_dirs(kmc_ml_root: Path) -> dict:
    soap = kmc_ml_root / "soap-KMC"
    dirs = ["NEB_new_data", "NEB_SIA"]
    absent = [d for d in dirs if not (soap / d).is_dir()]
    if absent:
        return {"name": "neb data dirs", "status": "missing",
                "detail": f"absent under {soap}: {absent}",
                "fix": "restore NEB datasets before ingest/training stages"}
    return {"name": "neb data dirs", "status": "ok", "detail": str(soap)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Preflight doctor for neb-ml-kmc-autoflow.")
    ap.add_argument("--kmc-ml-root", type=Path, default=DEFAULT_KMC_ML_ROOT)
    ap.add_argument("--mietclaw-root", type=Path, default=DEFAULT_MIETCLAW_ROOT)
    ap.add_argument("--lmp-path", type=str, default=None,
                    help="explicit LAMMPS executable path (never guessed)")
    ap.add_argument("--potential", type=Path, default=None,
                    help="EAM potential file; default <kmc-ml-root>/" + DEFAULT_POTENTIAL_NAME)
    ap.add_argument("--python-bin", type=Path, default=None,
                    help="ML venv python; default <kmc-ml-root>/.venv/bin/python")
    args = ap.parse_args()

    potential = args.potential or (args.kmc_ml_root / DEFAULT_POTENTIAL_NAME)

    checks = [
        check_root("kmc-ml", args.kmc_ml_root),
        check_root("miet-claw", args.mietclaw_root),
        check_lmp(args.lmp_path),
        check_neb_mpi(args.lmp_path),
        check_misa_kmc(args.kmc_ml_root, args.mietclaw_root),
        check_python_bin(args.python_bin, args.kmc_ml_root),
        check_potential(potential),
        check_data_dirs(args.kmc_ml_root),
    ]

    summary = {"ok": 0, "warn": 0, "missing": 0}
    for c in checks:
        summary[c["status"]] += 1
    report = {"stage": "s0_doctor", "checks": checks, "summary": summary}
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if summary["missing"]:
        print("\n❌ doctor: missing items must be fixed before the pipeline runs.",
              file=sys.stderr)
        return 1
    if summary["warn"]:
        print("\n⚠️  doctor: warnings present, pipeline may proceed.", file=sys.stderr)
    else:
        print("\n✅ doctor: all checks passed.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

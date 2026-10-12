#!/usr/bin/env python3
"""Self-contained demo: dumbbell NEB on a tiny FCC Ar (LJ) system.

Validates the pipeline mechanics (structure generation → endpoint relax →
CI-NEB → energy parsing) end-to-end without any external files — no EAM
potential, no private datasets. The resulting barrier is a real LAMMPS
calculation on this toy system; it is NOT a materials result.

Usage:
    python run_demo.py [--lmp PATH]

Requires: LAMMPS with REPLICA package (lmp + lmp_mpi + mpiexec).
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

from structure_gen import generate_dumbbell_sample  # noqa: E402
from s2_run_neb import run_one, parse_energies, write_neb_txt  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmp", type=Path, default=None,
                    help="path to lmp binary (serial relax uses it; NEB uses "
                         "lmp_mpi + mpiexec next to it)")
    args = ap.parse_args()

    lmp = args.lmp
    if lmp is None:
        import shutil
        found = shutil.which("lmp") or shutil.which("lmp_mpi")
        lmp = Path(found) if found else None
    if lmp is None or not Path(lmp).is_file():
        print("❌ lmp not found on PATH; pass --lmp <path>", file=sys.stderr)
        return 2
    lmp = Path(lmp)

    # --- 1. Structure: 4x4x4 FCC Ar + center dumbbell along [111] ---
    initial, final, meta = generate_dumbbell_sample(
        species=["Ar"], fractions={}, pair_types=(1, 1),
        n_cells=(4, 4, 4), lattice=5.26, lattice_type="fcc",
        center_cells=(2, 2, 2),
        direction=(1, 1, 1), disp=1.2, seed=42)
    workdir = Path(tempfile.mkdtemp(prefix="lj_demo_"))
    (workdir / "data.lmp").write_text(initial)
    (workdir / "data_final.lmp").write_text(final)
    print(f"structure: {meta['n_atoms']} atoms, workdir={workdir}")

    # --- 2. NEB via the same code path s2 uses (task dict with LJ potential) ---
    task = {
        "id": "lj_ar_demo", "species": ["Ar"],
        "pair_style": "lj/cut 6.5",
        "pair_coeff_line": "* * 0.05 2.5",
        "masses_block": "mass 1 39.948\n",
        "potential": "",  # unused for lj/cut
        "workdir": str(workdir),
    }
    ok, msg = run_one(task, lmp, n_images=5, timeout=600, dry_run=False)
    print(f"NEB: {'✅' if ok else '❌'} {msg}")
    if not ok:
        return 1

    neb_txt = (workdir / "neb.txt").read_text()
    print("--- neb.txt ---")
    print(neb_txt)
    barrier = task.get("barrier_ev")
    print(f"demo barrier = {barrier:.4f} eV  (toy LJ system — pipeline check only)")
    print("\nPipeline mechanics validated. For real alloys, wire your own EAM "
          "potential and train_migration_soap-style trainer (see README).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

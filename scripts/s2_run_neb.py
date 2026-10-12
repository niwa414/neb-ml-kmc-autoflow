#!/usr/bin/env python3
"""s2_run_neb — execute pending NEB tasks from campaign.json (local LAMMPS mode).

Pure stdlib. Resumable: only pending/failed-with-strikes<2 tasks are picked up.
Two-strike discipline: a task failing twice is left in 'failed' state and the
run stops with a report (user's loop discipline).

SKELETON→REAL (2026-10-12): full local NEB flow now mirrors the legacy
calc/ pipeline — endpoint relax (in.rlxfi) → final.mosia extraction
(relaxda2final.sh port) → CI-NEB (in.neb.mosia alignment, partition n_img×1)
→ neb.txt. SLURM backend is still a TODO stub.

neb.txt format emitted: '#reaction_crodinate de' + lines '<rc> <dE>' with
endpoints 0/1 (same contract as the legacy neb.sh output).

Exit codes: 0 all tasks done (or none pending), 1 some tasks failed/two-strike hit.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

CAMPAIGN_FILE = "campaign.json"
NEB_HEADER = "#reaction_crodinate de\n"
MAX_STRIKES = 2


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_campaign(path: Path) -> dict:
    return json.loads(path.read_text())


def save_campaign(path: Path, campaign: dict) -> None:
    campaign["updated_utc"] = utc_now()
    path.write_text(json.dumps(campaign, indent=2))


def _pair_block(task: dict) -> str:
    """pair_style + pair_coeff + masses block. Default: EAM/fs with a potential
    file; set task['pair_style']/task['pair_coeff_line']/task['masses_block']
    for other potentials (e.g. the LJ demo)."""
    pair_style = task.get("pair_style", "eam/fs")
    if pair_style == "eam/fs":
        species = task["species"]
        masses = "\n".join(f"mass            {i+1} 95.95" for i in range(len(species)))
        return (f"pair_style      eam/fs\n"
                f"pair_coeff      * * {task['potential']} {' '.join(species)}\n{masses}")
    pair_coeff = task.get("pair_coeff_line", "")
    masses = task.get("masses_block", "")
    block = f"pair_style      {pair_style}\n"
    if pair_coeff:
        block += f"pair_coeff      {pair_coeff}\n"
    if masses:
        block += masses
    return block


def render_relax_input(task: dict, endpoint: str) -> str:
    data_file = "data_final.lmp" if endpoint == "final" else "data.lmp"
    return f"""# endpoint relax ({endpoint}) for {task['id']} — s2_run_neb local template
units           metal
dimension       3
atom_style      atomic
boundary        p p p
atom_modify     sort 0 0.0
neighbor        3 bin
neigh_modify    delay 5
read_data       {data_file}
{_pair_block(task)}
min_style       cg
minimize        1.0e-10 1.0e-10 100000 100000
write_data      relaxed_{endpoint}.lmp nocoeff
"""


def render_neb_input(task: dict, n_replicas: int) -> str:
    """CI-NEB input mirroring legacy calc/in.neb.mosia: relax initial in-place,
    fix neb, read relaxed-final coords from final.mosia, per-replica dumps."""
    return f"""# CI-NEB for {task['id']} — aligned with legacy NEB_new_data/calc/in.neb.mosia
units           metal
dimension       3
atom_style      atomic
boundary        p p p
atom_modify     map array
atom_modify     sort 0 0.0
neighbor        3 bin
neigh_modify    delay 5
read_data       relaxed_initial.lmp
{_pair_block(task)}
variable        u uloop 20
min_style       cg
minimize        1.0e-10 1.0e-10 100000 100000
reset_timestep  0
fix             1 all neb 1.0
thermo          100
timestep        0.001
min_style       quickmin
neb             1e-12 1e-12 10000 10000 200 final final.mosia
dump            1 all atom 1 dump.neb.$u
run             0
"""


def write_mosia(relaxed_data: Path, out_path: Path) -> None:
    """stdlib port of legacy relaxda2final.sh: '<n_atoms>' then 'id x y z'."""
    lines = relaxed_data.read_text().splitlines()
    atoms, in_atoms = [], False
    for line in lines:
        s = line.strip()
        if s.lower().startswith("atoms"):
            in_atoms = True
            continue
        if in_atoms:
            if not s:
                continue
            parts = s.split()
            if len(parts) >= 5 and parts[0].lstrip("-").isdigit():
                atoms.append(f"{parts[0]} {float(parts[2]):.6f} {float(parts[3]):.6f} {float(parts[4]):.6f}")
            elif atoms:
                break
    if not atoms:
        raise ValueError(f"no atoms parsed from {relaxed_data}")
    out_path.write_text(f"{len(atoms)}\n" + "\n".join(atoms) + "\n")


def parse_energies(log_dir: Path) -> list[float] | None:
    """Harvest per-replica PE from log.lammps.<i> files.

    LAMMPS prints a block after each minimize/neb stage:
        Energy initial, next-to-last, final =
           <E0>  <E1>  <E2>
        Force two-norm initial, final = ...
    The barrier uses the 'next-to-last' (converged) energy — same convention as
    legacy neb.sh. We take the LAST such block in each replica log (the NEB-stage
    converged value), matching `grep -A3 'next-to-last' | tail -3 | head -1`."""
    energies = []
    for i in range(100):
        logf = log_dir / f"log.lammps.{i}"
        if not logf.exists():
            if i == 0:
                continue
            break
        e = None
        lines = logf.read_text(errors="replace").splitlines()
        for idx, line in enumerate(lines):
            if "next-to-last" in line:
                # energy row: <E_initial> <E_next-to-last> <E_final>
                for j in range(idx + 1, min(idx + 4, len(lines))):
                    parts = lines[j].split()
                    if len(parts) >= 3:
                        try:
                            e = float(parts[2])  # final column (legacy neb.sh $3)
                            break
                        except ValueError:
                            continue
        if e is None:
            return None
        energies.append(e)
    return energies if energies else None


def write_neb_txt(workdir: Path, energies: list[float]) -> float:
    e0 = energies[0]
    n = len(energies) - 1
    lines = [NEB_HEADER]
    for i, e in enumerate(energies):
        rc = i / n if n else 0.0
        lines.append(f"{rc:.3f} {e - e0:.6f}\n")
    (workdir / "neb.txt").write_text("".join(lines))
    return max(e - e0 for e in energies)


def run_one(task: dict, lmp_path: Path, n_images: int, timeout: int,
            dry_run: bool) -> tuple[bool, str]:
    wd = Path(task["workdir"])
    for need in ("data.lmp", "data_final.lmp"):
        if not (wd / need).is_file():
            return False, f"missing {wd}/{need}"
    if dry_run:
        return True, "dry-run (skipped lammps)"

    # Stage 1: endpoint relax (initial + final), mirroring legacy in.rlxfi.
    for endpoint in ("initial", "final"):
        (wd / f"in.relax.{endpoint}").write_text(render_relax_input(task, endpoint))
        proc = subprocess.run(
            [str(lmp_path), "-in", f"in.relax.{endpoint}"],
            cwd=wd, capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            return False, f"relax {endpoint} failed: {(proc.stderr or proc.stdout)[-200:]}"
        if not (wd / f"relaxed_{endpoint}.lmp").is_file():
            return False, f"relax {endpoint} produced no relaxed_{endpoint}.lmp"

    # Stage 2: final.mosia from relaxed final (stdlib port of relaxda2final.sh).
    try:
        write_mosia(wd / "relaxed_final.lmp", wd / "final.mosia")
    except ValueError as exc:
        return False, f"mosia extraction failed: {exc}"

    # Stage 3: CI-NEB. Multi-replica NEB needs MPI ranks == partitions:
    # mpiexec -np <n> lmp_mpi -partition <n>x1 -in in.neb
    (wd / "in.neb").write_text(render_neb_input(task, n_images))
    lmp_mpi = lmp_path.parent / ("lmp_mpi" if lmp_path.name != "lmp_mpi" else "lmp_mpi")
    mpiexec = lmp_path.parent / "mpiexec"
    if lmp_mpi.is_file() and mpiexec.is_file():
        neb_cmd = [str(mpiexec), "--oversubscribe", "-np", str(n_images),
                   str(lmp_mpi), "-partition", f"{n_images}x1", "-in", "in.neb"]
    else:
        # single-process fallback: lmp can run NEB in serial with 1 replica only
        # when compiled appropriately; most builds refuse — report, don't guess.
        return False, (f"NEB needs mpiexec + lmp_mpi next to lmp "
                       f"(looked in {lmp_path.parent}); serial fallback not attempted")
    proc = subprocess.run(neb_cmd, cwd=wd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        return False, f"neb run failed: {(proc.stderr or proc.stdout)[-300:]}"

    energies = parse_energies(wd)
    if energies is None:
        return False, "no per-replica energies parsed from log.lammps.*"
    if len(energies) != n_images:
        return False, (f"replica count mismatch: parsed {len(energies)} energies "
                       f"but campaign requested {n_images} replicas — refusing to "
                       "accept silently (check mpiexec slots / partition config)")
    barrier = write_neb_txt(wd, energies)
    task["barrier_ev"] = round(barrier, 6)
    return True, f"barrier={barrier:.4f} eV ({len(energies)} replicas)"


def main() -> int:
    ap = argparse.ArgumentParser(description="Run pending NEB tasks (local lmp) with two-strike discipline.")
    ap.add_argument("--campaign", type=Path, required=True)
    ap.add_argument("--lmp-path", type=Path, required=True)
    ap.add_argument("--n-images", type=int, default=5,
                    help="NEB replicas = mpiexec ranks (default 5, matches legacy "
                         "neb.sh rc=i/4 convention)")
    ap.add_argument("--timeout", type=int, default=3600, help="per-invocation seconds")
    ap.add_argument("--max-tasks", type=int, default=0, help="0 = all pending")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.lmp_path.is_file():
        print(f"❌ lmp not found: {args.lmp_path}", file=sys.stderr)
        return 1
    campaign = load_campaign(args.campaign)
    tasks = campaign.get("tasks", [])
    pending = [t for t in tasks if t["state"] == "pending"
               or (t["state"] == "failed" and t.get("strikes", 0) < MAX_STRIKES)]
    if args.max_tasks:
        pending = pending[:args.max_tasks]
    print(f"s2_run_neb: {len(pending)} tasks to attempt")

    failures = []
    for t in pending:
        t["state"] = "running"
        t.setdefault("attempts", []).append({"utc": utc_now(), "mode": "local"})
        save_campaign(args.campaign, campaign)
        started = time.time()
        ok, msg = run_one(t, args.lmp_path, args.n_images, args.timeout, args.dry_run)
        t["attempts"][-1].update({"ok": ok, "msg": msg,
                                  "seconds": round(time.time() - started, 1)})
        if ok:
            t["state"] = "done"
            t["strikes"] = 0
        else:
            t["strikes"] = t.get("strikes", 0) + 1
            t["state"] = "failed"
            failures.append((t["id"], t["strikes"], msg))
        save_campaign(args.campaign, campaign)
        print(f"  {'✅' if ok else '❌'} {t['id']}: {msg}")

    if failures:
        two_strike = [f for f in failures if f[1] >= MAX_STRIKES]
        print(f"\n⚠️  {len(failures)} failures; {len(two_strike)} hit two-strike:", file=sys.stderr)
        for tid, strikes, msg in two_strike:
            print(f"   ❌ {tid} (strikes={strikes}): {msg}", file=sys.stderr)
        if two_strike:
            print("stopping per two-strike discipline — report to user.", file=sys.stderr)
        return 1
    print("✅ all attempted tasks done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""lammps_data.py — tiny stdlib reader for LAMMPS data files (atom types only).

Only supports the 'atomic' atom style columns (id type x y z), which is what
the kmc-ml MoRe data.lmp files use (units metal, atom_style atomic).
"""
from __future__ import annotations

from pathlib import Path


def read_atom_types(data_lmp: Path) -> list[int]:
    """Return per-atom integer types ordered by atom id from a LAMMPS data file."""
    lines = data_lmp.read_text().splitlines()
    atoms_start = None
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("atoms"):
            atoms_start = i + 1
            break
    if atoms_start is None:
        raise ValueError(f"no 'Atoms' section in {data_lmp}")
    # skip blank line / comment line after the section header
    entries: dict[int, int] = {}
    for line in lines[atoms_start:]:
        s = line.strip()
        if not s or s.startswith("#"):
            if entries:
                break  # section ended
            continue
        parts = s.split()
        if len(parts) < 3:
            if entries:
                break
            continue
        try:
            atom_id = int(parts[0])
            atom_type = int(float(parts[1]))
        except ValueError:
            # hit a non-data line (next section header like 'Velocities')
            if entries:
                break
            continue
        entries[atom_id] = atom_type
    if not entries:
        raise ValueError(f"no atom rows parsed in {data_lmp}")
    return [entries[k] for k in sorted(entries)]


def read_box(data_lmp: Path) -> list[float]:
    """Return [xlen, ylen, zlen] from the box header lines."""
    lo = [0.0, 0.0, 0.0]
    hi = [0.0, 0.0, 0.0]
    for line in data_lmp.read_text().splitlines():
        s = line.strip()
        for axis, tag in ((0, "xlo xhi"), (1, "ylo yhi"), (2, "zlo zhi")):
            if s.endswith(tag):
                parts = s.split()
                lo[axis], hi[axis] = float(parts[0]), float(parts[1])
    return [hi[i] - lo[i] for i in range(3)]

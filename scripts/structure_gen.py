#!/usr/bin/env python3
"""structure_gen.py — endpoint structure generator for dumbbell-pair NEB samples.

Pure stdlib (random + math). Ported from kmc-ml soap-KMC/NEB_new_data/generate/
Mo_Re_alloy_MoRe.py and generalized to N species.

Endpoint construction rule (verified against real sample
NEB_new_data/MoRe/Re_0.05/model_1/{data.lmp,data_final.lmp}, 2026-10-12):

  initial: remove atom nearest center c, add dumbbell at c ± dir·disp
           (the two appended atoms get the largest ids — 2000/2001 at 10³ cells)
  target:  lattice atom nearest c + dir·(a/2)   (BCC [111] half-step)
  final:   dumbbell near atom  → target_site − dir·disp
           target lattice atom → target_site + dir·disp
           dumbbell far atom   → c (becomes lattice atom)

Masses: IUPAC standard weights, keyed by element symbol.
"""
from __future__ import annotations

import math
import random

MASSES = {
    "Mo": 95.94, "Re": 186.207, "Ga": 69.723, "W": 183.84,
    "Ta": 180.948, "Nb": 92.906, "V": 50.942, "Fe": 55.845,
}


def _bcc_sites(n_cells: tuple[int, int, int], a: float) -> list[list[float]]:
    nx, ny, nz = n_cells
    sites = []
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                sites.append([x * a, y * a, z * a])
                sites.append([(x + 0.5) * a, (y + 0.5) * a, (z + 0.5) * a])
    return sites


def _fcc_sites(n_cells: tuple[int, int, int], a: float) -> list[list[float]]:
    nx, ny, nz = n_cells
    sites = []
    basis = [(0.0, 0.0, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)]
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                for bx, by, bz in basis:
                    sites.append([(x + bx) * a, (y + by) * a, (z + bz) * a])
    return sites


def _nearest(positions: list[list[float]], point: list[float]) -> int:
    best, best_d = -1, float("inf")
    for i, p in enumerate(positions):
        d = (p[0] - point[0]) ** 2 + (p[1] - point[1]) ** 2 + (p[2] - point[2]) ** 2
        if d < best_d:
            best, best_d = i, d
    return best


def _unit(vec: tuple[float, float, float]) -> list[float]:
    n = math.sqrt(sum(v * v for v in vec))
    return [v / n for v in vec]


def render_lammps_data(positions: list[list[float]], types: list[int],
                       species: list[str], box: tuple[float, float, float],
                       title: str) -> str:
    out = [f"{title}\n\n",
           f"{len(positions)} atoms\n",
           f"{len(species)} atom types\n\n",
           f"0.0 {box[0]:.6f} xlo xhi\n",
           f"0.0 {box[1]:.6f} ylo yhi\n",
           f"0.0 {box[2]:.6f} zlo zhi\n\n",
           "Masses\n\n"]
    for i, sym in enumerate(species, start=1):
        out.append(f"{i} {MASSES.get(sym, 100.0)}\n")
    out.append("\nAtoms\n\n")
    for i, (t, p) in enumerate(zip(types, positions), start=1):
        out.append(f"{i} {t} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
    return "".join(out)


def generate_dumbbell_sample(
    species: list[str],
    fractions: dict[int, float],
    pair_types: tuple[int, int],
    n_cells: tuple[int, int, int] = (10, 10, 10),
    lattice: float = 3.147,
    lattice_type: str = "bcc",
    center_cells: tuple[int, int, int] = (5, 5, 5),
    direction: tuple[float, float, float] = (1, 1, 1),
    disp: float = 0.3,
    seed: int = 12345,
) -> tuple[str, str, dict]:
    """Return (initial_data, final_data, meta). meta carries the pivot atom ids
    (pair_near, pair_far, target) needed for the combined_data.csv index row."""
    rng = random.Random(seed)
    nx, ny, nz = n_cells
    if lattice_type == "bcc":
        positions = _bcc_sites(n_cells, lattice)
    elif lattice_type == "fcc":
        positions = _fcc_sites(n_cells, lattice)
    else:
        raise ValueError(f"lattice_type must be 'bcc' or 'fcc', got {lattice_type!r}")
    types = [1] * len(positions)

    # random substitution: type1 is the remainder species
    indices = list(range(len(positions)))
    for t_idx in sorted(fractions):
        frac = fractions[t_idx]
        if t_idx == 1 or frac <= 0:
            continue
        n_sub = int(len(positions) * frac)
        chosen = rng.sample(indices, n_sub)
        for i in chosen:
            types[i] = t_idx

    c = [center_cells[0] * lattice, center_cells[1] * lattice, center_cells[2] * lattice]
    u = _unit(direction)

    # initial: remove center atom, add dumbbell c ± u·disp
    del_idx = _nearest(positions, c)
    del positions[del_idx]
    del types[del_idx]
    near = [c[k] + u[k] * disp for k in range(3)]
    far = [c[k] - u[k] * disp for k in range(3)]
    positions.append(near)
    types.append(pair_types[0])
    positions.append(far)
    types.append(pair_types[1])
    n_atoms = len(positions)
    id_near, id_far = n_atoms - 1, n_atoms  # appended last two (1-based ids)

    # target lattice site: nearest to c + u·(a/2)
    t_site = [c[k] + u[k] * (lattice / 2.0) for k in range(3)]
    tgt_idx = _nearest(positions[: n_atoms - 2], t_site)  # exclude dumbbell
    id_target = tgt_idx + 1
    t_exact = positions[tgt_idx]

    # final positions
    fin_pos = [p[:] for p in positions]
    fin_types = types[:]
    fin_pos[id_near - 1] = [t_exact[k] - u[k] * disp for k in range(3)]
    fin_pos[id_target - 1] = [t_exact[k] + u[k] * disp for k in range(3)]
    fin_pos[id_far - 1] = c[:]

    box = (nx * lattice, ny * lattice, nz * lattice)
    title_i = (f"initial dumbbell {pair_types} in "
               f"{'-'.join(species)} (gen by neb-ml-kmc-autoflow)")
    title_f = "final " + title_i[len("initial "):]
    initial = render_lammps_data(positions, types, species, box, title_i)
    final = render_lammps_data(fin_pos, fin_types, species, box, title_f)
    meta = {
        "pair_near_id": id_near, "pair_far_id": id_far, "target_id": id_target,
        "pair_near_type": pair_types[0], "pair_far_type": pair_types[1],
        "target_type": types[id_target - 1],
        "n_atoms": n_atoms, "lattice": lattice, "direction": list(direction),
        "disp": disp, "seed": seed,
    }
    return initial, final, meta

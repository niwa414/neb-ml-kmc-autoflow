#!/usr/bin/env python3
"""s1_draft_neb — draft an NEB campaign: generate task workspaces + campaign.json.

Pure stdlib. Idempotent: existing campaign.json is merged — new tasks are
appended, existing tasks keep their state. Never overwrites a done task.

Structures are REAL (structure_gen.generate_dumbbell_sample — BCC alloy +
center dumbbell pair + [111] half-step final, verified against the legacy
Mo_Re_alloy_MoRe.py generator and real model_1 sample, 2026-10-12).

campaign.json task schema:
  {"id": "MoMo_Re_0.05_0007", "pair_type": "MoMo", "conc": "Re_0.05",
   "kind": "vacancy", "species": ["Mo","Re"], "potential": "<abs path>",
   "state": "pending|running|done|failed", "strikes": 0,
   "workdir": "<abs path>", "barrier_ev": null, "attempts": []}

Exit codes: 0 drafted, 2 usage.
"""
from __future__ import annotations

import argparse
import json
import sys
import zlib
from datetime import datetime, timezone
from pathlib import Path

from structure_gen import generate_dumbbell_sample

CAMPAIGN_FILE = "campaign.json"


def parse_pair_label(pair: str, species: list[str]) -> tuple[int, int]:
    """'MoGa' with species ['Mo','Re','Ga'] -> (1,3). Greedy longest match."""
    def match_one(s: str) -> tuple[int, str]:
        for i, sym in sorted(enumerate(species, start=1), key=lambda x: -len(x[1])):
            if s.startswith(sym):
                return i, s[len(sym):]
        raise ValueError(f"cannot parse pair label '{pair}' against species {species}")
    a, rest = match_one(pair)
    b, rest2 = match_one(rest)
    if rest2:
        raise ValueError(f"trailing junk in pair label '{pair}'")
    return (a, b)


def pair_label_from_types(pair_types: tuple[int, int], species: list[str]) -> str:
    return species[pair_types[0] - 1] + species[pair_types[1] - 1]


def make_sample_structures(task: dict, seed: int) -> dict:
    """Real structure generation via structure_gen (replaces the old STUB)."""
    species = task["species"]
    pair_types = parse_pair_label(task["pair_type"], species)
    frac = {}
    conc = task.get("conc", "")
    # concentration tags look like Re_0.05 / Ga_0.02 — map symbol prefix to type idx
    if "_" in conc:
        sym, val = conc.rsplit("_", 1)
        try:
            t_idx = species.index(sym) + 1
            frac[t_idx] = float(val)
        except (ValueError, IndexError):
            pass
    initial, final, meta = generate_dumbbell_sample(
        species=species, fractions=frac, pair_types=pair_types,
        lattice=task.get("lattice", 3.147),
        lattice_type=task.get("lattice_type", "bcc"),
        disp=task.get("disp", 0.3), seed=seed)
    wd = Path(task["workdir"])
    (wd / "data.lmp").write_text(initial)
    (wd / "data_final.lmp").write_text(final)
    return meta


def pair_matrix(n_species: int, species: list[str]) -> list[str]:
    if n_species == 2:
        return [f"{a}{b}" for a in species for b in species]
    return [f"{a}{b}" for a in species for b in species]


def draft_tasks(species: list[str], concs: list[str], kinds: list[str],
                per_cell: int, potential: Path, work_root: Path) -> list[dict]:
    tasks = []
    pairs = pair_matrix(len(species), species)
    for kind in kinds:
        for pair in pairs:
            for conc in concs:
                cell_dir = work_root / kind / pair / conc
                start = 1
                existing = sorted(p for p in cell_dir.glob("sample_*") if p.is_dir()) if cell_dir.exists() else []
                if existing:
                    start = max(int(p.name.split("_")[1]) for p in existing) + 1
                for i in range(start, start + per_cell):
                    sid = f"{pair}_{conc}_{i:04d}"
                    tasks.append({
                        "id": f"{kind[:3]}_{sid}",
                        "pair_type": pair, "conc": conc, "kind": kind,
                        "species": species,
                        "potential": str(potential),
                        "state": "pending", "strikes": 0,
                        "workdir": str(cell_dir / f"sample_{i:04d}"),
                        "barrier_ev": None, "attempts": [],
                    })
    return tasks


def main() -> int:
    ap = argparse.ArgumentParser(description="Draft NEB campaign workspaces + campaign.json.")
    ap.add_argument("--species", default="Mo,Re",
                    help="chemistry in atom-type order; ternary e.g. 'Mo,Re,Ga'")
    ap.add_argument("--concs", default="Re_0.05",
                    help="comma concentration tags, e.g. 'Re_0.05,Re_0.10'")
    ap.add_argument("--kinds", default="vacancy",
                    help="comma kinds: vacancy,sia (cluster later)")
    ap.add_argument("--per-cell", type=int, default=5,
                    help="new samples per (kind,pair,conc) cell")
    ap.add_argument("--potential", type=Path, required=True,
                    help="EAM potential file (ternary for MoReX)")
    ap.add_argument("--work-root", type=Path, required=True,
                    help="campaign workspace root (e.g. runs/neb_campaign_v1)")
    ap.add_argument("--campaign", type=Path, default=None,
                    help=f"campaign.json path (default <work-root>/{CAMPAIGN_FILE})")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    species = [s.strip() for s in args.species.split(",") if s.strip()]
    if len(species) < 2:
        print("❌ need ≥2 species", file=sys.stderr)
        return 2
    concs = [c.strip() for c in args.concs.split(",") if c.strip()]
    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    if not args.potential.is_file():
        print(f"❌ potential not found: {args.potential}", file=sys.stderr)
        return 2

    campaign_path = args.campaign or (args.work_root / CAMPAIGN_FILE)
    existing: dict[str, dict] = {}
    if campaign_path.exists():
        old = json.loads(campaign_path.read_text())
        for t in old.get("tasks", []):
            existing[t["id"]] = t

    new_tasks = draft_tasks(species, concs, kinds, args.per_cell,
                            args.potential.resolve(), args.work_root.resolve())
    merged = dict(existing)
    added = 0
    for t in new_tasks:
        if t["id"] not in merged:
            merged[t["id"]] = t
            added += 1
        # never touch existing states — idempotent merge

    plan_tasks = [t for t in merged.values() if t["state"] == "pending"]
    print(f"draft: {added} new tasks added, {len(plan_tasks)} pending total, "
          f"{len(merged)} campaign size")

    if args.dry_run:
        for t in plan_tasks[:5]:
            print(f"  would create: {t['id']} → {t['workdir']}")
        if len(plan_tasks) > 5:
            print(f"  ... and {len(plan_tasks) - 5} more")
        print("dry-run, nothing written.")
        return 0

    for t in plan_tasks:
        wd = Path(t["workdir"])
        wd.mkdir(parents=True, exist_ok=True)
        meta = {}
        if not (wd / "data.lmp").exists():
            # crc32: stable across processes (hash() is per-process randomized)
            seed = zlib.crc32(t["id"].encode()) % (2**31)
            meta = make_sample_structures(t, seed)
        (wd / "meta.json").write_text(json.dumps({
            "id": t["id"], "pair_type": t["pair_type"], "conc": t["conc"],
            "kind": t["kind"], "species": t["species"],
            "potential": t["potential"], "drafted_utc": datetime.now(timezone.utc).isoformat(),
            **({"structure_meta": meta} if meta else {}),
        }, indent=2))

    campaign_path.parent.mkdir(parents=True, exist_ok=True)
    campaign_path.write_text(json.dumps({
        "version": int(json.loads(campaign_path.read_text()).get("version", 0)) + 1
        if campaign_path.exists() else 1,
        "species": species, "updated_utc": datetime.now(timezone.utc).isoformat(),
        "tasks": sorted(merged.values(), key=lambda x: x["id"]),
    }, indent=2))
    print(f"✅ campaign written: {campaign_path}")
    print("structures: real dumbbell endpoints via structure_gen (data.lmp + data_final.lmp).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""s3_ingest — QC finished NEB samples and ingest them into the training layout.

Pure stdlib. Idempotent: re-running on the same source skips already-ingested
samples (matched by source path recorded in the manifest).

Data contract (verified against train_migration_fnn.py:load_samples, 2026-10-11):
  vacancy → <soap-root>/NEB_new_data/<pair>/Re_<conc>/model_<N>/
             index: <pair>/combined_data.csv  (directory,atom_2000_id/type,
                    atom_2001_id/type,atom_1111_id/type,max_value)
  sia     → <soap-root>/NEB_SIA/<pair>/configs/N<count>/N<idx>/
             index: <pair>/configs/combined_data.csv (directory,gap1_id/type,
                    gap2_id/type,target_id/type,max_value)
  every index dir also holds all_atom_types.csv:
             <directory>,directory,<t1>,<t2>,... (per-atom types)

Each sample dir must contain: data.lmp (initial), neb.txt (rc dE curve),
optionally data_final.lmp + dump.neb.* images. max_value := max(dE).

Exit codes: 0 = ingested (or nothing to do), 1 = QC failures blocked ingest,
2 = usage error.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from lammps_data import read_atom_types

REQUIRED_SAMPLE_FILES = ["data.lmp", "neb.txt"]
VAC_HEADER = ("directory,atom_2000_id,atom_2000_type,atom_2001_id,"
              "atom_2001_type,atom_1111_id,atom_1111_type,max_value\n")
SIA_HEADER = ("directory,gap1_id,gap1_type,gap2_id,gap2_type,"
              "target_id,target_type,max_value\n")


def parse_neb_txt(path: Path) -> tuple[list[float], list[float]]:
    rc, de = [], []
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 2:
            continue
        try:
            rc.append(float(parts[0]))
            de.append(float(parts[1]))
        except ValueError:
            continue
    return rc, de


def qc_sample(sample_dir: Path, endpoint_tol: float, min_images: int) -> tuple[bool, str, float]:
    """Return (ok, reason, barrier). barrier = max(dE) from neb.txt."""
    for name in REQUIRED_SAMPLE_FILES:
        if not (sample_dir / name).is_file():
            return False, f"missing required file: {name}", 0.0
    rc, de = parse_neb_txt(sample_dir / "neb.txt")
    if len(rc) < min_images:
        return False, f"neb.txt has {len(rc)} points (< {min_images})", 0.0
    if abs(rc[0]) > 1e-6 or abs(rc[-1] - 1.0) > 1e-6:
        return False, f"reaction coordinate must span 0..1, got {rc[0]}..{rc[-1]}", 0.0
    if abs(de[0]) > endpoint_tol:
        return False, f"initial endpoint |dE|={abs(de[0]):.4f} > {endpoint_tol}", 0.0
    if abs(de[-1]) > endpoint_tol:
        return False, f"final endpoint |dE|={abs(de[-1]):.4f} > {endpoint_tol}", 0.0
    barrier = max(de)
    if barrier <= 0:
        return False, f"non-positive barrier {barrier:.4f}", 0.0
    return True, "ok", barrier


def read_all_atom_types(types_csv: Path) -> dict[str, list[str]]:
    """Tolerant reader matching train_migration_fnn.load_all_atom_types:
    skips a bare 'directory' header line; accepts both
    '<dir>,directory,t1,t2,...' (vacancy) and '<dir>,t1,t2,...' (sia) layouts."""
    mapping: dict[str, list[str]] = {}
    if not types_csv.exists():
        return mapping
    lines = [l.strip() for l in types_csv.read_text().splitlines() if l.strip()]
    if lines and lines[0] == "directory":
        lines = lines[1:]
    for line in lines:
        parts = [p for p in line.split(",") if p != ""]
        if len(parts) < 2:
            continue
        directory = parts[0]
        start = 2 if parts[1] == "directory" else 1
        mapping[directory] = parts[start:]
    return mapping


def append_index_rows(index_dir: Path, header: str, rows: list[str]) -> None:
    combined = index_dir / "combined_data.csv"
    if not combined.exists():
        combined.write_text(header)
    with combined.open("a") as f:
        for row in rows:
            f.write(row.rstrip("\n") + "\n")


def append_types_rows(index_dir: Path, rows: list[str]) -> None:
    types_csv = index_dir / "all_atom_types.csv"
    with types_csv.open("a") as f:
        for row in rows:
            f.write(row.rstrip("\n") + "\n")


def next_model_idx(base: Path, pattern: str) -> int:
    best = 0
    rx = re.compile(pattern)
    if base.is_dir():
        for child in base.iterdir():
            m = rx.fullmatch(child.name)
            if m:
                best = max(best, int(m.group(1)))
    return best + 1


def load_manifest(soap_root: Path) -> dict:
    path = soap_root / "data_manifest.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"version": 0, "ingested": []}


def save_manifest(soap_root: Path, manifest: dict) -> None:
    manifest["version"] = int(manifest.get("version", 0)) + 1
    manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
    (soap_root / "data_manifest.json").write_text(json.dumps(manifest, indent=2))


def discover_samples(source: Path) -> list[Path]:
    """A source dir either IS one sample (has data.lmp+neb.txt) or CONTAINS sample dirs."""
    if all((source / f).is_file() for f in REQUIRED_SAMPLE_FILES):
        return [source]
    found = []
    for child in sorted(source.iterdir()):
        if child.is_dir() and all((child / f).is_file() for f in REQUIRED_SAMPLE_FILES):
            found.append(child)
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description="QC + ingest NEB samples into training layout.")
    ap.add_argument("--soap-root", type=Path,
                    default=Path(os.environ.get("SOAP_ROOT", "")).expanduser() or None,
                    help="training-data root (NEB_new_data/ NEB_SIA/ live here); "
                         "env SOAP_ROOT; required")
    ap.add_argument("--kind", choices=["vacancy", "sia"], required=True)
    ap.add_argument("--pair-type", required=True,
                    help="e.g. MoRe, MoGa (migrating-atom pair label)")
    ap.add_argument("--source", type=Path, required=True,
                    help="dir containing finished NEB sample dirs (or one sample dir)")
    ap.add_argument("--conc", default=None,
                    help="vacancy: concentration tag like Re_0.05 (required for vacancy)")
    ap.add_argument("--cluster", default=None,
                    help="sia: cluster count tag like N1 (required for sia)")
    ap.add_argument("--site-ids", default=None,
                    help="comma triple of pivot site ids, e.g. 2000,2001,1111 "
                         "(vacancy) or gap1,gap2,target (sia). Default reuses "
                         "the ids from the most recent index row.")
    ap.add_argument("--site-types", default=None,
                    help="comma triple of site types, e.g. 1,2,1. Default同上.")
    ap.add_argument("--endpoint-tol", type=float, default=0.05,
                    help="eV tolerance for endpoint |dE| (default 0.05)")
    ap.add_argument("--min-images", type=int, default=3,
                    help="minimum NEB points in neb.txt (default 3)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.soap_root is None:
        print("❌ --soap-root (or env SOAP_ROOT) is required", file=sys.stderr)
        return 2
    if not args.soap_root.is_dir():
        print(f"❌ --soap-root not found: {args.soap_root}", file=sys.stderr)
        return 2
    if args.kind == "vacancy" and not args.conc:
        print("❌ --conc is required for vacancy ingest (e.g. Re_0.05)", file=sys.stderr)
        return 2
    if args.kind == "sia" and not args.cluster:
        print("❌ --cluster is required for sia ingest (e.g. N1)", file=sys.stderr)
        return 2

    soap_root = args.soap_root
    if args.kind == "vacancy":
        index_dir = soap_root / "NEB_new_data" / args.pair_type
        sample_base = index_dir / args.conc
        header = VAC_HEADER
        dest_pattern = r"model_(\d+)"
    else:
        index_dir = soap_root / "NEB_SIA" / args.pair_type / "configs"
        sample_base = index_dir / args.cluster
        header = SIA_HEADER
        dest_pattern = r"N(\d+)"

    samples = discover_samples(args.source)
    if not samples:
        print(f"❌ no samples with {REQUIRED_SAMPLE_FILES} under {args.source}",
              file=sys.stderr)
        return 2

    # site id/type defaults: reuse the newest existing index row
    existing_types = read_all_atom_types(index_dir / "all_atom_types.csv")
    if args.site_ids and args.site_types:
        site_ids = [s.strip() for s in args.site_ids.split(",")]
        site_types = [s.strip() for s in args.site_types.split(",")]
        if len(site_ids) != 3 or len(site_types) != 3:
            print("❌ --site-ids/--site-types must be comma triples", file=sys.stderr)
            return 2
    else:
        site_ids = site_types = None

    manifest = load_manifest(soap_root)
    ingested_sources = {e["source"] for e in manifest.get("ingested", [])}

    plan, qc_fail = [], []
    for sample_dir in samples:
        src_key = str(sample_dir.resolve())
        if src_key in ingested_sources:
            continue  # idempotent skip
        ok, reason, barrier = qc_sample(sample_dir, args.endpoint_tol, args.min_images)
        if not ok:
            qc_fail.append((sample_dir.name, reason))
            continue
        plan.append((sample_dir, src_key, barrier))

    if qc_fail:
        print(f"⚠️  {len(qc_fail)} samples failed QC:", file=sys.stderr)
        for name, reason in qc_fail:
            print(f"   ❌ {name}: {reason}", file=sys.stderr)
    if not plan:
        print("nothing to ingest (all skipped or failed QC).")
        return 1 if qc_fail else 0

    # resolve site ids/types per sample
    rows_plan = []
    next_idx = next_model_idx(sample_base, dest_pattern)
    for offset, (sample_dir, src_key, barrier) in enumerate(plan):
        dest_name = (f"model_{next_idx + offset}" if args.kind == "vacancy"
                     else f"N{next_idx + offset}")
        rel_dir = f"./{args.conc}/{dest_name}" if args.kind == "vacancy" \
            else f"./{args.cluster}/{dest_name}"
        sids, stypes = site_ids, site_types
        if sids is None:
            # reuse ids/types from the latest existing index row of this pair
            if existing_types:
                last_dir = next(reversed(existing_types))
                # ids/types live in combined_data.csv, not all_atom_types — read last combined row
                combined = index_dir / "combined_data.csv"
                last = [l for l in combined.read_text().splitlines() if l.strip()][-1]
                cols = last.split(",")
                if args.kind == "vacancy":
                    sids = [cols[1], cols[3], cols[5]]
                    stypes = [cols[2], cols[4], cols[6]]
                else:
                    sids = [cols[1], cols[3], cols[5]]
                    stypes = [cols[2], cols[4], cols[6]]
            else:
                print("❌ no existing index rows to infer --site-ids/--site-types from; "
                      "pass them explicitly", file=sys.stderr)
                return 2
        if args.kind == "vacancy":
            row = f"{rel_dir},{sids[0]},{stypes[0]},{sids[1]},{stypes[1]},{sids[2]},{stypes[2]},{barrier:.6f}"
        else:
            row = f"{rel_dir},{sids[0]},{stypes[0]},{sids[1]},{stypes[1]},{sids[2]},{stypes[2]},{barrier:.6f}"
        rows_plan.append((sample_dir, src_key, sample_base / dest_name, rel_dir, row, barrier))

    print(f"ingest plan: {len(rows_plan)} samples → {sample_base}")
    for sample_dir, _, dest, rel_dir, _, barrier in rows_plan:
        print(f"  {sample_dir.name:24s} → {rel_dir:24s} barrier={barrier:.4f} eV")
    if args.dry_run:
        print("dry-run, nothing written.")
        return 0

    for sample_dir, src_key, dest, rel_dir, row, barrier in rows_plan:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(sample_dir, dest)
        append_index_rows(index_dir, header, [row])
        # all_atom_types.csv must keep 1 row per combined row (load_samples enforces
        # len(type_map)==len(combined)); types come from the ingested data.lmp.
        atom_types = read_atom_types(dest / "data.lmp")
        types_row = f"{rel_dir},directory," + ",".join(str(t) for t in atom_types)
        append_types_rows(index_dir, [types_row])
        manifest.setdefault("ingested", []).append({
            "source": src_key, "dest": str(dest), "rel_dir": rel_dir,
            "pair_type": args.pair_type, "kind": args.kind,
            "barrier_ev": round(barrier, 6), "n_atoms": len(atom_types),
            "utc": datetime.now(timezone.utc).isoformat(),
        })

    save_manifest(soap_root, manifest)
    print(f"✅ ingested {len(rows_plan)} samples; manifest version "
          f"{manifest['version']} written to {soap_root/'data_manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

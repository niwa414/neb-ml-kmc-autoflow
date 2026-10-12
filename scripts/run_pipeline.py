#!/usr/bin/env python3
"""run_pipeline — orchestrate the s0→s5 stages from one JSON config.

Pure stdlib. Reads a pipeline config (JSON), runs stages in order, stops on the
first failing stage, and writes a run report. Two-strike retry lives INSIDE
s2_run_neb (per NEB task); the orchestrator itself runs each stage exactly once.

Config shape (see config/example_pipeline.json):
{
  "name": "more_ga_v1",
  "species": ["Mo", "Re", "Ga"],
  "kinds": ["vacancy"],
  "concs": ["Re_0.05"],
  "per_cell": 5,
  "lmp_path": "/abs/path/to/lmp",
  "potential": "/abs/path/MoReGa.eam.fs",
  "work_root": "runs/neb_campaign_more_ga_v1",
  "datasets": "NEB_new_data",
  "output_dir": "runs/soap_more_ga_v1",
  "mae_threshold": 0.20,
  "gate": "strict",
  "extra_train_args": "--soap-sparse",
  "smoke_input": null,
  "stages": ["s0", "s1", "s2", "s3", "s4", "s5"]
}

Exit codes: 0 all stages ok, 1 a stage failed (report printed), 2 usage.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
STAGE_ORDER = ["s0", "s1", "s2", "s3", "s4", "s5"]


def run_stage(name: str, cmd: list[str], log_path: Path) -> tuple[bool, str]:
    print(f"\n{'='*60}\n▶ {name}: {' '.join(cmd)}\n{'='*60}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    log_path.write_text(
        f"$ {' '.join(cmd)}\nexit={proc.returncode}\n--- stdout ---\n{proc.stdout}\n"
        f"--- stderr ---\n{proc.stderr}")
    tail = (proc.stdout or proc.stderr)[-600:]
    print(tail)
    return proc.returncode == 0, tail


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the full NEB→ML→KMC pipeline.")
    ap.add_argument("config", type=Path, help="pipeline JSON config")
    ap.add_argument("--stages", type=str, default=None,
                    help="comma stage subset, e.g. 's0,s1' (default: config's stages)")
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    species = cfg["species"]
    n_species = len(species)
    type_symbols = ",".join(f"{i+1}:{s}" for i, s in enumerate(species))
    species_str = ",".join(species)
    work_root = Path(cfg["work_root"]).expanduser().resolve()
    campaign = work_root / "campaign.json"
    logs = work_root / "stage_logs"
    logs.mkdir(parents=True, exist_ok=True)
    report = {"name": cfg.get("name"), "started_utc": datetime.now(timezone.utc).isoformat(),
              "stages": []}

    wanted = (args.stages.split(",") if args.stages else cfg.get("stages", STAGE_ORDER))
    stage_cmds: dict[str, list[str]] = {
        "s0": ["python3", str(SCRIPTS / "s0_doctor.py"),
               "--lmp-path", cfg["lmp_path"], "--potential", cfg["potential"]],
        "s1": ["python3", str(SCRIPTS / "s1_draft_neb.py"),
               "--species", species_str, "--concs", ",".join(cfg["concs"]),
               "--kinds", ",".join(cfg["kinds"]), "--per-cell", str(cfg["per_cell"]),
               "--potential", cfg["potential"], "--work-root", str(work_root)],
        "s2": ["python3", str(SCRIPTS / "s2_run_neb.py"),
               "--campaign", str(campaign), "--lmp-path", cfg["lmp_path"]],
        "s3": ["python3", str(SCRIPTS / "s3_ingest.py"),
               "--kind", cfg["kinds"][0], "--pair-type", cfg["pair_types"][0] if "pair_types" in cfg else "MoRe",
               "--conc", cfg["concs"][0], "--source", str(work_root)],
        "s4": ["python3", str(SCRIPTS / "s4_train.py"),
               "--data-root", cfg.get("data_root", os.environ.get("SOAP_ROOT", ".")),
               "--datasets", cfg.get("datasets", "NEB_new_data"),
               "--species", species_str, "--type-symbols", type_symbols,
               "--output-dir", cfg["output_dir"],
               "--mae-threshold", str(cfg.get("mae_threshold", 0.20)),
               "--gate", cfg.get("gate", "strict"),
               "--extra-args", cfg.get("extra_train_args", "")],
        "s5": ["python3", str(SCRIPTS / "s5_deploy_kmc.py"),
               "--model-dir", cfg["output_dir"],
               "--mae-threshold", str(cfg.get("mae_threshold", 0.20)),
               "--gate", cfg.get("gate", "strict")]
               + (["--misa-kmc", cfg["misa_kmc"], "--smoke-input", cfg["smoke_input"]]
                  if cfg.get("smoke_input") and cfg.get("misa_kmc") else []),
    }

    failed = None
    for stage in STAGE_ORDER:
        if stage not in wanted:
            continue
        ok, tail = run_stage(stage, stage_cmds[stage], logs / f"{stage}.log")
        report["stages"].append({"stage": stage, "ok": ok, "tail": tail[-300:]})
        if not ok:
            failed = stage
            break

    report["failed_stage"] = failed
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    report_path = work_root / "pipeline_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    if failed:
        print(f"\n❌ pipeline stopped at {failed}. Report: {report_path}", file=sys.stderr)
        return 1
    print(f"\n✅ pipeline complete. Report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

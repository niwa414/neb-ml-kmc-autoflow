---
name: neb-ml-kmc-autoflow
description: "Run the full LAMMPS-NEB → SOAP/ML training → misa-kmc deployment pipeline for MoRe/MoReX migration-barrier models. Use when the user asks to compute NEB datasets, retrain the barrier model, deploy a model to KMC, or run the end-to-end autoflow on ~/Desktop/kmc-ml + ~/Desktop/miet-claw."
---

# neb-ml-kmc-autoflow

Orchestration layer for the user's kmc-ml × miet-claw atomistic stack. This skill does NOT
own domain knowledge — read `materials-simulation:kmc-ml-automation` first (repo map,
MoRe→MoReX extension points, pipeline roadmap, pitfalls). This skill only *runs* the stages.

## Layout

```
scripts/s0_doctor.py     preflight: repos, lmp, misa-kmc, ML venv, potential, data dirs
scripts/s1_draft_neb.py  (P3) generate NEB workspaces + campaign.json
scripts/s2_run_neb.py    (P3) execute NEB locally/SLURM, parse neb.txt, update state
scripts/s3_ingest.py     (P2) QC NEB samples → training dir contract + dataset manifest
scripts/s4_train.py      (P4) call train_migration_soap.py + MAE admission gate
scripts/s5_deploy_kmc.py (P2) versioned model deploy + KMC smoke regression
scripts/run_pipeline.py  (P5) orchestrator: runs stages in order, two-strike retry
```

## Hard rules (user's loop discipline, inherited)

1. **Never guess paths.** Every external binary/file is `--flag`, env var, or a checked
   default. Missing → report and stop. (s0_doctor embodies this.)
2. **Two-strike per task family.** A NEB task that fails twice stops the campaign and the
   report goes to the user — no silent v3 retries.
3. **Deploy gate.** s5 only promotes a model whose per-bucket test MAE passes thresholds
   from the run config (`strict`), or explicitly tags it ⚠️ (`warn`).
4. **Binary compatibility.** Default CLI flags reproduce the pre-MoReX MoRe behavior
   byte-for-byte (P1 verified); ternary is opt-in via `--species/--type-symbols`.

## Typical invocations

```bash
# Preflight only (safe any time)
python3 scripts/s0_doctor.py                     # uses ~/Desktop defaults
python3 scripts/s0_doctor.py --lmp-path ~/miniconda3/envs/miet-stack/bin/lmp

# Full pipeline (P3+ scripts land later)
python3 scripts/run_pipeline.py config/more_w_pipeline.json
```

## Current status (see kmc-ml-automation references/pipeline-plan.md for live state)

All 7 scripts exist; `--help` smoke-passed. s0/s3/s5 are validated against real
kmc-ml data; s1/s2/run_pipeline have functional state machines with STUBs noted
inline (s1 structure generator, s2 NEB input templates — wire before production).

- ✅ s0_doctor — validated on this machine (finds the real gap: no lmp on PATH)
- ✅ s3_ingest — validated dry-run on real MoRe SIA/vacancy samples (QC catches
  bad neb.txt; layout contract verified vs train_migration_fnn.load_samples)
- ✅ s5_deploy_kmc — validated (gate read, self-deploy guard, rollback-tested
  staging archive; metrics MAE 0.0210 eV live model)
- 🚧 s1_draft_neb — campaign.json state machine real (27-task ternary draft
  verified); structure generator is a STUB (wire gen_final_one.py)
- 🚧 s2_run_neb — local executor + neb.txt parser + two-strike state machine
  real (dry-run verified); endpoint/NEB input templates are STUB-grade
- ✅ s4_train — trainer invocation + strict/warn gate
- ✅ run_pipeline — JSON-config orchestrator, stage logs, stop-on-first-fail

## User decisions on record (2026-10-12)

- X element ≈ **Ga** (placeholder; species stays fully parameterized, no Ga hardcode)
- Ternary EAM potential: user will supply later (s0 --potential flags missing until then)
- LAMMPS: **local** execution (s2 local mode is the primary path; SLURM TODO)
- Scope now: skeleton first (user approved this batch)

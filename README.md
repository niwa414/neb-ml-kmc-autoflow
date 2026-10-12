# neb-ml-kmc-autoflow

[![ci](https://github.com/niwa414/neb-ml-kmc-autoflow/actions/workflows/ci.yml/badge.svg)](https://github.com/niwa414/neb-ml-kmc-autoflow/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Automate the **LAMMPS NEB → machine-learning barrier model → KMC deployment**
pipeline for atomistic alloy-diffusion simulations. Built around a
`campaign.json` state machine: every stage is idempotent, resumable, and
runnable on its own.

Originally developed for MoRe / MoReX (Mo-Re-Ga) dumbbell-pair migration
barriers in BCC alloys; the scripts are alloy-agnostic — species, pair types,
and potentials are all run-time configuration.

## What it does

```
s0_doctor      preflight: repos, lmp/lmp_mpi/mpiexec, ML venv, potential, data dirs
s1_draft_neb   generate dumbbell NEB workspaces (real BCC/FCC alloy structures)
s2_run_neb     local CI-NEB via mpiexec (relax endpoints → final.mosia → NEB → neb.txt)
s3_ingest      QC + ingest samples into the training layout (combined_data.csv contract)
s4_train       invoke your trainer + test-MAE admission gate (strict|warn)
s5_deploy_kmc  versioned model deploy + optional KMC smoke run (staging/rollback)
run_pipeline   orchestrate s0→s5 from one JSON config, stop-on-first-fail
```

## Requirements

- **LAMMPS** built with the **REPLICA** package: a serial `lmp` plus `lmp_mpi`
  and `mpiexec` in the same bin directory (multi-replica NEB needs MPI).
  Conda-forge `lammps` works.
- Python ≥ 3.10, stdlib only for s0–s3/s5 and the orchestrator.
- s4 additionally needs your trainer (see "Trainer contract") — e.g. a
  scikit-learn/dscribe-based script.

## Quickstart (zero external files)

The LJ demo validates the full NEB mechanics on a toy FCC Ar system — no
potential file, no datasets:

```bash
python examples/lj_demo/run_demo.py --lmp /path/to/lmp
# → real 5-replica CI-NEB, writes neb.txt, prints the barrier
```

## Real campaign

```bash
# 1. preflight
python scripts/s0_doctor.py --lmp-path /path/to/lmp --potential /path/to/alloy.eam.fs

# 2. draft a campaign (9 pair types for a ternary, concentration grid)
python scripts/s1_draft_neb.py --species Mo,Re,Ga --concs Re_0.05 \
    --kinds vacancy --per-cell 5 --potential /path/to/alloy.eam.fs \
    --work-root runs/campaign_v1

# 3. run NEB (two-strike discipline per task; resumable)
python scripts/s2_run_neb.py --campaign runs/campaign_v1/campaign.json \
    --lmp-path /path/to/lmp

# 4. ingest into training layout
python scripts/s3_ingest.py --soap-root /path/to/data-root \
    --kind vacancy --pair-type MoRe --conc Re_0.05 --source runs/campaign_v1

# 5. train + gate
python scripts/s4_train.py --trainer /path/to/train_migration_soap.py \
    --data-root /path/to/data-root --datasets NEB_new_data \
    --species Mo,Re,Ga --type-symbols 1:Mo,2:Re,3:Ga \
    --output-dir runs/soap_v1 --mae-threshold 0.20

# 6. deploy (staging → archive → activate, rollback on smoke failure)
python scripts/s5_deploy_kmc.py --model-dir runs/soap_v1 \
    --runs-root /path/to/kmc-runs --mae-threshold 0.20
```

or everything at once:

```bash
python scripts/run_pipeline.py config/example_pipeline.json
```

## Configuration (env vars / flags, never hardcoded paths)

| Knob | Env var | CLI |
|---|---|---|
| alloy/ML repo roots | `KMC_ML_ROOT`, `MIETCLAW_ROOT` | `--kmc-ml-root`, `--mietclaw-root` |
| LAMMPS binary | `LMP_BIN` | `--lmp-path` |
| training-data root | `SOAP_ROOT` | `--soap-root`, `--data-root` |
| trainer script | `NEB_TRAINER` | `--trainer` |
| trainer python | `NEB_TRAIN_PYTHON` | `--python-bin` |
| KMC model live dir | `KMC_RUNS_ROOT` | `--runs-root` |

## Trainer contract

s4 invokes your trainer as a subprocess and then gates on
`<output-dir>/metrics.json` (it searches recursively for a block with
`mae`/`rmse`). s5 deploys a directory containing `model.joblib` +
`scaler.npz` (+ `metrics.json`). Any trainer that emits those artifacts works.

Training-data layout expected by s3 (vacancy kind):

```
<soap-root>/NEB_new_data/<PairType>/Re_<conc>/model_<N>/{data.lmp,data_final.lmp,dump.neb.*,neb.txt}
<soap-root>/NEB_new_data/<PairType>/combined_data.csv   # directory,ids/types...,max_value
<soap-root>/NEB_new_data/<PairType>/all_atom_types.csv  # <directory>[,directory],t1,t2,...
```

SIA kind uses `NEB_SIA/<PairType>/configs/N<count>/N<idx>/` with
`gap1/gap2/target` columns.

## Current status & honest limits

- ✅ End-to-end validated on macOS with conda-forge LAMMPS (real 5-replica
  CI-NEB on BCC MoRe EAM and the LJ demo).
- ✅ s1 structures verified against a legacy MoRe dataset generator.
- ⚠️ s4→s5 chain validated on real trainer artifacts, but no full KMC smoke
  regression in this repo (needs your `misa-kmc` binary + input deck).
- ⬜ SLURM executor for s2 is a TODO (local MPI only today).
- ⬜ Cluster-translation event sampling (collective dumbbell moves) is planned
  for phase 2 — the KMC side already supports it.

## Contributing

Issues and PRs welcome. Keep the four hard rules: never guess paths, two-strike
per task family, strict deploy gate, and byte-identical backward compatibility
of default flags.

## License

MIT — see [LICENSE](LICENSE).

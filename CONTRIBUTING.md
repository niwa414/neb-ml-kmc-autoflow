# Contributing to neb-ml-kmc-autoflow

Thanks for wanting to improve this pipeline. This document is short on purpose —
the rules below are load-bearing, and the extension-point map tells you where
new capability belongs.

## Hard rules (load-bearing, do not break)

1. **Never guess paths.** Every external binary/file is a CLI flag, an env var,
   or a checked default. Missing → report and stop. Rationale: this pipeline
   runs long campaigns; a silent wrong guess costs hours and corrupts datasets.
2. **Two-strike per task family.** A failing task retries at most twice, then
   the campaign stops with a report. No silent v3. Rationale: inherited loop
   discipline from the original research workflow.
3. **Strict deploy gate.** s5 promotes a model only when its test MAE passes
   the configured threshold (`strict`), or is explicitly tagged ⚠️ (`warn`).
   Rationale: a bad barrier model silently poisons every downstream KMC run.
4. **Default-flag backward compatibility.** Default CLI flags must reproduce
   the original (pre-ternary binary) behavior byte-for-byte. New chemistry is
   always opt-in via `--species`/`--type-symbols` and friends. If you must
   change a default, document it as a breaking change in the PR and bump the
   minor version.

## Extension points (where new capability goes)

| Want to add... | Where | Notes |
|---|---|---|
| SLURM / cluster executor for NEB | `scripts/s2_run_neb.py` `run_one()` stage 3 | Add a `--backend slurm` branch; keep the local MPI path as default. Preserve the `neb.txt` output contract and the two-strike state machine. |
| New defect type (e.g. vacancy clusters) | `scripts/structure_gen.py` + `scripts/s1_draft_neb.py` (`make_sample_structures`) | Keep the dumbbell convention for dumbbell kinds; add a sibling generator with its own `meta` schema. s3_ingest's `--kind` already plumbs the index columns. |
| New lattice / geometry | `scripts/structure_gen.py` `_bcc_sites`/`_fcc_sites` | Add `_xxx_sites` and a `lattice_type` branch. Validate with the overlap checker (no two atoms closer than 0.5 Å at generation time). |
| A different ML model backend | `scripts/s4_train.py` only | Contract: `<output-dir>/model.joblib` + `scaler.npz` + `metrics.json` (with a `mae` block). s5 does not care how the model was trained. |
| QC rules for NEB samples | `scripts/s3_ingest.py` `qc_sample()` | Add checks, keep them conservative — rejecting a borderline sample is cheaper than training on garbage. Log every rejection with its reason. |
| KMC smoke test variants | `scripts/s5_deploy_kmc.py` `smoke_test()` | The smoke contract is: run misa-kmc on a small deck, assert ML-barrier activity in the log, assert sane barrier magnitudes. |

## What NOT to change without an issue first

- The `campaign.json` task schema and state machine (`pending/running/done/failed`
  + `strikes`). Other tooling (resume, reporting) reads it.
- The training-data layout contract (`combined_data.csv` / `all_atom_types.csv`
  columns, directory naming). It is consumed by existing trainers.
- The `neb.txt` format (`#reaction_crodinate de` + `<rc> <dE>` lines, endpoints
  0/1). Downstream feature extraction parses it.

## Local development loop

```bash
# 1. run the zero-dependency demo (needs LAMMPS with REPLICA)
python examples/lj_demo/run_demo.py --lmp /path/to/lmp

# 2. static checks (py_compile all scripts, then --help each)
python -m py_compile scripts/*.py
for s in scripts/s*_*.py; do python "$s" --help >/dev/null || echo "FAIL $s"; done

# 3. draft-only dry run (no LAMMPS needed)
python scripts/s1_draft_neb.py --species Mo,Re --concs Re_0.05 \
    --kinds vacancy --per-cell 1 --potential /path/to/pot.eam.fs \
    --work-root /tmp/draft_check --dry-run
```

## PR requirements

- One logical change per PR.
- Include evidence: what command you ran, and its output (or CI link). This
  project treats "I changed X" without evidence as not done — see the ✅/❌
  convention in existing commits.
- State explicitly whether default behavior changes (rule 4).
- Keep scripts stdlib-only unless the dependency is justified in the PR and
  documented in README.

## Issues

File issues for: bug reports (include the failing command + tail of output),
feature proposals (say which extension point), and documentation gaps.

# scripts

## Purpose
Phase command entry points.

## Files
`00_verify_environment.py` verifies the environment. Scripts 01–08 cover audit, safe derived data and annotations. `09_train_baseline.py` owns Phase 5 baseline runs. `phase6_smoke_overfit.py` verifies HOSP-BoneNet on 16 fit-only samples. Scripts 10–15 remain phase-gated.

## Inputs
Configuration, installed package, immutable source data, label schema, and Phase 2 manifests in stage order.

## Outputs
JSON summaries, derived workspace tables, QA artifacts, immutable experiment bundles, and nonzero status for failed invariants.

## Main Libraries
argparse, pathlib, Pandas and reusable package audit modules.

## Important Functions / Classes
Each `main` resolves derived destinations through `ProjectPaths`; atomic writers avoid partial tables. Script 04 retains every source row in `dataset_clean.csv` and logs exclusions separately. Script 05 validates source hashes before caching. Script 06 refuses overlapping/unassigned groups and verifies a batch from every split.

## Theory
Thin scripts keep core logic reusable. The corrected partitions reference immutable source paths, so leakage repair never changes official folders. Calibration selection uses only clean official-training labels/groups.

## Execution
Use `./scripts/run_phase.ps1 -Action test` for tests or `python scripts/phase6_smoke_overfit.py` for the Phase 6 acceptance run.

## Notes
Never edit Main or generated outputs manually. Phase 3 depends on final Phase 2 master/duplicate reports. Training must consume `workspace/splits/fit_train.csv`, never official folders or the calibration/held-out manifests.

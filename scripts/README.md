# scripts

## Purpose
Phase command entry points.

## Files
`00_verify_environment.py` verifies the environment. Scripts 01–08 cover audit, cache and annotations. `09_train_baseline.py` owns Phase 5 baselines; `10_train_hosp_bonenet.py` owns Phase 7 HOSP training/reuse. Scripts 11–15 remain phase-gated.

## Inputs
Configuration, installed package, immutable source data, label schema, and Phase 2 manifests in stage order.

## Outputs
JSON summaries, derived workspace tables, QA artifacts, immutable experiment bundles, and nonzero status for failed invariants.

## Main Libraries
argparse, pathlib, Pandas and reusable package audit modules.

## Important Functions / Classes
Each `main` resolves safe derived paths. Script 05 builds/reuses the required fit/validation cache. Scripts 09/10 validate and reuse exact completed runs by default; forced runs are explicit.

## Theory
Thin scripts keep core logic reusable. The corrected partitions reference immutable source paths, so leakage repair never changes official folders. Calibration selection uses only clean official-training labels/groups.

## Execution
Use `.venv\Scripts\python.exe scripts/05_build_cache.py` for cache reuse and `python tools/refresh_checkpoint_registry.py` for selected checkpoints.

## Notes
Never edit Main or generated outputs manually. Phase 3 depends on final Phase 2 master/duplicate reports. Training must consume `workspace/splits/fit_train.csv`, never official folders or the calibration/held-out manifests.

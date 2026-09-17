# scripts

## Purpose
Phase command entry points.

## Files
`00_verify_environment.py` verifies the environment. Scripts 01–08 cover audit, cache and annotations. `09_train_baseline.py` owns Phase 5 baselines; `10_train_hosp_bonenet.py` owns Phase 7 training; `11_tune_hosp_bonenet.py` owns Phase 8 optimization/reuse. Scripts 12–15 remain phase-gated.

## Inputs
Configuration, installed package, immutable source data, label schema, and Phase 2 manifests in stage order.

## Outputs
JSON summaries, derived workspace tables, QA artifacts, immutable experiment bundles, and nonzero status for failed invariants.

## Main Libraries
argparse, pathlib, Pandas and reusable package audit modules.

## Important Functions / Classes
Each `main` resolves safe derived paths. Script 05 builds/reuses the fit/validation cache. Scripts 09–11 validate and reuse exact completed runs by default; forced runs are explicit.

## Theory
Thin scripts keep core logic reusable. The corrected partitions reference immutable source paths, so leakage repair never changes official folders. Calibration selection uses only clean official-training labels/groups.

## Execution
Run Phase 8 with `.venv\Scripts\python.exe scripts\11_tune_hosp_bonenet.py --config configs\optimization.yaml`.

## Notes
Never edit Main or generated outputs manually. Phase 3 depends on final Phase 2 master/duplicate reports. Training must consume `workspace/splits/fit_train.csv`, never official folders or the calibration/held-out manifests.

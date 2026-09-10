# scripts

## Purpose
Phase command entry points.

## Files
`00_verify_environment.py` verifies the environment. Scripts 01–03 perform the Phase 2 audit. `04_build_clean_manifest.py` applies logged quality/conflict/leakage rules; `05_build_cache.py` optionally writes traceable resized PNGs; `06_make_internal_splits.py` creates grouped partitions and QA batches. `07_launch_annotation_tool.py` checks/launches local review; `08_validate_annotations.py` validates project/source records and exports CSV, with --import-source creating a new conversion generation. Scripts 09–15 remain phase-gated. `run_phase.ps1` exposes checks plus ordered Phase 2/3 workflows.

## Inputs
Configuration, installed package, immutable source data, label schema, and Phase 2 manifests in stage order.

## Outputs
JSON summaries on stdout, clean/split/cache tables under workspace, the preprocessing batch grid under artifacts, and nonzero status for unsafe paths or failed invariants.

## Main Libraries
argparse, pathlib, Pandas and reusable package audit modules.

## Important Functions / Classes
Each `main` resolves derived destinations through `ProjectPaths`; atomic writers avoid partial tables. Script 04 retains every source row in `dataset_clean.csv` and logs exclusions separately. Script 05 validates source hashes before caching. Script 06 refuses overlapping/unassigned groups and verifies a batch from every split.

## Theory
Thin scripts keep core logic reusable. The corrected partitions reference immutable source paths, so leakage repair never changes official folders. Calibration selection uses only clean official-training labels/groups.

## Execution
`./scripts/run_phase.ps1 -Action phase2`; then `./scripts/run_phase.ps1 -Action phase3`. Script 05 is optional and accepts `--limit`. Use `-Action test`, `lint` and `compile` for checks.

## Notes
Never edit Main or generated outputs manually. Phase 3 depends on final Phase 2 master/duplicate reports. Training must consume `workspace/splits/fit_train.csv`, never official folders or the calibration/held-out manifests.

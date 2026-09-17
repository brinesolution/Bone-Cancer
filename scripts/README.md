# scripts

## Purpose
Phase command entry points.

## Files
Scripts 01–08 cover audit/data/annotations; 09 trains baselines, 10 trains HOSP, 11 optimizes it, and 12 performs the sealed Phase 9 evaluation. Scripts 13–15 remain phase-gated.

## Inputs
Configuration, installed package, immutable source data, label schema, and Phase 2 manifests in stage order.

## Outputs
JSON summaries, derived workspace tables, QA artifacts, immutable experiment bundles, and nonzero status for failed invariants.

## Main Libraries
argparse, pathlib, Pandas and reusable package audit modules.

## Important Functions / Classes
Each `main` resolves safe derived paths. Scripts 09–11 reuse training artifacts; script 12 reuses checksum-sealed test predictions.

## Theory
Thin scripts keep core logic reusable. The corrected partitions reference immutable source paths, so leakage repair never changes official folders. Calibration selection uses only clean official-training labels/groups.

## Execution
Run Phase 9 with `.venv\Scripts\python.exe scripts\12_evaluate.py --config configs\evaluation.yaml`; after technical visual QA, append `--finalize-visual-qa` to publish it.

## Notes
Never edit Main or generated outputs manually. Phase 3 depends on final Phase 2 master/duplicate reports. Training must consume `workspace/splits/fit_train.csv`, never official folders or the calibration/held-out manifests.

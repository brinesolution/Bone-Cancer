# scripts

## Purpose
Phase command entry points.

## Files
`00_verify_environment.py` verifies the environment. `01_index_dataset.py` builds master/quality/label-structure records; `02_audit_dataset.py` validates provenance and renders figures; `03_find_duplicates.py` builds duplicate/leakage groups. Scripts 04–15 remain phase-gated scaffolds. `run_phase.ps1` exposes common checks and the ordered Phase 2 audit.

## Inputs
Configuration, installed package, immutable source data, and Phase 2 manifests in stage order.

## Outputs
JSON summaries on stdout, generated CSV/PNG/JSON evidence in workspace/artifacts/logs, and nonzero status for unsafe paths or structural label/semantic failures.

## Main Libraries
argparse, pathlib, Pandas and reusable package audit modules.

## Important Functions / Classes
Each `main` validates output paths before writes. Script 01 contains per-image decode failures; script 02 requires verified binary semantics; script 03 reports leakage but does not fail merely because candidates exist. `run_phase.ps1` stops the sequence after a failed stage.

## Theory
Thin scripts keep data logic reusable and atomic writers prevent partial derived tables. Leakage findings are evidence for Phase 3 rather than source mutations.

## Execution
`./scripts/run_phase.ps1 -Action phase2`, or run scripts 01, 02 and 03 in order. Use `-Action test`, `lint` and `compile` for checks.

## Notes
Never reorder the Phase 2 stages because audit/dedupe consume the master. Do not edit Main or start training from official folders before Phase 3 produces clean manifests.

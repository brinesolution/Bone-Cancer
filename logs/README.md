# logs

## Purpose
Local execution evidence.

## Files
Phase 1 environment/source evidence is retained. `dataset_index_summary.json`, `dataset_audit_summary.json`, `duplicate_summary.json` and `phase2_*.txt` record Phase 2 execution; `source_after_phase2.json` records the final immutability comparison.

## Inputs
Executed installation, tests, environment, dataset audit and integrity checks.

## Outputs
Local logs; sensitive source paths remain ignored by Git.

## Main Libraries
Python logging/json/hashlib; pytest command output.

## Important Functions / Classes
`snapshot_source` reads originals; `configure_logging` emits JSON lines; the three Phase 2 CLIs atomically write summaries.

## Theory
Recorded commands and hashes distinguish executed evidence from intended behavior.

## Execution
Run the owning CLI; use `python scripts/00_verify_environment.py --require-cuda --output logs/environment_new.json` only for a new environment report path.

## Notes
Do not edit evidence manually or overwrite Phase 1 baselines. Logs contain local absolute paths and remain Git-ignored.

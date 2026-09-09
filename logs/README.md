# logs

## Purpose
Local execution evidence.

## Files
Phase 1–2 environment/audit evidence is retained. Phase 3 adds `clean_manifest_summary.json`, `internal_split_summary.json`, `cache_summary.json`, `phase3_data_verification.json`, command transcripts, and `source_after_phase3.json`.

## Inputs
Executed installation, tests, environment, dataset audit, cleaning/split/cache workflows and integrity checks.

## Outputs
Local logs; sensitive source paths remain ignored by Git.

## Main Libraries
Python logging/json/hashlib; pytest command output.

## Important Functions / Classes
`snapshot_source` reads originals; `configure_logging` emits JSON lines; active Phase 2/3 CLIs atomically write summaries.

## Theory
Recorded commands and hashes distinguish executed evidence from intended behavior.

## Execution
Run the owning CLI. Phase 3 ordered execution is `./scripts/run_phase.ps1 -Action phase3`; optional cache uses script 05.

## Notes
Do not edit evidence manually or overwrite Phase 1 baselines. Logs contain local absolute paths and remain Git-ignored.

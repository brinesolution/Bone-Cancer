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

Phase 4 records full import/validation, initial rejected conversion, pytest output, LabelMe installation/CLI checks, unchanged source snapshot and preserved DataModule split counts. phase4_source_import.json preserves the accepted generation and coverage; phase4_annotation_validation.json is the latest validation report. phase4_make_qa_grid.py assembles existing overlays for technical inspection. These logs report source/project provenance separately and contain no model-performance claims.

# logs

## Purpose
Local execution evidence.

## Files
source_before_phase1.json: initial file hashes/sizes/mtimes; source_after_phase1.json: final comparison snapshot when verified; environment_phase1.json: verifier report; install_*.log/txt: dependency output; phase1_*.txt: tests/checks; .gitkeep: folder marker.

## Inputs
Executed installation, tests, environment and integrity checks.

## Outputs
Local logs; sensitive source paths remain ignored by Git.

## Main Libraries
Python logging/json/hashlib; pytest command output.

## Important Functions / Classes
snapshot_source reads originals; configure_logging emits JSON lines.

## Theory
Recorded commands and hashes distinguish executed evidence from intended behavior.

## Execution
python scripts/00_verify_environment.py --require-cuda --output logs/environment_new.json

## Notes
Do not edit generated records manually. Keep original dataset files untouched. Future-phase scaffolds are not completed functionality; update this README when implementation becomes active.

# src/bone_cancer_ai/utils

## Purpose
Shared infrastructure.

## Files
config.py: safe unique-key YAML; paths.py: source/output boundaries; logging.py: UTC JSON records; seed.py: RNG/backends; device.py: explicit CUDA/CPU and synthetic check; reproducibility.py: read-only hashes; environment.py: dependency and compute report.

## Inputs
Explicit configs, seeds, validated output destinations.

## Outputs
Configuration dicts, paths, log records, snapshots and environment JSON.

## Main Libraries
PyYAML, NumPy, torch; pathlib/logging/hashlib/importlib/json.

## Important Functions / Classes
load_config rejects ambiguous mappings; ProjectPaths.output_path resolves and guards destinations; configure_logging owns package handlers; seed_everything seeds Python/NumPy/torch; get_device never silently downgrades; verify_tensor_device performs no model training; snapshot_source hashes read-only files; environment_report verifies package imports.

## Theory
Source safety is a destination contract: every output caller must validate the path. PYTHONHASHSEED must be set before process start; deterministic algorithms can raise on unsupported operations.

## Execution
python scripts/00_verify_environment.py; pytest tests/test_utils.py -q

## Notes
Do not edit generated records manually. Keep original dataset files untouched. Future-phase scaffolds are not completed functionality; update this README when implementation becomes active.

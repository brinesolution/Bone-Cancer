# artifacts/checkpoints

## Purpose
Promoted model weights and their verified alias registry.

## Files
README.md: folder contract; registry.json: tracked metadata only; `*.pt`: local immutable weights.

## Inputs
Selected `best.pt` checkpoints from verified runs.

## Outputs
`registry.json` maps stable aliases to project-relative checkpoints and their SHA-256 digests.

## Main Libraries
Python standard library (`hashlib`, `json`, `os`, `shutil`).

## Important Functions / Classes
`bone_cancer_ai.artifacts.registry.promote_checkpoint` and `resolve_checkpoint` own promotion;
`replace_alias=True` is required to replace a verified alias after a validated rerun.

## Theory
Promotion hard-links when possible, falls back to an atomic copy, and refuses invalid or changed files.
Alias replacement is explicit and retains prior content-addressed checkpoint files.

## Execution
Later training phases register selected checkpoints through the shared API.

## Notes
Do not edit `registry.json` or generated weights manually. Git tracks registry metadata, never large checkpoints.

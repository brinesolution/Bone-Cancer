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
`replace_alias=True` is required to replace a matching verified `.selected` alias after a validated rerun.

## Theory
Promotion hard-links when possible, falls back to an atomic copy, and refuses invalid or changed files.
Alias replacement is explicit, requires the same model family and phase, and retains prior
content-addressed checkpoint files.

## Execution
Phase 5/7 training registers selected checkpoints through the shared API. Run
`python tools/refresh_checkpoint_registry.py` to revalidate existing selections and
register missing or idempotent aliases without training. Refresh refuses to replace a
selected alias that identifies a different artifact.

## Notes
Do not edit `registry.json` or generated weights manually. Git tracks registry metadata, never large checkpoints.

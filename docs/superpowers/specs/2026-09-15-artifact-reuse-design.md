# Artifact Reuse Design

## Goal

Reuse verified model and preprocessing artifacts across reruns without overwriting experiments or reading labels from filenames.

## Contracts

- `artifacts/checkpoints/registry.json` maps stable aliases to immutable, SHA-256-verified checkpoints.
- Promoted checkpoints use an NTFS hard link when possible and an atomic copy fallback.
- Registry paths are project-relative and must remain inside the project root.
- An exact completed HOSP run is validated and reused by default; `--force-new-run` is required to retrain the same configuration.
- An exact failed HOSP run resumes from `last.pt`; an explicit `--resume-run` remains available.
- Baseline and future scripts use the same registry/resolver API.
- The cache stores deterministic decoded, grayscale, padded 384×384 PNGs before random augmentation and normalization.
- Cache entries are keyed by source SHA-256 plus preprocessing configuration and verified before use.
- `dataset/Main` remains read-only. Test labels remain untouched until Phase 9.

## Stored state

Keep `best.pt` and `last.pt` in every run. Promote only selected `best.pt` files into the checkpoint store. Registry entries include alias, model family, phase, source run, selected metric, checkpoint path/hash, configuration fingerprint, and test status.

## Failure behavior

Missing, outside-root, changed, ambiguous, incomplete, or checksum-invalid artifacts fail closed. A stale cache entry falls back only when cache use is explicitly optional; production fit/validation cache use is required after the cache is built.

## Verification

Unit tests cover registry promotion/resolution, tampering, completed-run reuse, failed-run resume, force-new behavior, cache reuse/equivalence, source preservation, and CLI validation. Full pytest, Ruff, compileall, bundle validation, and source snapshot comparison remain required.

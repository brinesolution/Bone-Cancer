# Latest phase handoff

PHASE 3 COMPLETE

Created/changed:
- Clean-manifest schema/policy with opaque IDs, explicit eligibility, exclusions, reviews and held-out-priority leakage correction.
- Deterministic group-respecting fit/calibration assignment, full split-generation fingerprint, optional weighted sampler and fail-closed joint DataModule validation.
- Safe grayscale/RGB/RGBA decoding, aspect-preserving padding, aligned image/mask transforms, conservative fit augmentation and deterministic evaluation preprocessing.
- Partial-supervision Dataset/DataModule, runtime corrupt-image containment, optional traceable resized cache and real-batch QA grid.
- Active scripts 04–06, ordered PowerShell/VS Code Phase 3 workflow, 30 focused Phase 3 tests, updated project/folder documentation and generated local evidence.

Validation performed:
- Rebuilt the clean manifest and all four split manifests through `./scripts/run_phase.ps1 -Action phase3`; both stages exited 0.
- Loaded one real batch of eight from fit, calibration, validation and test; every image tensor was `[8,3,384,384]` and mask tensor `[8,1,384,384]`.
- Verified 9,063 eligible records occur exactly once, with zero record/sample/group overlap, zero stale generation mismatch, zero unassigned rows and zero official validation/test rows moved into training.
- Tested deterministic validation/loaders/cache, seeded safe augmentation, shape/padding, filename independence, missing/malformed optional labels, aligned masks, a complete mixed valid/corrupt DataLoader epoch, and unsafe/stale split rejection.
- Visually inspected `preprocessing_batch_grid.png`; anatomy, grayscale conversion and padding are reasonable across all partitions.
- Pytest: 89 passed; focused Phase 3: 30 passed; Ruff, compileall and pip check passed; independent review found no remaining Critical, Important or Minor findings after fixes.
- Optional cache: 16 requested, 16 cached, 0 failed; source/cache hashes verified and stale outputs are removed after a failed rebuild.
- Full source snapshot matched Phase 2: 12,560 files and 1,120,418,007 bytes unchanged.

Results:
- Clean manifest: 12,557 total, 9,063 eligible, 3,494 excluded. Reasons: 3,490 lower-priority cross-split duplicates and four records from two exact-byte label-conflict groups.
- Review flags: 37 records in three pHash-only label-conflict groups and two near-blank records; all remain grouped/traceable.
- Final split counts: fit 4,871; calibration 857; validation 1,453; test 1,882. Calibration is 14.9616% of the 5,728-record clean official-training pool.
- Final class counts: fit 2,197 normal/2,674 bone_tumour; calibration 385/472; validation 711/742; test 1,023/859.
- Binary/sample validity is active. Subtype, risk, concepts and spatial masks remain disabled pending Phase 4 gates.
- No training, checkpoints, accuracy, AUROC, AUPRC, F1, Dice, calibration or clinical verification results exist.

Outstanding issues:
- Phase 4 must review pHash-only label conflicts, the one ambiguous BTXRD subtype row, and validate annotation coordinate conversion/overlays before enabling optional heads.
- No patient/study identifier is available; duplicate connected groups remain the strongest independence control.
- Mixed orientation, burned-in text/markers, film photographs and low-exposure/capture artifacts remain shortcut and quality risks for later evaluation.

Do not rerun:
- Environment creation, dependency installation, scaffold generation, Phase 1 snapshots or Phase 2 audit unless their source/configuration changes.
- Clean/split/cache generation unless upstream manifests or Phase 3 policy/configuration changes; never edit generated CSVs/cache files manually.
- Do not modify `dataset/Main`, train from official folders, or enable subtype/risk/spatial/concept heads before validation gates pass.

Next recommended model:
ASTRA

Why that model:
Phase 4 requires local annotation-tool integration, source/overlay visual inspection, coordinate validation and end-to-end reviewer workflow checks.

Next phase:
Phase 4 — Annotation and Labeling System

Resume file:
E:\Bone Cancer\PROJECT_STATE.md

Manual model switch required: select ASTRA before continuing; automatic switching is unavailable.

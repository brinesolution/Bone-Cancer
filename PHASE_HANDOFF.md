# Phase handoff

Phase completed: BTXRD migration Phase A — provenance audit and manifests

Important outputs:
- Plan: `docs/superpowers/plans/2026-09-18-btxrd-subtype-migration.md`
- Audit: `scripts/16_prepare_btxrd.py`; derived artifacts: `workspace/btxrd/`
- Separate schema/config: `configs/btxrd_label_schema.yaml`, `configs/hosp_btxrd_subtype.yaml`
- Wiring smoke: `scripts/17_validate_btxrd_subtype.py`

Tests:
- PASS — focused migration/config/HOSP tests (31)
- PASS — real BTXRD audit: 3,746/3,746 images, exact Main hash/path linkage, no missing/extra files
- PASS — duplicate-aware groups/splits; CPU and CUDA masked subtype forward/backward smoke
- PASS — source tree is read-only; frozen production artifacts unchanged

Outstanding issue:
- Phase B training, fresh calibration, subtype metrics and checkpoint alias are Pending.

Next model: SOL
Next phase: BTXRD migration Phase B — subtype-enabled HOSP training and fresh evaluation
Start command: `.venv\Scripts\python.exe scripts\17_validate_btxrd_subtype.py --device cuda`
Resume from: `PROJECT_STATE.md`

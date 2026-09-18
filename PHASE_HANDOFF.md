# Phase handoff

Phase completed: BTXRD post-training review and subtype-aware application integration

Important outputs:
- Plan: `docs/superpowers/plans/2026-09-18-btxrd-subtype-migration.md`
- Audit: `scripts/16_prepare_btxrd.py`; derived artifacts: `workspace/btxrd/`
- Separate schema/config: `configs/btxrd_label_schema.yaml`, `configs/hosp_btxrd_subtype.yaml`
- Wiring smoke: `scripts/17_validate_btxrd_subtype.py`
- Trainer/evaluator: `src/bone_cancer_ai/training/hosp_trainer.py`, `src/bone_cancer_ai/evaluation/hosp_btxrd.py`
- Runner: `scripts/18_train_btxrd_subtype.py` (isolated `experiments/btxrd_subtype`)
- Run: `experiments/btxrd_subtype/run-a8ae625f34c1`; best epoch 15; val AUROC 0.9588021106
- Selected alias: `hosp_bonenet.btxrd.subtype.selected`
- Review: `artifacts/metrics/btxrd_subtype_review.json`
- App: `app/streamlit_app.py` with original/preprocessed images, top-k subtype probabilities, conformal sets and provenance

Tests:
- PASS — focused migration/config/HOSP and subtype/app/review tests
- PASS — real BTXRD audit: 3,746/3,746 images, exact Main hash/path linkage, no missing/extra files
- PASS — duplicate-aware groups/splits; CPU and CUDA masked subtype forward/backward smoke
- PASS — source tree is read-only; frozen production artifacts unchanged

Outstanding issue:
- No clinical/pathology validation; explainability remains qualitative.

Next model: LUNA
Next phase: future subtype experiments or app review
Start command: `python launch_web_app.py`
Resume from: `PROJECT_STATE.md`

# Project state

Current phase: BTXRD post-training review and subtype-aware application integration COMPLETE
Completed phases: 1–10; migration Phases A–B
Next phase: future subtype experiments or app review

## Environment

- Python 3.11.15; PyTorch 2.14.0+cu130; torchvision 0.29.0+cu130; timm 1.0.29
- CUDA 13.0; NVIDIA GeForce RTX 4070 Laptop GPU, 8 GB
- Branch: `phase1-scaffold`

## Dataset

- Immutable source: `dataset/Main`
- Source: 12,557 images; 0=`normal` 6,827, 1=`bone_tumour` 5,730
- Source preservation: 12,560 files / 1,120,418,007 bytes; exact Phase 1 match after Phase 9
- Derived splits: fit 4,871; calibration 857; validation 1,453; test 1,882
- Leakage: zero record/sample/group overlap; split generation unchanged
- BTXRD: 3,746 images/metadata rows; 1,879 normal, 1,867 tumour; 1,866 valid subtype rows
- BTXRD linkage: 3,746/3,746 exact SHA-256 Main matches; 3,746/3,746 paths resolved to Main
- BTXRD derived splits: fit 2,222; calibration 389; val 606; test 529; 36 duplicate groups; 1 subtype conflict; 4 exact-label conflict rows masked
- Test use: four baselines evaluated in Phase 5; frozen Phase 8 HOSP evaluated once in Phase 9

## Frozen models and test results

| Model | AUROC | AUPRC | F1 |
|---|---:|---:|---:|
| DenseNet121 | 0.963149 | 0.965194 | 0.889888 |
| EfficientNet-B3 | 0.978097 | 0.976115 | 0.916239 |
| ConvNeXt-Tiny | 0.964013 | 0.963942 | 0.890519 |
| Swin-Tiny | 0.965165 | 0.963553 | 0.889160 |
| HOSP-BoneNet | 0.965033 | 0.965161 | 0.889512 |

- HOSP AUROC 95% bootstrap CI: 0.957058–0.972108; 1,000 resamples
- HOSP confusion matrix: TN=892, FP=131, FN=66, TP=793
- HOSP balanced accuracy/ECE/Brier: 0.897556 / 0.102005 / 0.101912
- Best test AUROC/AUPRC: EfficientNet-B3; no superiority claim for HOSP

## Uncertainty and explainability

- Split conformal: alpha 0.10; calibration n=857; test coverage 0.894261
- Prediction sets: 1,878 singleton, 4 empty/rejected, 0 ambiguous
- Galleries: four examples each for TP, TN, FP, FN and rejected cases
- Visual QA: complete with limitations; no inversion, broken maps or overlay misalignment
- Limitations: saturated lesion maps; some Grad-CAM focus on markers/borders/artifacts
- Frozen production concepts/open-set performance: Unavailable; BTXRD subtype metrics are recorded below

## Artifact reuse

- Selected alias: `hosp_bonenet.phase8.selected`
- Checkpoint SHA-256: `c24ed27c83c00aba245d2e28d7a39c3cd9fb027b311049625e3c915b0a711ec5`
- Test prediction SHA-256: `44b4befad823c55eb2ed0691713208560c88a1cedc2823ec0664a3b184fe564e`
- Test ledger evaluation count: 1; full config/schema/checksum contract required for reuse
- Phase 9 run: `experiments/evaluation/phase9-2a9cd712f0e1`

## Verification

- `pytest -q`: 482 passed; launcher/UI focused tests: 14 passed
- Ruff, compileall and `pip check`: PASS
- Frozen checkpoint reload and CUDA forward pass: PASS
- Real CUDA single-image CLI inference: PASS; JSON written to `logs/phase10_inference.json`
- Streamlit headless startup: PASS; HTTP 200 on local server
- Direct `launch_web_app.py` startup and two-tab results view: PASS
- Comparison and explanation commands: PASS; sealed outputs reused without rerunning test inference
- Required comparison/figures/failure gallery: generated and checksum-bound
- Exact completed rerun and QA finalizer: PASS; changed config fails closed
- Source preservation: PASS; exact Phase 1 snapshot match
- Independent review: CLEAN after provenance and wording fixes
- Phase 10: frozen predictor, CLI, Streamlit demo and final command wiring implemented
- BTXRD Phase A audit, exact linkage, duplicate-aware manifests and subtype mask smoke pass
- BTXRD Phase B trainer/evaluator tests: 3 focused + full suite 482 passed
- BTXRD Phase B: 20 epochs; best epoch 15; validation AUROC 0.9588021106
- BTXRD fresh test: binary AUROC 0.9653896010, AUPRC 0.9694396362, F1 0.9058823529
- BTXRD subtype test: n=262, accuracy 0.7213740458, macro F1 0.4933201773, AUROC 0.9134007955
- Conformal: binary coverage 0.8994307400 (calibration 389/test 527); subtype coverage 0.9007633588 (calibration 213/test 262)
- Post-training review: PASS; fresh protocol and registry/checkpoint fingerprints verified

## Important paths

- Config: `configs/evaluation.yaml`
- Runner: `scripts/12_evaluate.py`
- Core: `src/bone_cancer_ai/evaluation/phase9.py`
- Comparison: `artifacts/metrics/final_model_comparison.csv`
- Figures: `artifacts/figures/`
- Explanations: `artifacts/explainability/phase9/`
- Ledger: `artifacts/metrics/hosp_phase9_test_ledger.json`
- Inference config: `configs/inference.yaml`
- Inference CLI: `scripts/15_infer_image.py`
- Demo: `app/streamlit_app.py`
- Direct launcher: `launch_web_app.py`
- BTXRD audit: `scripts/16_prepare_btxrd.py`; artifacts: `workspace/btxrd/`
- BTXRD subtype validation: `scripts/17_validate_btxrd_subtype.py`; config: `configs/hosp_btxrd_subtype.yaml`
- BTXRD Phase B runner: `scripts/18_train_btxrd_subtype.py`; trainer/evaluator: `src/bone_cancer_ai/training/hosp_trainer.py`, `src/bone_cancer_ai/evaluation/hosp_btxrd.py`
- BTXRD run: `experiments/btxrd_subtype/run-a8ae625f34c1`; alias: `hosp_bonenet.btxrd.subtype.selected`
- BTXRD checkpoint SHA-256: `7e999e7d958be92297c137d82a2b7de5409b64918dcae9f1d01cd393e00fa8f0`
- Review: `artifacts/metrics/btxrd_subtype_review.json`; subtype predictor: `src/bone_cancer_ai/inference/btxrd.py`

## Open issues

- Explainability is qualitative and not clinically or spatially validated.
- Test calibration is imperfect; HOSP ECE is 0.102005.
- Multi-seed, external/OOD and clinical validation remain unavailable.
- BTXRD subtype output is research-only; no clinical/pathology validation claimed. Frozen binary test metrics were not reused.

## Next action

- Launch app with `python launch_web_app.py`; the image tab offers frozen binary and BTXRD subtype-aware inference.

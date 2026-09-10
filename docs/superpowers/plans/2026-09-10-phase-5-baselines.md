# Phase 5 Standard Comparison Baselines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and evaluate one fair, reproducible real run for DenseNet121, EfficientNet-B3, ConvNeXt-Tiny and Swin-Tiny through one configurable binary-classification pipeline.

**Architecture:** A small model registry creates a common `TimmClassifier` with one binary logit. A shared trainer consumes the verified Phase 3 DataModule, selects checkpoints using validation AUROC only, then evaluates the selected checkpoint once on the untouched test split. Run artifacts are immutable and a summary writer combines only executed metrics.

**Tech Stack:** Python 3.11, PyTorch 2.14, timm 1.0.29, TorchMetrics/scikit-learn, Pandas, PyYAML, CUDA AMP, pytest.

**Spec:** `BONE_CANCER_10_PHASE_BUILD_DIRECTIVE.md`, especially Sections 10–12 and Phase 5.

## Global Constraints

- Treat `E:\Bone Cancer\dataset\Main` as read-only; do not rewrite source images, source labels or Phase 3 split files.
- Use the verified `fit_train`, `val` and `test` manifests; leave `calibration` unused for model selection and threshold fitting in this phase.
- Binary semantics remain 0=`normal`, 1=`bone_tumour`; no filename-derived input or label is permitted.
- All models use image size 384, the same preprocessing/augmentation, binary loss, metrics, validation AUROC selection rule and 40-epoch maximum with patience 8.
- Defaults: batch 8, AdamW, learning rate 1e-4, weight decay 1e-4, cosine schedule, two warmup epochs, AMP, gradient clip 1.0 and seed 42.
- If memory fails, reduce per-device batch size and compensate with gradient accumulation before changing resolution; never silently fall back to CPU.
- Store only executed results. Undefined metrics use `null`; unavailable clinical verification remains `Unavailable`.
- Do not enable subtype, risk, concept or spatial heads. Do not train HOSP-BoneNet in Phase 5.

---

### Task 1: Common model registry and baseline wrapper

**Files:**
- Create: `configs/train.yaml`
- Create: `configs/baseline_densenet121.yaml`
- Create: `configs/baseline_efficientnet_b3.yaml`
- Create: `configs/baseline_convnext_tiny.yaml`
- Create: `configs/baseline_swin_tiny.yaml`
- Modify: `src/bone_cancer_ai/models/registry.py`
- Modify: `src/bone_cancer_ai/models/baselines/timm_classifier.py`
- Modify: `src/bone_cancer_ai/models/baselines/factory.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Produces: `BaselineSpec`, `BASELINE_REGISTRY`, `TimmClassifier(spec, pretrained, image_size)`, `create_baseline(name, pretrained, image_size)` and `count_trainable_parameters(model)`.
- Contract: all aliases return logits shaped `[B]`; only `swin_tiny` receives the constructor `img_size=384`; every final config points to the same common training config.

- [ ] Write tests that enumerate exactly four approved aliases, reject unknown names, instantiate all four without downloads, assert `[2]` finite logits at 384 pixels, and verify parameter counts are positive.
- [ ] Run `\.venv\Scripts\python.exe -m pytest tests/test_models.py -q` and confirm failure because the registry/wrapper are scaffolds.
- [ ] Implement the frozen alias-to-timm mapping and wrapper. Use timm model names available in 1.0.29: `densenet121`, `efficientnet_b3`, `convnext_tiny`, and `swin_tiny_patch4_window7_224` with the Swin image-size override.
- [ ] Add common/four model YAMLs. Each baseline config contains only model identity plus `common_config: train.yaml`; the common file owns every fairness-sensitive training/data/output setting.
- [ ] Re-run the focused tests, then run one CUDA AMP forward/backward for each architecture at batch 8 and record actual peak memory. Change only a model's batch/accumulation setting if this probe raises CUDA OOM.
- [ ] Commit the reviewed task.

### Task 2: Honest binary metrics and report artifacts

**Files:**
- Modify: `src/bone_cancer_ai/evaluation/classification.py`
- Modify: `src/bone_cancer_ai/evaluation/reports.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Produces: `binary_classification_metrics(logits, targets, threshold=0.5) -> dict`, `confusion_matrix_frame(metrics) -> DataFrame`, `measure_inference_latency(model, example, device, warmup=10, repetitions=50) -> dict`, and atomic JSON/CSV report writers.
- Metrics: AUROC, AUPRC, accuracy, balanced accuracy, precision, recall/sensitivity, specificity, F1, MCC, Brier score, ECE and a 2x2 confusion matrix. One-class AUROC/balanced metrics become `None` rather than fabricated values.

- [ ] Write deterministic tests against hand-computed perfect/imperfect predictions, confusion-matrix ordering `[[TN, FP], [FN, TP]]`, ECE/Brier bounds, one-class undefined behavior and CPU latency output.
- [ ] Run `\.venv\Scripts\python.exe -m pytest tests/test_metrics.py -q` and confirm expected failures from scaffold modules.
- [ ] Implement metrics from sigmoid probabilities using explicit validation of finite shapes/labels. Avoid hidden threshold tuning.
- [ ] Implement atomic report files and latency synchronization on CUDA.
- [ ] Re-run focused tests and commit the reviewed task.

### Task 3: Shared AMP trainer, checkpoint selection and resume-safe run records

**Files:**
- Modify: `src/bone_cancer_ai/training/amp.py`
- Modify: `src/bone_cancer_ai/training/optimizer.py`
- Modify: `src/bone_cancer_ai/training/scheduler.py`
- Modify: `src/bone_cancer_ai/training/callbacks.py`
- Modify: `src/bone_cancer_ai/training/checkpointing.py`
- Modify: `src/bone_cancer_ai/training/trainer.py`
- Test: `tests/test_training_smoke.py`

**Interfaces:**
- Produces: `class_weight_from_frame`, `build_optimizer`, `build_scheduler`, `EarlyStopping`, `save_checkpoint`, `load_checkpoint`, `BinaryTrainer.fit(train_loader, val_loader)` and `.predict(loader)`.
- `fit` returns epoch history plus selected best epoch/AUROC. `predict` returns finite CPU logits, binary targets and opaque sample IDs. Invalid samples/tasks are filtered and counted, never assigned a borrowed target.

- [ ] Write CPU synthetic tests for weighted BCE inputs, gradient accumulation, warmup/cosine learning rates, clipped finite updates, best/last checkpoint creation, early stopping, deterministic replay and rejection of a missing valid binary batch.
- [ ] Add a spy loader test proving `fit` sees only train/validation and test evaluation occurs only after loading the best checkpoint.
- [ ] Run `\.venv\Scripts\python.exe -m pytest tests/test_training_smoke.py -q` and confirm expected failures.
- [ ] Implement AMP with `torch.amp.autocast(device_type=...)` and `torch.amp.GradScaler`, AdamW, linear warmup plus cosine decay, gradient clipping and epoch-level JSON-safe history.
- [ ] Save `last.pt` every epoch and replace `best.pt` only when validation AUROC improves. Checkpoints include model alias, state dictionaries, epoch, selection metric and resolved configuration; loading validates the alias.
- [ ] Re-run focused tests and commit the reviewed task.

### Task 4: Baseline CLI, immutable run bundles and summary aggregation

**Files:**
- Modify: `scripts/09_train_baseline.py`
- Modify: `scripts/run_phase.ps1`
- Modify: `.vscode/tasks.json`
- Modify: `src/bone_cancer_ai/evaluation/reports.py`
- Test: `tests/test_baseline_cli.py`

**Interfaces:**
- CLI: `python scripts/09_train_baseline.py --config <baseline.yaml> [--epochs N] [--pretrained true|false] [--limit-batches N]`.
- A real default run uses pretrained weights and all fit/validation/test batches. Override flags exist only for smoke tests and are recorded in the resolved config/notes.
- Each run writes `config.yaml`, `environment.txt`, `git_commit.txt`, `metrics.json`, `history.csv`, `best.pt`, `last.pt`, `best_checkpoint_path.txt`, `confusion_matrix.csv`, `notes.md`, and `test_predictions.csv` under `experiments/baselines/<model>/<unique-run-id>/`.
- `artifacts/metrics/baseline_summary.csv` has the required columns and one latest valid row per model/seed/run.

- [ ] Write subprocess/config tests for invalid model/device, unique run directories, no test selection, required artifact completeness, atomic summary updates and failed-run status without fake metrics.
- [ ] Run the focused CLI tests and confirm the scaffold fails.
- [ ] Implement strict config resolution relative to config files, safe outputs under the project root, CUDA preflight, reproducibility capture and explicit CUDA OOM guidance.
- [ ] Implement train → load best → validation report → one test evaluation → latency → artifact/summary flow. Summary columns are exactly `model,seed,best_epoch,AUROC,AUPRC,balanced_accuracy,F1,sensitivity,specificity,MCC,params,latency_ms`, with additional metrics retained in each run JSON.
- [ ] Add one-command PowerShell/VS Code tasks for each baseline and a Phase 5 validation action.
- [ ] Re-run focused tests and commit the reviewed task.

### Task 5: Real four-model execution, review, verification and handoff

**Files:**
- Modify: `src/bone_cancer_ai/models/README.md`
- Modify: `src/bone_cancer_ai/models/baselines/README.md`
- Modify: `src/bone_cancer_ai/training/README.md`
- Modify: `src/bone_cancer_ai/evaluation/README.md`
- Modify: `experiments/README.md`
- Modify: `experiments/baselines/README.md`
- Modify: `artifacts/metrics/README.md`
- Modify: `PROJECT_GUIDE.md`
- Modify: `PROJECT_STATE.md`
- Modify: `CHANGELOG.md`
- Modify: `PHASE_HANDOFF.md`
- Modify: `PHASE_5_WORK_NOTES.md`

**Interfaces:**
- Consumes the four baseline configs through script 09.
- Produces four valid unique runs, four best checkpoints, a real summary CSV and exact Phase 5 handoff.

- [ ] Run each baseline with seed 42, pretrained weights, maximum 40 epochs and early stopping patience 8. Do not run models concurrently on the single 8 GB GPU.
- [ ] For each run verify CUDA device, AMP, actual effective batch, finite forward/backward, best/last checkpoint load, history, validation selection and final test artifacts before starting the next model.
- [ ] Independently review all Phase 5 code/results against fairness, leakage, checkpointing, reproducibility and no-fabrication requirements; fix Critical/Important findings with red-green tests.
- [ ] Run `pytest -q`, `ruff check .`, `python -m compileall -q src scripts app tools tests`, `python -m pip check`, and the Phase 5 artifact validator. Confirm all four summary rows point to valid real run bundles.
- [ ] Compare `dataset/Main` to `logs/source_after_phase4.json` and revalidate Phase 3 split fingerprints/counts. Save Phase 5 source/split verification evidence.
- [ ] Document actual model parameters, epochs, metrics, latency, limitations and executed commands. Mark multi-seed comparison Pending if only seed 42 ran; do not extrapolate clinical performance.
- [ ] Update the exact handoff format, commit the verified milestone and recommend SOL for Phase 6.

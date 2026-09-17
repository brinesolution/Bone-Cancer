# Phase 10 Integration and Final QA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Deliver a runnable frozen HOSP-BoneNet inference demo, repeatable PowerShell workflows, and final team documentation.

**Architecture:** `inference/predictor.py` owns checkpoint/config loading and single-image prediction. `inference/postprocess.py` owns JSON-safe result formatting and supported-output decisions. Streamlit calls those modules only after upload validation; it never reads ground-truth labels.

**Tech Stack:** Python 3.11, PyTorch, torchvision/timm, OpenCV/Pillow, NumPy, Streamlit, PyTest, Ruff.

**Spec:** `BONE_CANCER_10_PHASE_BUILD_DIRECTIVE.md`, Phase 10.

## Global Constraints

- `dataset/Main` remains read-only.
- Inference resolves `hosp_bonenet.phase8.selected` through the checksum registry.
- Binary labels are the only enabled verified output; subtype/concepts/OOD remain unavailable.
- Test artifacts and metrics are never used as inference ground truth.
- No fabricated metrics, annotations, or clinical claims.

### Task 1: Frozen Single-Image Inference

**Files:**
- Create: `src/bone_cancer_ai/inference/predictor.py`, `src/bone_cancer_ai/inference/postprocess.py`
- Modify: `configs/inference.yaml`, `src/bone_cancer_ai/inference/README.md`
- Test: `tests/test_inference_predictor.py`

- [ ] Write tests for upload validation, deterministic 384-pixel preprocessing, binary probability, conformal set/abstention, disabled optional outputs, and JSON serialization.
- [ ] Run the focused tests and confirm they fail because the interfaces are absent.
- [ ] Implement `FrozenPredictor.from_project_root()` and `.predict_array()` using the registry, frozen loader, `RadiographTransform`, `BinaryConformalCalibrator`, and Grad-CAM/spatial attention outputs.
- [ ] Implement `validate_upload()` and `result_to_json()` with finite-value and shape checks.
- [ ] Run focused tests, then a CUDA one-image smoke inference on a copied test image.

### Task 2: Streamlit Application

**Files:**
- Modify: `app/streamlit_app.py`, `app/ui_helpers.py`, `app/README.md`
- Test: `tests/test_streamlit_app.py`

- [ ] Write tests for app import, image validation, supported-output gating, disclaimer text, and JSON export helper.
- [ ] Run them red.
- [ ] Implement upload, validation, inference, binary/confidence/abstain display, attention/Grad-CAM figures, optional-output messages, JSON download, and research-use disclaimer.
- [ ] Run headless import/AppTest or a Streamlit smoke command without exposing labels.

### Task 3: One-Command Workflows and Documentation

**Files:**
- Modify: `scripts/run_phase.ps1`, `.vscode/tasks.json`, `README.md`, `PROJECT_GUIDE.md`, `PROJECT_STATE.md`, `PHASE_HANDOFF.md`, `CHANGELOG.md`
- Create or update concise `README.md` files for active major folders.

- [ ] Add verify, audit, annotation, baseline, HOSP, evaluate, app, and test actions.
- [ ] Add matching VS Code tasks and exact fresh-terminal commands.
- [ ] Replace Phase 10 pending text with implemented inference/demo behavior and limitations.

### Task 4: Final Verification and Handoff

- [ ] Run `pytest -q`, `ruff check .`, and `python -m compileall src scripts app tools`.
- [ ] Run environment verification and one real single-image inference using a copied image, preserving `dataset/Main`.
- [ ] Launch Streamlit headlessly or through AppTest and record the result.
- [ ] Request independent review, fix findings, update state/handoff, and commit the verified Phase 10 change.

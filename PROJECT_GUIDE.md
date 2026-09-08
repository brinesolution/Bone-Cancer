# Project guide

Phase 1 is verified: 31 tests, dependency/CUDA checks and source-preservation comparison passed. Future-phase descriptions below are the approved design, not completed functionality.

## 1. Project Overview

HOSP-BoneNet is the single proposed model. Ten acceptance-gated phases build this repository. Phase 1 provides environment and infrastructure; all training and model metrics are Not run.

## 2. Problem Definition

Input: a musculoskeletal radiograph. Minimum prediction: binary classification under a verified schema. Subtype, risk, concepts and supervised localization require actual labels. This is a research prototype, not autonomous clinical diagnosis.

## 3. What the Dataset Contains

Filesystem inventory: train has 8,789 JPG/JPEG files and one CSV; val and test each have 1,884 JPG/JPEG files and one CSV. These are file counts, not validated image/label counts. Semantics, row matching, quality and grouping remain Pending Phase 2. Other dataset directories include BTXRD and exported archives; do not assume they share semantics or merge them without the audit.

## 4. Folder Structure

`src/bone_cancer_ai` owns reusable logic; `scripts` orchestrates; `configs` holds decisions. `dataset/Main` is immutable. `workspace` stores local derived metadata/cache; `experiments` stores unique runs; `artifacts` stores outputs; `logs` stores evidence. Annotation tools, app and notebooks are inactive scaffolds. Every authored meaningful folder has a README; none is inserted into source data.

## 5. Environment Setup

See requirements/README.md for commands and official sources. Python 3.11.15 is installed through uv. The .venv uses torch 2.14.0+cu130 and torchvision 0.29.0+cu130. The RTX 4070 Laptop GPU has 8 GB VRAM. Use the project interpreter rather than default Python 3.12/3.13. Freeze only after verification.

## 6. Data Audit

Pending Phase 2. Index every image, map every label row, report missing/extra records, decode failures, dimensions, modes and class counts. Verify 0/1 meanings from authoritative source documentation. Phase 1 hashes establish immutability; they do not constitute the image audit or duplicate report.

## 7. Duplicate and Leakage Prevention

Pending Phase 2/3. Compare SHA-256, perceptual hashes and canonical source IDs, including augmentation copies. Correct splits through manifests without moving files. Keep each duplicate/source/patient group in one fit/calibration/val/test partition. Patient independence requires reliable patient identifiers.

## 8. Preprocessing

Pending Phase 3. Decode safely, remove alpha, consistently form three channels, preserve aspect ratio, pad square, resize to configured 384 pixels and normalize for pretrained encoders. Percentile clipping needs validation. Cache derived images with traceability to originals.

## 9. Data Augmentation

Pending Phase 3. Mild rotations (up to 7 degrees), translation and intensity changes are intended. Horizontal and vertical flips are disabled. Severe geometry/color changes are unsuitable defaults. CLAHE is a validated ablation only.

## 10. Annotation Workflow

Pending Phase 4. Record image ID, labels including unknown, reviewer, timestamp, notes and revisions. Use reliable spatial tooling for boxes/masks and verify alignment and reload/edit/export. Project annotations are not radiologist/pathology ground truth without real verification.

## 11. DataLoader Design

Pending Phase 3. Load manifest-approved images and return tensors, labels, validity masks and separate metadata. Filenames never enter predictive tensors. Mask missing labels and align image/lesion transforms. Windows workers start at zero until multiprocessing is validated.

## 12. Baseline Models

Pending Phase 5. DenseNet121 connects earlier features to later blocks; EfficientNet-B3 scales depth/width/resolution; ConvNeXt-Tiny is a modern convolutional encoder; Swin-Tiny uses hierarchical windowed attention. One timm classifier and common training/evaluation enforce fair comparison. Verify Swin resolution/window compatibility during implementation.

## 13. HOSP-BoneNet Architecture

Pending Phase 6. A shared ConvNeXt-Tiny encoder feeds global pooling, a lesion head, mask-weighted local pooling, gated fusion and supported optional concept/hierarchical heads. Calibrate open-set and conformal postprocessing on allowed partitions only.

## 14. Encoder

Pending Phase 6. timm will expose multiscale spatial features. Early maps retain detail; later maps represent broader structure. Obtain channels and strides from the actual model API.

## 15. Localization Head

Pending Phase 6. A lightweight feature pyramid produces lesion logits [B,1,H,W]. Masks permit Dice plus BCE/focal supervision; boxes provide weak supervision. Without spatial labels, use only weak attention/consistency and claim no segmentation accuracy.

## 16. Mask-Weighted Local Pooling

Pending Phase 6. Average spatial encoder features weighted by lesion probabilities, protecting a small denominator. This focuses a vector on a learned region; it does not prove lesion correctness.

## 17. Concept Bottleneck

Pending Phase 6. Pool per-concept activation maps into scores supervised by verified concept labels. Unavailable concepts remain disabled. Interpretation is limited by annotation coverage and validity.

## 18. Global-Local Fusion

Pending Phase 6. A learned gate combines vectors: fused = gate * local + (1 - gate) * global. The global path retains context; the local path emphasizes the candidate lesion. Supported concept features may be concatenated.

## 19. Hierarchical Prediction

Pending Phase 6. Binary output becomes mandatory after label verification. Subtype/risk heads need verified classes. Consistency losses require an explicit verified class relationship; filenames cannot provide it.

## 20. Open-Set Recognition

Pending supported subtype labels. Compute energy from logits and calibrate rejection on permitted validation/calibration data. Leave-one-subtype-out evaluation requires enough classes and samples.

## 21. Conformal Uncertainty

Pending calibration data and trained models. Hold out a grouped subset of official training data. Calibration scores define prediction sets; test labels never choose thresholds. Report coverage and set size with distribution-shift limitations.

## 22. Loss Functions

Pending Phase 5/6. BCE/cross-entropy penalizes classification errors; focal loss emphasizes hard examples; Dice measures mask overlap; grounding penalizes concept activation outside verified regions. Sum supported losses with missing-label masks. Unavailable supervision supplies no invented targets.

## 23. Training Loop

Pending Phase 5/7. Defaults: 40 epochs, AdamW at 1e-4, weight decay 1e-4, cosine schedule, two warmup epochs, AMP, gradient clipping and early stopping. Verify CUDA and a synthetic forward/backward first. Reduce batch size then accumulate gradients before reducing resolution on OOM. Never silently run long training on CPU.

## 24. Checkpointing

Pending Phase 5/7. Unique runs record config, environment, Git revision, history and checkpoint paths. Save best.pt and last.pt without overwriting another run. Select using predefined validation metrics, never test performance.

## 25. Evaluation Metrics

Pending predictions. AUROC assesses ranking; AUPRC emphasizes positive retrieval; balanced accuracy averages recalls; F1 balances precision/recall; MCC summarizes the confusion matrix. Sensitivity/specificity measure positive/negative recall. Dice/IoU require spatial labels; Brier/ECE assess calibration. Clinical class names require verified meanings.

## 26. Explainability

Pending Phase 9. Generate lesion probabilities, Grad-CAM/Grad-CAM++ and supported concept overlays. Inspect anatomy, padding, burned-in text, failures and uncertainty. Grad-CAM is an explanation, never ground-truth segmentation.

## 27. Inference Pipeline

Pending Phase 9/10. Validate input, apply frozen preprocessing, load a selected checkpoint, predict supported heads, calibrate/reject and return results/overlays. Work without labels and expose no ground truth. No checkpoint exists yet.

## 28. Streamlit Demo

Pending Phase 10. Upload, validate, predict, show calibrated confidence/abstention and supported outputs, export JSON and display research-use limitations. The current entry point reports its pending phase.

## 29. How to Run Each Phase

Phase 1: `python scripts/00_verify_environment.py`, `pytest -q`, `ruff check .`, `python -m compileall -q src scripts app tools`. PowerShell: `./scripts/run_phase.ps1 -Action verify` or `-Action test`. Later commands are in master directive Section 22 and remain inactive. Stop after Phase 1 and manually select SOL for Phase 2.

## 30. Common Errors and Troubleshooting

For initial creation when `py -3.11` cannot resolve uv Python, use `uv venv --python 3.11 --seed .venv`. On import failure, activate .venv and install editable. On CUDA failure, record nvidia-smi and torch.version.cuda; do not start CPU training. Configs must be mappings without duplicate keys. Output paths cannot escape the project or overlap source data. Deterministic settings do not promise bitwise portability across hardware/software.

## 31. Team Review Study Notes

Member 1: data and leakage. Member 2: annotation/localization. Member 3: baselines/training. Member 4: HOSP-BoneNet/optimization. Study folder READMEs before changing interfaces. Installed libraries do not establish implemented features. Phase 1 uses pathlib/hashlib/logging, PyYAML, NumPy and torch utilities. Other packages are import-verified infrastructure for later phases; document actual APIs as they become used.

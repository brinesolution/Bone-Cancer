# Project guide

Phases 1 and 2 establish the environment and audited data inventory. The audit executed against every Main image; Phase 3 and later descriptions below remain approved design rather than completed functionality.

## 1. Project Overview

HOSP-BoneNet is the single proposed model. Ten acceptance-gated phases build this repository. Phase 1 provides environment and infrastructure; all training and model metrics are Not run.

## 2. Problem Definition

Input: a musculoskeletal radiograph. The verified source schema defines 0=`normal` and 1=`bone_tumour`. These are dataset meanings, not a clinical finding independently confirmed for every image. Partial subtype/risk/spatial metadata needs validity masks and Phase 4 validation. This is a research prototype, not autonomous clinical diagnosis.

## 3. What the Dataset Contains

The audited Main dataset contains 12,557 labeled images: train 8,789, validation 1,884 and test 1,884. Class 0 (`normal`) has 6,827 images and class 1 (`bone_tumour`) has 5,730. All label rows map to files and all files decode. Stored formats are 8,838 `.jpg` plus 3,719 `.jpeg`; 2,895 decode as grayscale and 9,662 as RGB, of which 9,109 have effectively identical channels. Resolution varies widely. Two low-exposure images are conservatively flagged `near_blank` but retain visible anatomy.

Separate local metadata maps one-to-one by canonical source ID across all Main records and establishes the binary names without parsing diagnostic filename words. Every Main canonical group has an exact source-image SHA-256 anchor. BTXRD supplies benign/malignant risk metadata for 1,867 tumour images and unambiguous subtype metadata for 1,866. It also has 1,867 images with box and polygon source annotations. One workbook row activates two subtype columns; its subtype remains blank and the conflict is explicitly reported.

## 4. Folder Structure

`src/bone_cancer_ai` owns reusable logic; `scripts` orchestrates; `configs` holds decisions. `dataset/Main` is immutable. `workspace` stores local derived metadata/cache; `experiments` stores unique runs; `artifacts` stores outputs; `logs` stores evidence. Annotation tools, app and notebooks are inactive scaffolds. Every authored meaningful folder has a README; none is inserted into source data.

## 5. Environment Setup

See requirements/README.md for commands and official sources. Python 3.11.15 is installed through uv. The .venv uses torch 2.14.0+cu130 and torchvision 0.29.0+cu130. The RTX 4070 Laptop GPU has 8 GB VRAM. Use the project interpreter rather than default Python 3.12/3.13. Freeze only after verification.

## 6. Data Audit

Scripts 01–03 index, validate and group the dataset. `dataset_master.csv` includes stable record/canonical IDs, original absolute paths, split, dimensions, channels, byte/perceptual hashes, labels, partial-supervision flags and pixel-quality measurements. `quality_report.csv` isolates review fields, while `label_audit.csv` retains structural/provenance issues. No source image is altered, normalized or recompressed. Empty labels remain empty rather than guessed.

## 7. Duplicate and Leakage Prevention

Phase 2 found 2,045 connected duplicate candidate groups covering 6,968 images and 13,225 pair edges. Of those, 6,112 edges cross official splits, including 508 exact-byte cross-split edges. Evidence includes 1,080 SHA-256 matches, 11,016 canonical-ID matches and 4,300 pHash-distance candidates; an edge may contain more than one evidence type. Six candidate pairs have binary-label conflicts, including two exact-byte conflicts. Perceptual matches remain review candidates and are not automatically deleted.

Phase 3 must produce corrected manifest-only fit/calibration/validation/test assignments and keep every connected group in one partition. Source folders stay untouched. No patient or study ID column exists, so canonical/duplicate grouping is the strongest available independence control; this limitation must remain visible in evaluation claims.

## 8. Preprocessing

Pending Phase 3. Decode safely, remove alpha, consistently form three channels, preserve aspect ratio, pad square, resize to configured 384 pixels and normalize for pretrained encoders. Percentile clipping needs validation. Cache derived images with traceability to originals.

## 9. Data Augmentation

Pending Phase 3. Mild rotations (up to 7 degrees), translation and intensity changes are intended. Horizontal and vertical flips are disabled. Audit previews show mixed stored orientation, photographic film captures and burned-in text/markers, so Phase 3 must test orientation handling and prevent text/background shortcuts. Severe geometry/color changes are unsuitable defaults. CLAHE is a validated ablation only.

## 10. Annotation Workflow

Pending Phase 4. Existing BTXRD boxes/polygons and subtype/risk metadata are available for 1,867 images but remain disabled until coordinate conversion, overlays and source conflicts are validated. Record image ID, labels including unknown, reviewer, timestamp, notes and revisions. Project annotations are not radiologist/pathology ground truth without real verification.

## 11. DataLoader Design

Pending Phase 3. Load only manifest-approved images and return tensors, labels, validity masks and separate metadata. Filenames and source paths never enter predictive tensors. Mask the 10,690 records without subtype/risk/spatial supervision and align image/lesion transforms. Windows workers start at zero until multiprocessing is validated.

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

Phase 1 environment: `python scripts/00_verify_environment.py`. Phase 2 data audit: run `python scripts/01_index_dataset.py`, `python scripts/02_audit_dataset.py`, and `python scripts/03_find_duplicates.py` in that order, or `./scripts/run_phase.ps1 -Action phase2`. Validation: `pytest -q`, `ruff check .`, and `python -m compileall -q src scripts app tools`. Scripts 04–15 remain inactive until their phases.

## 30. Common Errors and Troubleshooting

For initial creation when `py -3.11` cannot resolve uv Python, use `uv venv --python 3.11 --seed .venv`. On import failure, activate .venv and install editable. Configs must be mappings without duplicate keys, and outputs cannot overlap source data. A script-01 structural error means inspect `label_audit.csv`; do not weaken validation. Script 02 exits 2 when binary semantics cannot be established. Duplicate counts are not a script error because Phase 3 owns exclusions/split repair. Two `near_blank` results are review flags, not automatic exclusions. Deterministic settings do not promise bitwise portability across hardware/software.

## 31. Team Review Study Notes

Member 1: study manifest contracts, SHA/pHash/canonical evidence and Phase 3 grouping. Member 2: study the BTXRD workbook, the one subtype conflict and Phase 4 spatial validation. Member 3: keep all baselines on the same future clean manifest. Member 4: keep HOSP-BoneNet optional heads masked/disabled according to coverage. Phase 2 uses pathlib/hashlib, Pillow, ImageHash, NumPy, Pandas, Matplotlib, PyYAML and openpyxl. Audit figures show data properties only; no model metric, radiologist review or clinical verification has been produced.

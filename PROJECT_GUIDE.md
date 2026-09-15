# Project guide

Phases 1–7 establish the environment, audited data, leakage-controlled loaders, annotation workflow, four baselines, the HOSP-BoneNet architecture, and its first complete validation-selected training run.

## 1. Project Overview

HOSP-BoneNet is the single proposed model. Ten acceptance-gated phases build this repository; the four Phase 5 baselines provide real seed-42 references before HOSP-BoneNet implementation.

## 2. Problem Definition

Input: a musculoskeletal radiograph. The verified source schema defines 0=`normal` and 1=`bone_tumour`. These are dataset meanings, not a clinical finding independently confirmed for every image. Partial subtype/risk/spatial metadata needs validity masks and Phase 4 validation. This is a research prototype, not autonomous clinical diagnosis.

## 3. What the Dataset Contains

The audited Main dataset contains 12,557 labeled images: train 8,789, validation 1,884 and test 1,884. Class 0 (`normal`) has 6,827 images and class 1 (`bone_tumour`) has 5,730. All label rows map to files and all files decode. Stored formats are 8,838 `.jpg` plus 3,719 `.jpeg`; 2,895 decode as grayscale and 9,662 as RGB, of which 9,109 have effectively identical channels. Resolution varies widely. Two low-exposure images are conservatively flagged `near_blank` but retain visible anatomy.

Separate local metadata maps one-to-one by canonical source ID across all Main records and establishes the binary names without parsing diagnostic filename words. Every Main canonical group has an exact source-image SHA-256 anchor. BTXRD supplies benign/malignant risk metadata for 1,867 tumour images and unambiguous subtype metadata for 1,866. It also has 1,867 images with box and polygon source annotations. One workbook row activates two subtype columns; its subtype remains blank and the conflict is explicitly reported.

## 4. Folder Structure

`src/bone_cancer_ai` owns reusable logic; `scripts` orchestrates; `configs` holds decisions. `dataset/Main` is immutable. `workspace` stores local metadata/cache/annotation revisions; `experiments` stores unique runs; `artifacts` stores outputs; `logs` stores evidence. `tools/annotation_app` is active; the inference app and notebooks remain scaffolds. Every major folder has a README; none is inserted into source data.

## 5. Environment Setup

See requirements/README.md for commands and official sources. Python 3.11.15 is installed through uv. The .venv uses torch 2.14.0+cu130 and torchvision 0.29.0+cu130. The RTX 4070 Laptop GPU has 8 GB VRAM. Use the project interpreter rather than default Python 3.12/3.13. Freeze only after verification.

## 6. Data Audit

Scripts 01–03 index, validate and group the dataset. `dataset_master.csv` includes stable record/canonical IDs, original absolute paths, split, dimensions, channels, byte/perceptual hashes, labels, partial-supervision flags and pixel-quality measurements. `quality_report.csv` isolates review fields, while `label_audit.csv` retains structural/provenance issues. No source image is altered, normalized or recompressed. Empty labels remain empty rather than guessed.

## 7. Duplicate and Leakage Prevention

Phase 2 found 2,045 connected duplicate candidate groups covering 6,968 images and 13,225 pair edges. Of those, 6,112 edges cross official splits, including 508 exact-byte cross-split edges. Evidence includes 1,080 SHA-256 matches, 11,016 canonical-ID matches and 4,300 pHash-distance candidates; an edge may contain more than one evidence type. Six candidate pairs have binary-label conflicts, including two exact-byte conflicts. Perceptual matches remain review candidates and are not automatically deleted.

Phase 3 produced corrected manifest-only fit/calibration/validation/test assignments and keeps every connected group in one partition. Source folders remain untouched. For a group crossing official splits, records in test have retention priority, then validation, then train; 3,490 lower-priority copies are excluded. Four records across two exact-byte label-conflict groups are also excluded. Three pHash-only conflict groups remain together and flag 37 records for Phase 4 review. No patient or study ID exists, so duplicate grouping is the strongest available independence control; this limitation must remain visible in evaluation claims.

## 8. Preprocessing

`decode_radiograph` applies stored EXIF orientation in memory, converts grayscale/RGB/RGBA sources to grayscale-replicated RGB, and never rewrites originals. `resize_and_pad` uses one scale factor, central black padding and 384×384 output; masks use the same geometry with nearest-neighbour interpolation. `RadiographTransform` applies ImageNet normalization. A 16-image optional cache smoke test wrote deterministic opaque PNGs with source/cache hashes and a configuration fingerprint. The DataLoader reads source paths from manifests; cache use is not silently enabled.

## 9. Data Augmentation

Fit-only transforms use up to ±7° rotation, 3% translation, 5% scale, mild brightness/contrast and gamma, plus low Gaussian noise. Horizontal/vertical flips and random crops are absent. Calibration, validation and test transforms are deterministic. The Phase 3 batch grid shows recognizable anatomy, preserved aspect ratio and expected padding. Burned-in text, markers, film captures and mixed stored orientation remain shortcut risks for later evaluation. CLAHE and percentile clipping remain disabled.

## 10. Annotation Workflow

Implemented in Phase 4. Script07 launches local metadata review with source image/ID, unknown-aware binary/subtype/risk/concepts, actual reviewer, notes, autosave, revision history, navigation, progress and CSV export. SQLite uses optimistic revisions to reject stale overwrites. Concept unknown is separate from assessed-absent. No review is inferred from filenames or source labels.

LabelMe edits opaque hash-bound copies. Manual polygon/box JSONs import into immutable spatial revisions. Native-resolution polygons rasterize as a 0/1 union; rectangles remain xyxy boxes without fabricated masks. Geometry and overlay checks preserve coordinate alignment. Script08 separately converted all 1,867 BTXRD JSONs: 1,825 unique Main-image records validate, 16 conflicting drawings across eight exact-image groups are quarantined, and 26 have no eligible exact match. One accepted subtype remains unknown. The first rejected conversion is retained as failure evidence; current indexes reference the accepted generation.

Exact bytes alone do not establish drawing agreement: duplicate source JSONs must agree before reuse. Latest indexes are checked against immutable generations, original hashes, metadata, masks and boxes. Human project reviews remain zero; test fixtures are isolated. Source mechanical validation and technical overlay QA are not radiologist/pathology verification. Optional tasks remain disabled until accepted labels are explicitly joined to unchanged fit/calibration/validation/test assignments and missing targets retain validity masks. See [the complete workflow](workspace/annotations/README.md).

## 11. DataLoader Design

`BoneCancerDataModule` reads only `fit_train.csv`, `calibration.csv`, `val.csv`, and `test.csv`. Executed counts are 4,871; 857; 1,453; and 1,882. A batch returns image `[B,3,384,384]`, binary/subtype/risk targets, concepts, mask `[B,1,384,384]`, nested `valid_task_mask`, and opaque metadata. Source paths, filenames, record IDs and canonical IDs are absent from the top level. Binary labels are valid; subtype, risk, concepts and masks remain invalid until their schema gates open. A runtime decode error yields a zero tensor with every task mask false and a logged error type, preventing a changed/corrupt file from crashing an epoch or borrowing another row's label. Windows workers default to zero; optional weighted sampling is seeded.

## 12. Baseline Models

Phase 5 uses one `TimmClassifier` and registry for DenseNet121, EfficientNet-B3, ConvNeXt-Tiny and Swin-Tiny. All use the same 384-pixel inputs, manifests, augmentation, loss, metrics, seed 42, validation-AUROC selection and effective batch 8. ConvNeXt alone uses measured microbatch 4 with accumulation 2 after two recorded batch-8 host-memory failures; the others use batch 8 directly.

## 13. HOSP-BoneNet Architecture

`HOSPBoneNet` uses a shared ConvNeXt-Tiny encoder, global pooling, a lesion FPN, mask-weighted local pooling and gated fusion. A strict factory derives optional output sizes from enabled label-schema sections. The production config enables verified binary supervision only.

## 14. Encoder

`TimmFeatureEncoder` exposes the backbone's discovered multiscale channels and reductions. ConvNeXt-Tiny supplies four stages at strides 4/8/16/32; the deepest map is projected into the configured feature dimension.

## 15. Localization Head

`LocalizationHead` fuses lateral multiscale features top-down and returns lesion logits `[B,1,H,W]` aligned to the input. Current mask loss weight is zero, so these maps are classification-guided attention and have no claimed segmentation accuracy.

## 16. Mask-Weighted Local Pooling

`MaskWeightedPool` resizes lesion probabilities to the deepest feature map and computes a protected FP32 weighted mean. It focuses local features but does not establish lesion correctness.

## 17. Concept Bottleneck

`ConceptHead` creates per-concept activation maps and pooled logits only when the schema and config enable verified concepts. The branch is currently disabled.

## 18. Global-Local Fusion

`GatedFusion` implements `gate * local + (1 - gate) * global`. Enabled concept probabilities are appended before the prediction heads.

## 19. Hierarchical Prediction

The binary head is mandatory. Subtype/risk heads appear only when enabled labels supply at least two classes. Hierarchy consistency requires an explicit subtype-to-binary relationship; its production weight remains zero.

## 20. Open-Set Recognition

Pending supported subtype labels. Compute energy from logits and calibrate rejection on permitted validation/calibration data. Leave-one-subtype-out evaluation requires enough classes and samples.

## 21. Conformal Uncertainty

Pending calibration data and trained models. Hold out a grouped subset of official training data. Calibration scores define prediction sets; test labels never choose thresholds. Report coverage and set size with distribution-shift limitations.

## 22. Loss Functions

`HOSPLoss` combines masked binary, subtype, risk, Dice+BCE segmentation, concept, grounding and hierarchy terms. Validity masks exclude missing targets; a requested task with valid labels but no output fails. Large AMP reductions use FP32 to prevent NaNs.

## 23. Training Loop

The shared trainer uses AdamW, cosine scheduling, warmup, AMP, gradient clipping and validation-AUROC checkpointing. Phase 7 ran binary Stage A for 35 epochs with early stopping, skipped B/C because their verified labels are unavailable, then fine-tuned Stage D for 10 epochs at 1e-5. Only fit_train and validation were constructed; test was not accessed.

## 24. Checkpointing

Each immutable run records resolved config, environment, Git state, history, `best.pt`, `last.pt`, predictions and metrics. Checkpoints validate model/config identity before mutation and support exact resume. The selected Phase 7 artifact is Stage D epoch 6 (validation AUROC 0.958355); fresh reload reproduced logits, targets and IDs exactly. Test evaluation remains reserved for Phase 9.

## 25. Evaluation Metrics

Seed-42 test results are: DenseNet121 AUROC 0.96315/AUPRC 0.96519; EfficientNet-B3 0.97810/0.97612; ConvNeXt-Tiny 0.96401/0.96394; Swin-Tiny 0.96517/0.96355. See `artifacts/metrics/baseline_summary.csv` for balanced accuracy, F1, sensitivity, specificity, MCC, parameters and measured latency. These are single-split research results; multi-seed and clinical validation are Pending/Unavailable.

## 26. Explainability

Pending Phase 9. Generate lesion probabilities, Grad-CAM/Grad-CAM++ and supported concept overlays. Inspect anatomy, padding, burned-in text, failures and uncertainty. Grad-CAM is an explanation, never ground-truth segmentation.

## 27. Inference Pipeline

Baseline checkpoint loading and prediction are implemented. The HOSP-BoneNet inference, calibration, rejection and overlay pipeline remains pending Phase 9/10.

## 28. Streamlit Demo

Pending Phase 10. Upload, validate, predict, show calibrated confidence/abstention and supported outputs, export JSON and display research-use limitations. The current entry point reports its pending phase.

## 29. How to Run Each Phase

Phase 1 environment: `python scripts/00_verify_environment.py`. Phase 2 audit: scripts 01–03. Phase 3 data: scripts 04–06. Phase 4 annotations: scripts 07–08. Phase 5 baselines: script 09. Phase 7 HOSP: `python scripts/10_train_hosp_bonenet.py --config configs/hosp_bonenet.yaml`; validate an existing run with `--validate-bundle <run_dir>`. Always use `.venv` and run pytest, Ruff and compileall after changes.

## 30. Common Errors and Troubleshooting

For initial creation when `py -3.11` cannot resolve uv Python, use `uv venv --python 3.11 --seed .venv`. On import failure, activate `.venv` and install editable; the system Python is not the project interpreter. Configs must be mappings without duplicate keys, and outputs cannot overlap source data. A script-01 structural error means inspect `label_audit.csv`; do not weaken validation. Script 02 exits 2 when binary semantics cannot be established. If script 04/06 changes counts, rerun Phase 2 first when source/audit evidence changed, then inspect clean/split summaries rather than editing CSVs. Two `near_blank` records remain eligible review flags. Deterministic settings reproduce assignments within the pinned environment but do not promise bitwise portability across other hardware/software.

## 31. Team Review Study Notes

Member 1: study duplicate grouping and split invariants. Member 2: study Phase 4 annotation validity and conflicts. Member 3: study preprocessing, the shared baseline trainer and saved metrics. Member 4: keep HOSP-BoneNet heads gated by verified coverage and reserve calibration for uncertainty work. Phase 5 metrics are executed model results; radiologist/pathology verification remains Unavailable.

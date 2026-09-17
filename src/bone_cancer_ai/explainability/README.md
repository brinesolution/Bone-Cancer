# src/bone_cancer_ai/explainability

## Purpose
Spatial debugging overlays and outcome/uncertainty galleries.

## Files
- `gradcam.py` — class-specific activation maps.
- `overlays.py` — denormalization and heatmap blending.
- `gallery.py` — deterministic TP/TN/FP/FN/rejected selection.

## Inputs
frozen model/images and supported outputs.

## Outputs
Phase 9 galleries under `artifacts/explainability/phase9/`.

## Main Libraries
PyTorch, NumPy, Pandas and Matplotlib.

## Important Functions / Classes
`GradCAM`, `overlay_heatmap`, `select_gallery_cases`.

## Theory
Attention is not ground truth; inspect text-marker shortcuts.

## Execution
`.venv\Scripts\python.exe scripts\12_evaluate.py --config configs\evaluation.yaml`

## Notes
Maps are qualitative only; Phase 9 found marker/border attention and saturated lesion maps.

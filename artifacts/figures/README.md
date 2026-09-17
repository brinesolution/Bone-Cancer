# artifacts/figures

## Purpose
Audit/evaluation figures.

## Files
Audit and preprocessing figures remain alongside Phase 9 ROC, PR, calibration and per-model confusion-matrix figures.

## Inputs
Audited manifests and one batch from each Phase 3 split.

## Outputs
Executed audit, preprocessing and frozen-test evaluation figures.

## Main Libraries
Matplotlib, Pillow and NumPy.

## Important Functions / Classes
`save_audit_figures`, `smoke_test_loaders`, and `plot_final_comparison` own outputs.

## Theory
The Phase 3 grid denormalizes tensors for display and uses opaque titles. Visual review confirmed recognizable anatomy, grayscale consistency and expected aspect padding across all four splits. Burned-in markers/capture artifacts remain a documented shortcut risk.

## Execution
Run script 02 for audit plots, script 06 for loader QA, and script 12 for Phase 9 plots.

## Notes
Do not edit generated figures or present technical visual QA as clinical review.

Phase 4 adds phase4_annotation_overlay_grid.png, assembled from 12 accepted source-conversion overlays. Red fills are native-coordinate polygon unions; yellow rectangles are source boxes. Three full overlays and the grid were inspected for alignment. Conversion uses annotations.spatial.render_overlay; the local grid command is `python logs/phase4_make_qa_grid.py`, using the preserved import report. This is technical QA only, with no model predictions or clinical verification.

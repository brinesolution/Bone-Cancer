# Baseline Experiments

Purpose: Real Phase 5 runs for the four approved reference models.

Folders:
- `densenet121/`
- `efficientnet_b3/`
- `convnext_tiny/`
- `swin_tiny/`

Each final run uses seed 42, 384-pixel inputs and validation-AUROC selection. ConvNeXt uses the recorded 4×2 memory exception; every effective batch equals 8.

Summary: `artifacts/metrics/baseline_summary.csv`.

Validate: `.\scripts\run_phase.ps1 -Action phase5`

Failed/smoke runs are evidence only and cannot populate the final summary.

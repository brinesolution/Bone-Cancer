# Baseline Models

Purpose: One configurable binary classifier for all four fair Phase 5 comparisons.

Files:
- `timm_classifier.py` — `TimmClassifier` with one binary logit.
- `factory.py` — validated construction and parameter counting.

Models: DenseNet121, EfficientNet-B3, ConvNeXt-Tiny, Swin-Tiny.

Main libraries: PyTorch, timm.

Flow: registry specification → timm backbone → `[B]` logits.

Run: `\.venv\Scripts\python.exe scripts\09_train_baseline.py --config configs\baseline_<model>.yaml`

Model aliases and output shape are fixed by tests; do not fork architecture-specific trainers.

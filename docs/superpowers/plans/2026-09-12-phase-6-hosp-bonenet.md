# Phase 6 HOSP-BoneNet Implementation Plan

**Goal:** Implement and verify the directive-defined HOSP-BoneNet before full training.

## Constraints

- Default backbone: pretrained `convnext_tiny`; source dataset remains read-only.
- Production config enables verified binary supervision only.
- Subtype, risk, concept and mask losses require explicit outputs plus validity masks.
- Phase 6 performs unit/CUDA checks and a 16-sample overfit, not full training.

## Design

- `encoder.py`: timm multi-stage feature encoder with discovered channel/stride metadata.
- `localization_head.py`: lightweight FPN and input-aligned lesion logits.
- `pooling.py`: global average/GeM and mask-weighted local pooling.
- `concept_head.py`: per-concept spatial maps and pooled logits.
- `fusion.py`: learned elementwise global/local gate.
- `hierarchy.py`: mandatory binary and configured subtype/risk heads.
- `output.py`: typed output mapping containing only enabled keys.
- `losses/`: masked classification, segmentation, concept, grounding, hierarchy, and total loss.

## Tasks

1. Add failing tests for configuration gates, shapes, inference, AMP, gradients, and masked losses.
2. Implement architecture modules and configuration parsing.
3. Implement task-aware loss modules and validate partial batches.
4. Run focused tests, Ruff, and review; fix all important findings.
5. Run a reproducible 16-sample CUDA overfit and save real evidence.
6. Run the full suite, source/split preservation checks, then update concise handoff docs.

## Acceptance

- Optional outputs enable/disable exactly; lesion logits match input height/width.
- Binary-only, subtype, mask, and full multi-task losses remain finite and backpropagate.
- CUDA AMP inference/backward succeeds; inference needs no labels.
- Tiny-batch loss decreases sharply with executed evidence.

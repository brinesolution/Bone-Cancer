# Latest phase handoff

PHASE 2 COMPLETE

Created/changed:
- Strict label/schema readers, content-hashed provenance loader, image index/quality audit, required figures, and deterministic duplicate/leakage grouping.
- Active scripts 01–03, ordered PowerShell/VS Code Phase 2 command, openpyxl dependency/freeze, 28 focused Phase 2 tests, and updated data documentation.
- Four generated manifests, three audit figures, execution summaries, final invariant report, and Phase 2 source snapshot.

Validation performed:
- Indexed and decoded all 12,557 Main images; mapped all 12,557 CSV rows with no missing/extra rows and no unreadable files.
- Required 100% SHA-256 content-anchored provenance coverage before recording binary semantics; standalone audit rerun remained idempotent.
- Reported exact, canonical and perceptual duplicate evidence, connected groups, cross-split leakage and label conflicts.
- Visually inspected audit figures and both near-blank flags; recorded orientation, film-capture, burned-in-text and exposure risks.
- Pytest: 59 passed; focused Phase 2: 28 passed; Ruff, compileall and pip check passed; independent review found no remaining findings.
- Full source snapshot matched the Phase 1 baseline: 12,560 files and 1,120,418,007 bytes unchanged.

Results:
- Binary label 0=`normal` (6,827); label 1=`bone_tumour` (5,730); semantics verified for all images through content-anchored evidence.
- Train 8,789; validation 1,884; test 1,884. Quality: 12,555 readable, 2 near-blank, 0 unreadable.
- Partial metadata: risk/boxes/polygons on 1,867 images; unambiguous subtype on 1,866; optional branches remain disabled pending Phase 4.
- Duplicate audit: 2,045 groups, 6,968 involved images, 13,225 edges, 6,112 cross-split edges including 508 exact-byte edges, and 6 label-conflict edges.
- No training, checkpoints, accuracy, AUROC, AUPRC, F1, Dice, calibration or clinical verification results exist.

Outstanding issues:
- Phase 3 must resolve 1,215 cross-split connected groups through a corrected manifest without moving source files; official folders are unsafe for direct training.
- Six duplicate candidate pairs conflict in binary label, including two exact-byte conflicts; retain both source labels and log the Phase 3 resolution rule.
- One BTXRD workbook row has two active subtype columns and remains blank pending Phase 4 review.
- No patient/study identifier is available. Mixed orientation, film photographs and burned-in text/markers require preprocessing/leakage controls.

Do not rerun:
- Environment creation, dependency installation, scaffold generation or Phase 1 source snapshot creation.
- Phase 2 audit unless source/configuration changes; never edit generated manifests manually.
- Do not modify `dataset/Main`, train from official folders, or enable optional heads before their gates.

Next recommended model:
SOL

Why that model:
Phase 3 requires careful duplicate-group split repair, label-conflict rules, deterministic preprocessing, partial-label DataLoader contracts and leakage tests.

Next phase:
Phase 3 — Cleaning, Manifest-Based Splits, Preprocessing and Loaders

Resume file:
E:\Bone Cancer\PROJECT_STATE.md

Manual model switch required: select SOL before continuing; automatic switching is unavailable.

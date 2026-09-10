# Phase 4 Annotation System Implementation Plan

**Goal:** Provide a working local metadata and LabelMe review workflow with revision history, strict schema validation, auditable spatial conversion and progress reporting.

**Spec:** BONE_CANCER_10_PHASE_BUILD_DIRECTIVE.md Sections 9 and Phase 4. Existing Phase 3 manifests and assignments are the inputs; no training occurs.

## IDEA

The existing source contains 1,867 LabelMe JSON files with rectangles and polygons, but neither conversion nor reviewer workflow exists. Their labels are dataset annotations, not new expert verification. Inputs include clean/split manifests, the audited class schema, BTXRD source images/JSON/workbook and disabled optional-task gates. Outputs belong under workspace/annotations, artifacts/figures and logs.

Risks: wrong-image reuse, coordinate/orientation mismatches, source overwrites, stale revisions, unknown becoming negative, diagnostic filename leakage, accidental activation of unreviewed supervision, and invented reviewer identity.

## DESIGN

- Reusable logic lives in src/bone_cancer_ai/annotations; tools/annotation_app is a thin Streamlit interface plus compatibility exports.
- Metadata uses SQLite transactions and optimistic revision numbers. Every saved revision records actual reviewer input, UTC timestamp, project-annotation provenance, status, notes, nullable binary/subtype/risk and tri-state concepts. No source label is silently copied into a review.
- Source image SHA-256 is checked before reuse. Only identical bytes may share spatial annotation; canonical IDs and pHash are review hints only.
- LabelMe edits copies staged under workspace/annotations/labelme, with opaque image names and a binding sidecar. Source JSON imports are kept separate from human revisions.
- Conversion validates shape classes/types, finite coordinates, nonzero geometry, image dimensions and identity. Polygon masks use original coordinates and rectangles remain boxes rather than fabricated segmentation masks. EXIF orientation ambiguity fails closed.
- Reports explicitly distinguish mechanically validated source annotations from project-reviewed labels and clinical verification. Optional heads remain disabled until an explicit validated supervision release; Phase 4 tool acceptance does not require inventing annotations or reviewing all images.
- Streamlit provides image/ID display, unknown-aware review fields, reviewer and notes, previous/next, autosave on edits, CSV download, progress and spatial staging/import actions. No network uploads.

## PLAN / acceptance mapping

### 1. Metadata contracts and storage
- [x] Write failing tests for invalid classes, unknown tri-state labels, reviewer requirement, create/edit/reload/history, stale revision rejection and exact-only reuse.
- [x] Implement annotations/schema.py and storage.py with transactional SQLite storage and CSV/progress exports.
- [x] Run focused tests and review persistence/provenance behavior.

### 2. Spatial conversion and LabelMe integration
- [x] Write failing geometry/identity tests; implement annotations/spatial.py and labelme_bridge.py.
- [x] Validate installed LabelMe CLI, stage safe copies and retain bindings/revisions.
- [x] Convert the existing BTXRD JSON set using byte-exact Main matches, produce masks/boxes/master CSV/issues and representative overlays. Reject invalid items explicitly.

### 3. Metadata UI and orchestration
- [x] Implement annotation_app/app.py, scripts 07/08 and annotation config.
- [x] Exercise create/edit/autosave/reload/navigation/unknown/export through Streamlit AppTest on synthetic fixtures; launch the local real-data UI and inspect its AppTest elements. Browser visual QA unavailable because CUA has no browser connection.
- [x] Add progress reporting and complete workflow commands/documentation.

### 4. REVIEW / FIX / VERIFY / DOCUMENT / HANDOFF
- [x] Independent read-only review; fix material findings with regression tests.
- [x] Run full pytest, Ruff, compileall, pip check, annotation validation CLI, overlay inspection and source snapshot comparison.
- [x] Update README files, PROJECT_GUIDE, PROJECT_STATE, CHANGELOG and PHASE_HANDOFF with only executed evidence.
- [x] Commit verified Phase 4 milestone and recommend SOL for Phase 5.

## Execution notes

The user authorized Phase 4 based on the directive, which supplies the subsystem design and allows LabelMe. Implementation proceeds within that authorization without a second design approval. Existing feature branch and project interpreter are reused. Source data and Phase 3 split assignments remain immutable; annotation releases are separate derived outputs.

2026-09-10 resume: 24 metadata/UI/spatial/native runtime tests passed before source-import regressions. LabelMe 6.3.1 installed; CLI --output requires a directory and explicit keep_prev=false prevents copying prior-image labels. Native Qt smoke uses a synthetic image and isolated settings; LabelMe must initialize ONNX Runtime before Qt on Windows. AppTest real-data load has no errors, 4 selectboxes and 16 concept checkboxes. Browser CUA reported no browser available, so no browser visual check is claimed.

Independent review found exact-image source duplicates with different drawings. Initial full import produced 1,851 rows from 1,841 JSONs, 26 unmatched sources and 13 validation failures; command exited 1. Evidence retained in logs/phase4_initial_import_rejected.json and generation dd8bf4e772474d6b89ee7b2d0e69b541. Corrective work quarantines conflicting drawings and enforces one index row per sample. CLI nonzero failure and overlay/box-label integrity regression tests pass. Final source import/checks/docs/handoff remain pending until this fix is verified.

## Final VERIFY / acceptance evidence

- Create/edit/save/reload: SQLite and Streamlit AppTest passed; native LabelMe loaded, edited, saved and reloaded a synthetic polygon using its installed Qt API and isolated preferences.
- Invalid names rejected: schema and spatial-class tests passed. No filename diagnosis copied: source-reference labels remain separate; test review fields initialize unknown.
- Mask alignment: synthetic geometry/union/EXIF tests passed. Full reconversion of 1,825 source records matched native masks and boxes exactly. Three full overlays and a 12-image grid were visually inspected; no systematic shift/scale/orientation error was observed. This establishes technical conversion consistency, not clinical correctness.
- Unknown supported: binary/subtype/risk may be null; concepts distinguish unassessed from assessed-absent. Empty drawings supply no segmentation target.
- Workflow README covers review, LabelMe staging/edit/import, history, exact duplicate agreement, source conversion, progress, export, validation, libraries, theory, commands and protected files.
- Full suite: 126 passed in 32.60s; Ruff, compileall and pip check passed. Independent re-review closed all material findings, including compact mask signatures to avoid ~5.79 GiB preflight retention.
- Accepted generation c204d1f6873644948e4f355f26b399ae: 1,867 JSONs inspected; 1,825 unique mask/box rows; 16 conflicting JSONs quarantined; 26 unmatched; zero accepted-artifact validation issues. One subtype stays unknown. Human/project clinical verification remains Unavailable, and real human review count is zero.
- Source snapshot equals Phase 3 exactly; DataModule revalidated existing split fingerprints/counts with no source/split mutation. Training and all model metrics are Not run.

## DOCUMENT / HANDOFF

PROJECT_STATE, PHASE_HANDOFF, CHANGELOG, PROJECT_GUIDE and relevant folder READMEs reflect executed evidence. Phase 5 remains pending and requires manual SOL selection. Optional annotations remain separate from training manifests until a deliberate validated supervision release. The first rejected generation is retained only for debugging provenance. The browser visual check could not run because no browser was connected; AppTest, native editor tests and local image inspection provide the recorded verification instead. Git-ignore verification confirmed generated annotation files and SQLite remain excluded; the milestone stages source, tests, configuration and documentation only.

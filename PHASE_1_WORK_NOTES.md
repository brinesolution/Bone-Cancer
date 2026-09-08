# Phase 1 implementation plan and work notes

**Goal:** Implement Phase 1 of the master directive, with no training and no source-data edits.
**Architecture:** The approved Section 5 layout is the design. Reusable environment, configuration, logging, seed, device, and path utilities live under `src/bone_cancer_ai/utils`; scripts orchestrate them. Later-phase modules are explicit inactive scaffolds, and data tables have headers only.
**Tech stack:** Windows, Python 3.11, PyTorch CUDA, PyYAML, NumPy, Python logging, pytest, Ruff.
**Spec:** `BONE_CANCER_10_PHASE_BUILD_DIRECTIVE.md`, read in full (2,625 lines).

## IDEA

Inputs: master directive and original dataset; no existing code, READMEs, PROJECT_STATE, PHASE_HANDOFF, configs, or Git repository. The user has approved the architecture and execution; no redesign is needed. Output: an importable, tested scaffold, working isolated environment, source-integrity evidence, and persistent handoff.

Constraints: work under this project; never write to dataset/Main; do not train, audit label semantics, infer diagnoses, or implement later phases. All model results remain Not run. Optional label heads stay disabled pending verification. Use existing Git identity only.

Risks: wrong Python launcher, incompatible CUDA wheel/driver, package dependency conflicts, accidental source output path, synthetic scaffolds mistaken for audited data, silent CPU fallback, duplicate logging handlers, nondeterministic seeds.

## PLAN / DESIGN

- [x] 1. Snapshot source relative paths, sizes, modification times and SHA-256 into logs; inventory local software. Use installed Python 3.11 directly because `py -3.11` does not discover the uv-managed runtime. Create `.venv` from that runtime.
- [x] 2. Create every Section 5 path, README, config, notebook shell and package marker. Inactive scripts exit nonzero with owning-phase guidance. Empty CSVs contain headers only and are explicitly not audited manifests. Add `.gitignore`, `.gitattributes`, `.env.example`, pyproject and VS Code settings/tasks.
- [x] 3. Verify current official PyTorch wheel selection and maintained Grad-CAM package name; install base/train/dev dependencies in `.venv`; run `pip check` and imports. Freeze only after verification.
- [x] 4. Write utility tests and observe expected failures before implementing. Contracts: `load_config(Path) -> dict` rejects malformed/non-mapping YAML; `ProjectPaths.from_config(dict)` resolves configured roots, `output_path(str) -> Path` rejects escape and source overlap; `configure_logging(path, level)` produces UTC JSON lines without duplicate handlers; `seed_everything(seed, deterministic)` seeds Python/NumPy/torch; `get_device(request)` rejects unavailable CUDA rather than falling back; `environment_report() -> dict` records installed package imports and a synthetic CUDA tensor forward/backward check, without model training.
- [x] 5. Implement those utilities in individual files and `scripts/00_verify_environment.py`. Configuration paths resolve independently of terminal working directory. Device errors are explicit. JSON serialization retains exception details. Reproducibility utilities record versions and source integrity using streaming hashes; never write source files.
- [x] 6. Run pytest, Ruff, compileall, editable-install import, environment verifier, pip check, PowerShell workflow and source snapshot comparison. Tests cover invalid configs, escaping/overlapping output paths, repeatable RNG, no silent CUDA fallback, JSON log reuse, source mutation detection, CLI success/failure and inactive later-phase commands.
- [x] 7. Request independent review using the code-review skill; fix important findings and rerun relevant checks. Final review checks every Phase 1 task and acceptance criterion, plus Section 5 file coverage and folder READMEs.
- [x] 8. Update PROJECT_GUIDE (all 31 sections, future behavior marked Pending), PROJECT_STATE, CHANGELOG and latest-only PHASE_HANDOFF. Create verified Git milestone using existing identity, excluding dataset/generated outputs. Stop for manual SOL selection before Phase 2.

## TEST / REVIEW / FIX / VERIFY

Execution evidence is local under `logs/`.

- IDEA/PLAN/DESIGN: reviewed all 2,625 directive lines; inspected the otherwise empty project and existing dataset/software. Used the approved design without architectural redesign. Source hash baseline captured before scaffolding.
- BUILD: completed all 201 prescribed paths and authored-folder READMEs; created editable package, 40 declared dependency imports and utility/CLI implementation. Later-phase code remains inactive; data tables have headers only.
- TEST: initial run had 25 expected failures from missing utility/CLI implementations and 3 existing scaffold/import passes (`phase1_tests_red.txt`). Strengthened the unsafe-output test to require the actual source-protection error and observed it fail before CLI implementation (`source_guard_red.txt`). Utility implementation then passed 24 tests, followed by 28 full-suite tests.
- REVIEW: independent read-only reviewer Arjuna inspected requirements, code, tests, configs and documentation. No Critical/Important findings. Minor items: misleading import-smoke name and missing environment-negative-path tests. Renamed the test and added three failure-report tests. No source or training work delegated.
- FIX: two Ruff line-length violations were formatting issues; formatted and reran. Package metadata showed both Albumentations and Grad-CAM depend on opencv-python-headless. Removed competing cv2 distributions and reinstalled only headless, as official package guidance requires; updated dependency intent and docs. The Python launcher issue was resolved by selecting the actual installed 3.11 interpreter; no global interpreter changes.
- VERIFY: 31 tests passed in 13.29s; Ruff, compileall, pip check, lock dry-run, CLI and PowerShell wrapper returned 0. All 40 declared imports succeeded. Synthetic tensor forward/backward used CUDA AMP with finite gradients. VS Code JSON and notebook schemas validated. All source paths/SHA-256/sizes/mtimes match across 12,560 files (1,120,418,007 bytes). All 201 required paths and folder READMEs exist. Tests are synthetic; no model training, annotations or clinical metrics were produced.
- DOCUMENT: completed the 31-section guide, folder docs, state, changelog and latest-only handoff. Recorded official package sources, interpreter workaround, headless OpenCV choice, inactive-scaffold semantics and upstream non-fatal import messages. Freeze contains 237 installed distributions and excludes the editable local project.
- HANDOFF: Phase 1 complete. Next: SOL for Phase 2 dataset audit/leakage analysis, after manual model switch. Do not regenerate scaffold/install packages or replace source snapshots during ordinary resume.

Source-safety acceptance is verified by actual byte hashes, not merely by lack of write commands. The full hash pass is an integrity snapshot, not Phase 2 duplicate analysis. Clean-room reinstall was not claimed; setup was created and verified locally, then lock consistency was dry-run checked.

## DOCUMENT / HANDOFF

Phase 1 passed every acceptance criterion. PROJECT_STATE.md and PHASE_HANDOFF.md contain the completed transition. Manual SOL selection is required before Phase 2.

"""Label/provenance validation, manifest enrichment, and audit figures."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from bone_cancer_ai.data.schema import ISSUE_COLUMNS

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _issue(row: pd.Series, issue_type: str, details: str, source: str) -> dict:
    return {
        "record_id": row.get("record_id", ""),
        "image_id": row.get("image_id", ""),
        "official_split": row.get("official_split", ""),
        "issue_type": issue_type,
        "details": details,
        "source": source,
    }


def _unique_nonempty(frame: pd.DataFrame, column: str) -> list[str]:
    if column not in frame:
        return []
    return sorted({str(value) for value in frame[column] if pd.notna(value) and str(value)})


def audit_labels(
    master: pd.DataFrame,
    provenance: pd.DataFrame,
    existing_issues: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Verify binary meanings and copy only unambiguous provenance labels into the manifest."""
    enriched = master.copy()
    for column in ("binary_semantic", "subtype_label", "risk_label"):
        enriched[column] = pd.Series("", index=enriched.index, dtype="string")
    for column in ("has_box", "has_mask"):
        enriched[column] = False
    issues = (
        []
        if existing_issues is None
        else existing_issues.loc[
            existing_issues.get("source", pd.Series(index=existing_issues.index, dtype=str))
            .fillna("")
            .ne("provenance")
        ]
        .reindex(columns=ISSUE_COLUMNS)
        .to_dict("records")
    )
    grouped = {key: value for key, value in provenance.groupby("canonical_source_id", sort=False)}
    main_hashes = {
        key: set(group["sha256"].dropna().astype(str))
        for key, group in enriched.groupby("canonical_source_id", sort=False)
    }
    provenance_hashes = provenance.get(
        "source_sha256", pd.Series(index=provenance.index, dtype="string")
    )
    anchored_provenance_mask = pd.Series(False, index=provenance.index)
    for canonical, evidence in grouped.items():
        hashes = main_hashes.get(canonical, set())
        anchored_provenance_mask.loc[evidence.index] = (
            provenance_hashes.loc[evidence.index].fillna("").astype(str).isin(hashes)
        )
    anchored_provenance = provenance.loc[anchored_provenance_mask]
    matched = 0
    content_anchored = 0
    semantic_matches = 0
    for index, row in enriched.iterrows():
        evidence = grouped.get(row["canonical_source_id"])
        if evidence is None or evidence.empty:
            issues.append(
                _issue(
                    row, "missing_provenance_evidence", "No canonical metadata match", "provenance"
                )
            )
            continue
        matched += 1
        hashes = main_hashes.get(row["canonical_source_id"], set())
        evidence = evidence.loc[
            evidence.get("source_sha256", pd.Series(index=evidence.index, dtype="string"))
            .fillna("")
            .astype(str)
            .isin(hashes)
        ]
        if evidence.empty:
            issues.append(
                _issue(
                    row,
                    "unverified_provenance_content",
                    "Canonical metadata has no SHA-256 match within the Main source group",
                    "provenance",
                )
            )
            continue
        content_anchored += 1
        source_issues = _unique_nonempty(evidence, "source_issue")
        if source_issues:
            issues.append(
                _issue(row, "provenance_source_issue", "; ".join(source_issues), "provenance")
            )
        semantics = _unique_nonempty(evidence, "binary_semantic")
        label = row.get("binary_label")
        expected = "normal" if label == 0 else "bone_tumour" if label == 1 else ""
        if len(semantics) != 1:
            issues.append(_issue(row, "ambiguous_binary_semantics", repr(semantics), "provenance"))
        elif semantics[0] != expected:
            issues.append(
                _issue(
                    row,
                    "binary_semantic_conflict",
                    f"CSV label {label!r} expects {expected!r}; provenance={semantics[0]!r}",
                    "provenance",
                )
            )
        else:
            enriched.at[index, "binary_semantic"] = semantics[0]
            semantic_matches += 1
        for column, issue_type in (("risk_label", "ambiguous_risk_label"),):
            values = _unique_nonempty(evidence, column)
            if len(values) == 1:
                enriched.at[index, column] = values[0]
            elif len(values) > 1:
                issues.append(_issue(row, issue_type, repr(values), "provenance"))
        subtype_values = _unique_nonempty(evidence, "subtype_label")
        subtype_source_invalid = any("subtype_" in issue for issue in source_issues)
        if len(subtype_values) == 1 and not subtype_source_invalid:
            enriched.at[index, "subtype_label"] = subtype_values[0]
        elif len(subtype_values) > 1:
            issues.append(
                _issue(row, "ambiguous_subtype_label", repr(subtype_values), "provenance")
            )
        enriched.at[index, "has_box"] = bool(evidence["has_box"].fillna(False).any())
        enriched.at[index, "has_mask"] = bool(evidence["has_mask"].fillna(False).any())
    issue_frame = pd.DataFrame(issues, columns=ISSUE_COLUMNS)
    labeled = int(enriched["binary_label"].notna().sum())
    binary_issue_types = {
        "missing_label_csv",
        "multiple_label_csv",
        "unreadable_label_csv",
        "unexpected_label_columns",
        "duplicate_label_id",
        "missing_image_id",
        "invalid_binary_label",
        "image_missing_label",
        "label_missing_image",
        "missing_provenance_evidence",
        "unverified_provenance_content",
        "ambiguous_binary_semantics",
        "binary_semantic_conflict",
    }
    has_binary_issue = bool(set(issue_frame.get("issue_type", [])) & binary_issue_types)
    summary = {
        "images": len(enriched),
        "labeled_images": labeled,
        "provenance_matched_images": matched,
        "provenance_coverage": matched / len(enriched) if len(enriched) else 0.0,
        "provenance_content_anchored_images": content_anchored,
        "provenance_content_coverage": content_anchored / len(enriched) if len(enriched) else 0.0,
        "provenance_rows": len(provenance),
        "content_anchored_provenance_rows": len(anchored_provenance),
        "nonexact_provenance_rows": len(provenance) - len(anchored_provenance),
        "binary_semantic_matches": semantic_matches,
        "binary_semantics_verified": bool(
            len(enriched) > 0
            and labeled == len(enriched)
            and semantic_matches == len(enriched)
            and not has_binary_issue
        ),
        "binary_class_names": {"0": "normal", "1": "bone_tumour"},
        "class_counts": {
            str(int(key)): int(value)
            for key, value in enriched["binary_label"].dropna().value_counts().sort_index().items()
        },
        "split_counts": {
            str(key): int(value) for key, value in enriched["official_split"].value_counts().items()
        },
        "subtype_labeled_images": int(enriched["subtype_label"].fillna("").ne("").sum()),
        "risk_labeled_images": int(enriched["risk_label"].fillna("").ne("").sum()),
        "images_with_boxes": int(enriched["has_box"].fillna(False).sum()),
        "images_with_polygon_masks": int(enriched["has_mask"].fillna(False).sum()),
        "provenance_sources": {
            str(key): int(value) for key, value in provenance["source_name"].value_counts().items()
        },
        "content_anchored_provenance_sources": {
            str(key): int(value)
            for key, value in anchored_provenance["source_name"].value_counts().items()
        },
        "patient_identifier_columns": [],
        "patient_grouping_status": "Unavailable: no patient/study identifier column found",
        "issues": len(issue_frame),
        "issue_counts": {
            str(key): int(value) for key, value in issue_frame["issue_type"].value_counts().items()
        },
    }
    return enriched, issue_frame, summary


def _save_figure(figure: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight", metadata={"Software": "Matplotlib"})
    plt.close(figure)
    return path


def _balanced_sample(master: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    readable = master[master["quality_status"].eq("readable")].copy()
    if readable.empty:
        return readable
    readable = readable.sort_values("record_id")
    groups = list(readable.groupby(["official_split", "binary_label"], dropna=False, sort=True))
    per_group = max(1, math.ceil(sample_size / len(groups)))
    sampled = [group.sample(min(len(group), per_group), random_state=seed) for _, group in groups]
    return pd.concat(sampled).sort_values("record_id").head(sample_size)


def save_audit_figures(
    master: pd.DataFrame, output_dir: Path, seed: int = 42, sample_size: int = 16
) -> list[Path]:
    """Render required class, dimension and anonymized sample-grid figures."""
    output_dir = Path(output_dir)
    outputs = []
    counts = master.groupby(["official_split", "binary_label"]).size().unstack(fill_value=0)
    figure, axis = plt.subplots(figsize=(8, 5))
    counts.plot(kind="bar", ax=axis)
    axis.set(
        title="Audited binary-label counts by official split",
        xlabel="Official split",
        ylabel="Images",
    )
    axis.legend(title="Binary label")
    figure.tight_layout()
    outputs.append(_save_figure(figure, output_dir / "dataset_class_distribution.png"))

    readable = master.dropna(subset=["width", "height"])
    figure, axis = plt.subplots(figsize=(8, 5))
    for split, group in readable.groupby("official_split", sort=True):
        axis.scatter(group["width"], group["height"], s=8, alpha=0.35, label=split)
    axis.set(title="Stored image dimensions", xlabel="Width (pixels)", ylabel="Height (pixels)")
    axis.legend(title="Official split")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    outputs.append(_save_figure(figure, output_dir / "dimension_distribution.png"))

    selected = _balanced_sample(master, sample_size, seed)
    columns = 4
    rows = max(1, math.ceil(max(len(selected), 1) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(12, 3 * rows), squeeze=False)
    for axis in axes.ravel():
        axis.axis("off")
    for case_number, ((_, row), axis) in enumerate(
        zip(selected.iterrows(), axes.ravel(), strict=False), 1
    ):
        with Image.open(row["absolute_path"]) as opened:
            image = ImageOps.exif_transpose(opened).convert("L")
            axis.imshow(np.asarray(image), cmap="gray", vmin=0, vmax=255)
        label = row.get("binary_semantic") or f"class_{int(row['binary_label'])}"
        axis.set_title(f"case-{case_number:04d} | {row['official_split']} | {label}", fontsize=8)
        axis.axis("off")
    figure.suptitle("Deterministic audit sample (manifest IDs withheld; source pixels unchanged)")
    figure.tight_layout()
    outputs.append(_save_figure(figure, output_dir / "sample_grid.png"))
    return outputs

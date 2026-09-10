"""Resolve the local annotation workspace and verify displayed image identities."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from bone_cancer_ai.annotations.storage import AnnotationStore
from bone_cancer_ai.utils.config import load_config
from bone_cancer_ai.utils.paths import ProjectPaths


@dataclass
class AnnotationContext:
    config: dict
    paths: ProjectPaths
    root: Path
    schema: dict
    manifest: pd.DataFrame
    store: AnnotationStore

    def image_path(self, row: dict) -> Path:
        path = Path(row["absolute_path"]).resolve()
        if not path.is_relative_to(self.paths.dataset_root):
            raise ValueError("Image escapes the read-only source root")
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Source image hash changed; re-audit before annotating")
        return path


def load_annotation_context(config_path: Path) -> AnnotationContext:
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    data_path = Path(config["data_config"])
    data = load_config(data_path if data_path.is_absolute() else config_path.parent / data_path)
    paths = ProjectPaths.from_config(data)
    root = paths.output_path(config["annotation_root"])
    if "dataset" in [part.lower() for part in root.parts]:
        raise ValueError("Annotation root cannot be a source dataset directory")
    database = paths.output_path(config["database"])
    if not database.is_relative_to(root):
        raise ValueError("Database must be inside the annotation root")
    schema_path = Path(data["schema_path"])
    schema = load_config(
        schema_path if schema_path.is_absolute() else paths.project_root / schema_path
    )
    manifest = pd.read_csv(paths.output_path(config["manifest"]), keep_default_na=False)
    required = {"sample_id", "image_id", "absolute_path", "sha256", "is_eligible"}
    if not required <= set(manifest):
        raise ValueError(f"Annotation manifest missing columns: {sorted(required - set(manifest))}")
    if manifest.sample_id.duplicated().any():
        raise ValueError("Duplicate sample IDs in annotation manifest")
    eligible = manifest.is_eligible.astype(str).str.lower()
    if not eligible.isin(["true", "false"]).all():
        raise ValueError("is_eligible must be a boolean")
    manifest = manifest.loc[eligible.eq("true")].reset_index(drop=True)
    if manifest.empty:
        raise ValueError("No eligible images to annotate")
    concepts = config["concept_names"]
    if not isinstance(concepts, list) or len(concepts) != len(set(concepts)):
        raise ValueError("concept_names must be a unique list")
    return AnnotationContext(
        config, paths, root, schema, manifest, AnnotationStore(database, schema, concepts)
    )

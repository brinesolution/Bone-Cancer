"""Resolve configured paths and reject outputs that could affect immutable source data."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProjectPaths:
    """Project and source roots, resolved through existing symlinks/junctions."""

    project_root: Path
    dataset_root: Path

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ProjectPaths":
        """Require an absolute project root; resolve relative source paths against it."""
        root = Path(config["project_root"]).expanduser()
        if not root.is_absolute():
            raise ValueError("project_root must be absolute")
        root = root.resolve()
        source = Path(config["dataset_root"]).expanduser()
        source = (source if source.is_absolute() else root / source).resolve()
        if not source.is_relative_to(root) or source == root:
            raise ValueError("dataset_root must be strictly inside project_root")
        if config.get("source_read_only", True) is not True:
            raise ValueError("Source dataset must remain read-only")
        return cls(root, source)

    def output_path(self, path: str | Path) -> Path:
        """Validate a destination without creating it; reject source ancestors too."""
        candidate = Path(path).expanduser()
        candidate = (
            candidate if candidate.is_absolute() else self.project_root / candidate
        ).resolve()
        if not candidate.is_relative_to(self.project_root):
            raise ValueError(f"Output escapes project root: {candidate}")
        if candidate.is_relative_to(self.dataset_root) or self.dataset_root.is_relative_to(
            candidate
        ):
            raise ValueError(f"Output overlaps read-only source dataset: {candidate}")
        return candidate

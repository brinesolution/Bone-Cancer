"""Transactional, append-only project annotation revisions with stale-write protection."""

import json
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from bone_cancer_ai.annotations.schema import validate_metadata


class AnnotationStore:
    """Persist reviewer-entered metadata without mutating source labels or old revisions."""

    def __init__(self, path: Path, schema: dict, concept_names: list[str]):
        self.path = Path(path).resolve()
        if "dataset" in [part.lower() for part in self.path.parts]:
            raise ValueError("Annotation output cannot overlap source dataset directories")
        self.schema = schema
        self.concept_names = concept_names
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS revisions (sample_id TEXT NOT NULL, "
                "revision INTEGER NOT NULL, payload TEXT NOT NULL, "
                "PRIMARY KEY(sample_id, revision))"
            )

    def _connect(self):
        return sqlite3.connect(self.path, timeout=30)

    def latest(self, sample_id: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM revisions WHERE sample_id=? ORDER BY revision DESC LIMIT 1",
                (sample_id,),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def save(
        self,
        sample_id: str,
        image_id: str,
        source_sha256: str,
        payload: dict,
        *,
        expected_revision: int,
    ) -> dict:
        data = validate_metadata(payload, self.schema, self.concept_names)
        if not re.fullmatch(r"[A-Za-z0-9_-]+", sample_id):
            raise ValueError("Unsafe sample identity")
        if not re.fullmatch(r"[0-9a-f]{64}", source_sha256):
            raise ValueError("Invalid source identity hash")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT revision, payload FROM revisions WHERE sample_id=? "
                "ORDER BY revision DESC LIMIT 1",
                (sample_id,),
            ).fetchone()
            revision = row[0] if row else 0
            if revision != expected_revision:
                raise ValueError("Stale revision; reload before saving")
            if row:
                old = json.loads(row[1])
                if old["source_sha256"] != source_sha256 or old["image_id"] != image_id:
                    raise ValueError("Image identity changed; re-audit before reviewing")
            data.update(
                sample_id=sample_id,
                image_id=image_id,
                source_sha256=source_sha256,
                revision=revision + 1,
                timestamp=datetime.now(UTC).isoformat(),
                provenance="project_annotation",
                clinical_verification="Unavailable",
            )
            connection.execute(
                "INSERT INTO revisions VALUES (?, ?, ?)",
                (sample_id, revision + 1, json.dumps(data, sort_keys=True)),
            )
        return data

    def history(self, sample_id: str) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload FROM revisions WHERE sample_id=? ORDER BY revision", (sample_id,)
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def export_frame(self) -> pd.DataFrame:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT r.payload FROM revisions r JOIN "
                "(SELECT sample_id, MAX(revision) revision FROM revisions "
                "GROUP BY sample_id) latest ON r.sample_id=latest.sample_id "
                "AND r.revision=latest.revision ORDER BY r.sample_id"
            ).fetchall()
        data = []
        for row in rows:
            value = json.loads(row[0])
            value["concept_labels"] = json.dumps(value.pop("concepts"), sort_keys=True)
            data.append(value)
        columns = [
            "sample_id",
            "image_id",
            "source_sha256",
            "binary_review",
            "subtype_label",
            "risk_label",
            "concept_labels",
            "reviewer",
            "review_status",
            "notes",
            "revision",
            "timestamp",
            "provenance",
            "clinical_verification",
        ]
        return pd.DataFrame(data, columns=columns)

    def progress(self, total: int) -> dict:
        frame = self.export_frame()
        return {
            "total": total,
            "saved": len(frame),
            "unsaved": total - len(frame),
            **{
                status: int(frame.review_status.eq(status).sum())
                for status in ("draft", "reviewed", "needs_review")
            },
        }

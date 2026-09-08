"""Read-only source snapshots for checking file-content and metadata preservation."""

import hashlib
from pathlib import Path


def snapshot_source(source: Path) -> dict:
    """Stream SHA-256 hashes; record paths, sizes and mtimes without writing source files."""
    source = Path(source).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Missing source directory: {source}")
    rows = []
    for path in sorted(source.rglob("*")):
        if path.is_file():
            stat = path.stat()
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            after = path.stat()
            if (stat.st_size, stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError(f"Source changed while hashing: {path}")
            rows.append(
                {
                    "path": path.relative_to(source).as_posix(),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "sha256": digest,
                }
            )
    return {"files": rows, "file_count": len(rows), "total_bytes": sum(r["size"] for r in rows)}

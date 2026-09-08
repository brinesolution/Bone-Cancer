"""UTC JSON-line logging with explicit application-owned handlers."""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path


class JsonFormatter(logging.Formatter):
    """Serialize event metadata and tracebacks as one JSON object per log record."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("event", "run_id", "details"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(log_path: Path | None = None, level: str = "INFO") -> logging.Logger:
    """Configure the package logger; callers must validate any file destination first."""
    logger = logging.getLogger("bone_cancer_ai")
    logger.setLevel(level)
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    console = logging.StreamHandler()
    console.setFormatter(JsonFormatter())
    logger.addHandler(console)
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    return logger

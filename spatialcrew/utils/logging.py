"""Logging utilities for SpatialCrew."""

from __future__ import annotations
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def setup_logging(
    log_dir: str = "./logs",
    level: int = logging.INFO,
    session_id: Optional[str] = None,
) -> logging.Logger:
    """Configure logging for the framework."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger("spatialcrew")
    root_logger.setLevel(level)

    if not root_logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        console_handler.setFormatter(fmt)
        root_logger.addHandler(console_handler)

        if session_id:
            file_handler = logging.FileHandler(
                Path(log_dir) / f"{session_id}.log", encoding="utf-8"
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(fmt)
            root_logger.addHandler(file_handler)

    return root_logger


class SessionLogger:
    """Structured logger for recording agent traces."""

    def __init__(self, output_path: Path, session_id: str):
        self._path = output_path
        self._session_id = session_id
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": self._session_id,
            "event": event_type,
            **data,
        }
        with open(self._path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

"""JSON file logs + readable console output."""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from uat.config import project_root

_CONFIGURED = False


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        for k, v in getattr(record, "extra_fields", {}).items():
            payload[k] = v
        return json.dumps(payload, default=str, ensure_ascii=False)


def setup(level: int = logging.INFO, logfile: str | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    log_dir = project_root() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    name = logfile or f"uat-{datetime.now():%Y%m%d}.jsonl"

    root = logging.getLogger("uat")
    root.setLevel(level)
    root.handlers.clear()

    fh = logging.FileHandler(log_dir / name, encoding="utf-8")
    fh.setFormatter(JsonFormatter())
    root.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s",
                                      datefmt="%H:%M:%S"))
    root.addHandler(ch)
    _CONFIGURED = True


def get(name: str) -> logging.Logger:
    setup()
    return logging.getLogger(f"uat.{name}")

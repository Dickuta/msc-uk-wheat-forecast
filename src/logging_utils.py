"""
Centralised structured logging for pipeline stages.

Emits each log record as a single JSON line on stdout, which is well suited
to CI and log aggregation.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


class JsonFormatter(logging.Formatter):
    """Format log records as JSON lines."""

    def format(self, record: logging.LogRecord) -> str:
        base = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "stage": getattr(record, "stage", None),
        }
        # Add any extra fields
        for key, value in record.__dict__.items():
            if key not in {
                "name",
                "msg",
                "args",
                "created",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "message",
                "name",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "thread",
                "threadName",
                "exc_info",
                "exc_text",
                "stack_info",
                "stage",
            }:
                base[key] = value
        return json.dumps(base, default=str)


def get_logger(name: str) -> logging.Logger:
    """Get a logger configured for the pipeline.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger that outputs JSON lines to stdout
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


class StageLogger:
    """Wrapper that injects stage into every log call."""

    def __init__(self, logger: logging.Logger, stage: str):
        self._logger = logger
        self._stage = stage

    def _log(self, level: int, msg: str, *args, **extra: Any):
        extra["stage"] = self._stage
        self._logger.log(level, msg, *args, extra=extra)

    def info(self, msg: str, *args, **extra: Any):
        self._log(logging.INFO, msg, *args, **extra)

    def warning(self, msg: str, *args, **extra: Any):
        self._log(logging.WARNING, msg, *args, **extra)

    def error(self, msg: str, *args, **extra: Any):
        self._log(logging.ERROR, msg, *args, **extra)

    def debug(self, msg: str, *args, **extra: Any):
        self._log(logging.DEBUG, msg, *args, **extra)


def get_stage_logger(name: str, stage: str) -> StageLogger:
    """Get a StageLogger that injects stage into every log call."""
    return StageLogger(get_logger(name), stage)


@contextmanager
def timed_block(logger: StageLogger, operation: str, **extra: Any):
    """Context manager that logs start/end with duration."""
    start = time.perf_counter()
    logger.info(f"START {operation}", **extra)
    try:
        yield
    except Exception as e:
        duration = time.perf_counter() - start
        logger.error(
            f"FAIL {operation}",
            **{**extra, "duration_sec": round(duration, 3), "error": str(e)},
        )
        raise
    else:
        duration = time.perf_counter() - start
        logger.info(f"END {operation}", **{**extra, "duration_sec": round(duration, 3)})


def log_stage_start(logger: StageLogger, stage: str, description: str):
    """Log stage start."""
    logger.info(f"STAGE {stage} START", description=description)


def log_stage_end(logger: StageLogger, stage: str, success: bool = True, **extra: Any):
    """Log stage end."""
    logger.info(
        f"STAGE {stage} {'OK' if success else 'FAIL'}", success=success, **extra
    )


def log_artifact(logger: StageLogger, path: Path, description: str, **extra: Any):
    """Log an artifact write with size."""
    size = path.stat().st_size if path.exists() else 0
    logger.info(
        f"ARTIFACT {path.name}",
        artifact=str(path),
        description=description,
        size_bytes=size,
        **extra,
    )

"""Project-wide logging setup: one format, written to the console and a file."""

import logging
from pathlib import Path

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def setup_logging(logs_dir: Path | None = None, level: int = logging.INFO) -> None:
    """Configure the root logger. Call this once at the start of a script."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if logs_dir is not None:
        logs_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(logs_dir / "app.log", encoding="utf-8"))

    logging.basicConfig(level=level, format=LOG_FORMAT, handlers=handlers, force=True)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger. Use get_logger(__name__) in every module."""
    return logging.getLogger(name)

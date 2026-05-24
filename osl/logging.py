"""Structured logging configuration via structlog.

Call :func:`configure_logging` once at process start (the Streamlit entrypoint
and any worker do this). Modules obtain loggers with :func:`get_logger`.
"""

from __future__ import annotations

import logging
from typing import Any

import structlog

_CONFIGURED = False


def configure_logging(level: str = "INFO", *, json: bool = False) -> None:
    """Configure structlog + the stdlib root logger.

    Parameters
    ----------
    level:
        Minimum log level name (e.g. ``"INFO"``, ``"DEBUG"``).
    json:
        Emit JSON lines when ``True`` (machine-readable), otherwise a
        human-friendly console renderer.
    """
    global _CONFIGURED

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(format="%(message)s", level=numeric_level)

    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer() if json else structlog.dev.ConsoleRenderer()
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _CONFIGURED = True


def get_logger(name: str | None = None, **initial_values: Any) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger, configuring logging on first use."""
    if not _CONFIGURED:
        configure_logging()
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name, **initial_values)
    return logger

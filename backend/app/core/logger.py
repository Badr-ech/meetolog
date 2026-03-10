"""Structured logging configuration using structlog.

Outputs machine-readable JSON logs with UTC timestamps, log levels,
and exception details.  All log entries automatically include any
context bound via ``structlog.contextvars`` (e.g. ``job_id``,
``worker_id``).

Usage::

    from app.core.logger import get_logger

    logger = get_logger()
    logger.info("event_name", key="value")
"""

from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(*, json_output: bool = True, log_level: str = "INFO") -> None:
    """Initialise structlog and stdlib logging.

    Call once at process startup (API entrypoint or worker ``__main__``).

    Args:
        json_output: Emit JSON lines when ``True`` (production default).
                     Uses coloured console output when ``False`` (local dev).
        log_level:   Root log level name (``DEBUG``, ``INFO``, ``WARNING``, …).
    """
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if json_output:
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Quiet down noisy third-party loggers
    for noisy in ("aioboto3", "aiobotocore", "botocore", "urllib3", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(**initial_context: object) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger.

    Any keyword arguments are permanently bound as context fields::

        logger = get_logger(component="worker")
        logger.info("started")  # {"component": "worker", "event": "started", …}
    """
    return structlog.get_logger(**initial_context)

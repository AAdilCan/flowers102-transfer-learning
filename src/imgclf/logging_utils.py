"""Small logging helper shared across the package.

Every module fetches its logger through :func:`get_logger` so that a single
call to :func:`configure_logging` (usually from the CLI) controls formatting
and verbosity for the whole run.
"""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%H:%M:%S"


def configure_logging(level: int = logging.INFO) -> None:
    """Install a single stream handler on the package root logger.

    Idempotent: calling it more than once only adjusts the level so that
    importing modules never accumulate duplicate handlers.
    """
    global _CONFIGURED
    root = logging.getLogger("imgclf")
    root.setLevel(level)
    if not _CONFIGURED:
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
        root.addHandler(handler)
        root.propagate = False
        _CONFIGURED = True
    else:
        for handler in root.handlers:
            handler.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``imgclf`` namespace."""
    return logging.getLogger(f"imgclf.{name}")

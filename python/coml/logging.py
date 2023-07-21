"""
Logging utilities.

This comes from my other projects.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict

_file_handlers: Dict[Path, logging.FileHandler] = {}
_logger_init: bool = False


def init_logger() -> None:
    """
    Initialize the logger. Log to stdout by default.
    """
    global _logger_init
    if _logger_init:
        return

    logger = logging.getLogger('coml')
    logger.setLevel(level=logging.INFO)
    add_handler(logger)

    _logger_init = True


def add_handler(logger: logging.Logger, file: Optional[Path] = None) -> logging.Handler:
    """
    Add a logging handler.
    If ``file`` is specified, log to file.
    Otherwise, add a handler to stdout.
    """
    fmt = '[%(asctime)s] %(levelname)s (%(threadName)s:%(name)s) %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'

    formatter = logging.Formatter(fmt, datefmt)
    if file is None:
        # Log to stdout.
        handler = logging.StreamHandler(sys.stdout)
    elif file in _file_handlers:
        # Log to file.
        # Reuse the existing handler.
        handler = _file_handlers[file]
    else:
        handler = logging.FileHandler(file)
        _file_handlers[file] = handler
    handler.setLevel(level=logging.DEBUG)  # Print all the logs.
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return handler

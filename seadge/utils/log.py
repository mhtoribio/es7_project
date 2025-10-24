from __future__ import annotations
import logging
from typing import Mapping
from tqdm.contrib.logging import _TqdmLoggingHandler as TqdmLoggingHandler

from seadge import config

# Public logger to import everywhere
log: logging.Logger = logging.getLogger("seadge")

_LEVELS: Mapping[str, int] = {
    "CRITICAL": logging.CRITICAL,
    "ERROR":    logging.ERROR,
    "WARNING":  logging.WARNING,
    "INFO":     logging.INFO,
    "DEBUG":    logging.DEBUG,
    "NOTSET":   logging.NOTSET,
}

def _level_from_str(s: str) -> int:
    return _LEVELS.get(s.upper(), logging.INFO)

def setup_logger(cfg: config.LoggingCfg) -> logging.Logger:
    level = _LEVELS.get(cfg.level.upper(), logging.INFO)
    fmt = cfg.format
    datefmt = "%Y-%m-%dT%H:%M:%S"

    logger = log
    logger.setLevel(level)

    if not any(isinstance(h, TqdmLoggingHandler) for h in logger.handlers):
        h = TqdmLoggingHandler()
        h.setLevel(level)
        h.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        logger.addHandler(h)
        logger.propagate = False
    else:
        for h in logger.handlers:
            h.setLevel(level)
            h.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    return logger

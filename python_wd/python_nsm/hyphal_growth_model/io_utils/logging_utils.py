# io_utils/logging_utils.py
import logging
import os

def setup_logging(name: str = "pycelium", default_level: str | None = None) -> logging.Logger:
    """
    Create/configure a process-wide logger.
    Priority for level (highest first):
      1) explicit default_level arg (e.g., "INFO", "DEBUG")
      2) env PYCELIUM_LOG_LEVEL
      3) WARNING
    """
    level_str = (default_level or os.getenv("PYCELIUM_LOG_LEVEL") or "WARNING").upper()
    level = getattr(logging, level_str, logging.WARNING)

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger

def parse_int_env(name: str, default: int) -> int:
    """Safely parse an int env var; return default on any failure/absence."""
    try:
        val = os.getenv(name)
        return int(val) if (val is not None and val.strip() != "") else default
    except Exception:
        return default

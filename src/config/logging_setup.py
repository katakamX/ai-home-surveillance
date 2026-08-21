"""One small helper to configure standard-library logging for the whole app."""

import logging

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
DATE_FORMAT = "%H:%M:%S"


def setup_logging(level: str = "INFO") -> None:
    """Send log records to the console at the given level.

    Call this once, at application (or script) startup. An unknown level name
    falls back to INFO rather than crashing.
    """
    logging.basicConfig(level=resolve_level(level), format=LOG_FORMAT, datefmt=DATE_FORMAT)


def resolve_level(level: str) -> int:
    """Turn a level name such as "DEBUG" into the numeric level logging expects."""
    value = getattr(logging, str(level).strip().upper(), None)
    return value if isinstance(value, int) else logging.INFO

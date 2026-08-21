"""Application settings, read from environment variables.

Values come from the real environment, or from a .env file if python-dotenv is
installed. Every setting has a sensible default, so the project runs with no
configuration at all.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Repository root, resolved from this file so it works from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL_FILE = "yolov8n.pt"


@dataclass
class Settings:
    """Everything the application can be configured with."""

    camera_source: int | str
    model_path: Path
    confidence_threshold: float
    person_only: bool
    log_level: str
    data_dir: Path
    model_dir: Path


def load_settings(env: Optional[Mapping[str, str]] = None) -> Settings:
    """Build Settings from environment variables.

    Pass an explicit mapping in tests to keep them independent of the real
    environment.
    """
    if env is None:
        _load_env_file()
        env = os.environ

    model_dir = _as_path(env.get("MODEL_DIR"), PROJECT_ROOT / "models")
    model_path = _as_path(env.get("MODEL_PATH"), model_dir / DEFAULT_MODEL_FILE)

    return Settings(
        camera_source=_as_source(env.get("CAMERA_SOURCE"), 0),
        model_path=model_path,
        confidence_threshold=_as_float(env.get("CONFIDENCE_THRESHOLD"), 0.5),
        person_only=_as_bool(env.get("PERSON_ONLY"), True),
        log_level=env.get("LOG_LEVEL", "INFO").strip().upper() or "INFO",
        data_dir=_as_path(env.get("DATA_DIR"), PROJECT_ROOT / "data"),
        model_dir=model_dir,
    )


def _load_env_file() -> None:
    """Load .env if python-dotenv is available. Plain environment variables work either way."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    load_dotenv(PROJECT_ROOT / ".env")


def _as_source(value: Optional[str], default: int | str) -> int | str:
    """A digits-only value is a webcam index; anything else is an RTSP URL or file path."""
    if value is None or not value.strip():
        return default

    value = value.strip()
    return int(value) if value.isdigit() else value


def _as_path(value: Optional[str], default: Path) -> Path:
    """Relative paths are resolved against the project root, not the working directory."""
    if value is None or not value.strip():
        return default

    path = Path(value.strip())
    return path if path.is_absolute() else PROJECT_ROOT / path


def _as_float(value: Optional[str], default: float) -> float:
    if value is None or not value.strip():
        return default

    try:
        return float(value)
    except ValueError:
        return default


def _as_bool(value: Optional[str], default: bool) -> bool:
    if value is None or not value.strip():
        return default

    return value.strip().lower() in {"1", "true", "yes", "on"}

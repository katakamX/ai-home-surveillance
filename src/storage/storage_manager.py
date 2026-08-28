"""Storage layout and retention for recorded events.

A StorageManager owns a root directory and organizes event files under
date-based subdirectories (YYYY/MM/DD), so files never pile up in one flat
folder::

    manager = StorageManager("data/events")
    event_dir = manager.event_dir_for(datetime.now())
    video_path = manager.video_path("evt123", datetime.now())

Nothing here talks to cv2 or knows about cameras, detection, or events: it
only computes paths, creates directories, reports usage, and deletes old
date directories.
"""

import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

VIDEO_SUFFIX = ".mp4"
SNAPSHOT_SUFFIX = ".jpg"
METADATA_SUFFIX = ".json"


class StorageManagerError(Exception):
    """Raised when the storage root or an event directory cannot be created or removed."""


class StorageManager:
    """Organizes event files under a root directory, by date."""

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)

    def event_dir_for(self, when: datetime) -> Path:
        """Return the YYYY/MM/DD directory for a given moment, creating it if needed."""
        event_dir = self.root_dir / when.strftime("%Y") / when.strftime("%m") / when.strftime("%d")
        self._ensure_dir(event_dir)
        return event_dir

    def video_path(self, event_id: str, when: datetime) -> Path:
        """Path for an event's video clip, under that day's directory."""
        return self.event_dir_for(when) / f"{event_id}{VIDEO_SUFFIX}"

    def snapshot_path(self, event_id: str, when: datetime) -> Path:
        """Path for an event's snapshot image, under that day's directory."""
        return self.event_dir_for(when) / f"{event_id}{SNAPSHOT_SUFFIX}"

    def metadata_path(self, event_id: str, when: datetime) -> Path:
        """Path for an event's metadata file, under that day's directory."""
        return self.event_dir_for(when) / f"{event_id}{METADATA_SUFFIX}"

    def get_usage_bytes(self) -> int:
        """Total size, in bytes, of every file under the root directory."""
        if not self.root_dir.exists():
            return 0

        return sum(path.stat().st_size for path in self.root_dir.rglob("*") if path.is_file())

    def delete_old_events(self, retention_days: int, now: Optional[datetime] = None) -> list[Path]:
        """Delete date directories older than retention_days. Returns the deleted paths.

        A directory is "old" if its YYYY/MM/DD path is entirely before the
        cutoff date. Directories that do not parse as a date (unexpected
        files under root_dir) are left alone.
        """
        if retention_days < 0:
            raise StorageManagerError(f"retention_days must be >= 0, got {retention_days!r}")

        if not self.root_dir.exists():
            return []

        cutoff = (now or datetime.now()).date() - timedelta(days=retention_days)
        deleted: list[Path] = []

        for day_dir in self._day_dirs():
            day_date = self._parse_day_dir(day_dir)
            if day_date is not None and day_date < cutoff:
                try:
                    shutil.rmtree(day_dir)
                except OSError as error:
                    raise StorageManagerError(f"Could not delete {day_dir}") from error
                logger.info("Deleted old event directory: %s", day_dir)
                deleted.append(day_dir)

        return deleted

    def _day_dirs(self) -> list[Path]:
        """Every YYYY/MM/DD directory that exists under root_dir."""
        return [
            day_dir
            for year_dir in self.root_dir.glob("*")
            if year_dir.is_dir()
            for month_dir in year_dir.glob("*")
            if month_dir.is_dir()
            for day_dir in month_dir.glob("*")
            if day_dir.is_dir()
        ]

    @staticmethod
    def _parse_day_dir(day_dir: Path) -> Optional[datetime.date]:
        """Parse a YYYY/MM/DD path back into a date, or None if it doesn't match."""
        try:
            return datetime.strptime(
                f"{day_dir.parent.parent.name}-{day_dir.parent.name}-{day_dir.name}", "%Y-%m-%d"
            ).date()
        except ValueError:
            return None

    def _ensure_dir(self, path: Path) -> None:
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise StorageManagerError(f"Could not create directory {path}") from error

"""Event metadata, stored as one JSON file per event.

A MetadataStore writes and reads EventMetadata records using the paths a
StorageManager computes, so metadata files land in the same YYYY/MM/DD
directory as an event's video and snapshot::

    store = MetadataStore(StorageManager("data/events"))
    store.save(EventMetadata(
        event_id="evt123",
        event_type="person_entered_zone",
        timestamp=time.time(),
        track_id=1,
        label="person",
        zone="front_door",
        confidence=0.92,
    ))
    record = store.load("evt123", when)

Independent of OpenCV, YOLO, Event, and the pipeline: it only knows about
EventMetadata and StorageManager, so it can be unit tested with plain data
and wired into the pipeline later.
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.storage.storage_manager import StorageManager

logger = logging.getLogger(__name__)


class MetadataError(Exception):
    """Raised when event metadata cannot be saved, or is missing or malformed on read."""


@dataclass
class EventMetadata:
    """Everything worth remembering about one completed event."""

    event_id: str
    event_type: str
    timestamp: float
    track_id: int
    label: str
    zone: str
    confidence: float
    duration: Optional[float] = None
    video_path: Optional[str] = None
    snapshot_path: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "EventMetadata":
        try:
            return cls(
                event_id=data["event_id"],
                event_type=data["event_type"],
                timestamp=data["timestamp"],
                track_id=data["track_id"],
                label=data["label"],
                zone=data["zone"],
                confidence=data["confidence"],
                duration=data.get("duration"),
                video_path=data.get("video_path"),
                snapshot_path=data.get("snapshot_path"),
            )
        except KeyError as error:
            raise MetadataError(f"Metadata is missing required field: {error}") from error


class MetadataStore:
    """Saves and loads EventMetadata as JSON files, using StorageManager paths."""

    def __init__(self, storage_manager: StorageManager) -> None:
        self.storage_manager = storage_manager

    def save(self, metadata: EventMetadata, when: Optional[datetime] = None) -> Path:
        """Write metadata as JSON and return the file path.

        when defaults to the event's own timestamp, so callers normally don't
        need to pass it.
        """
        path = self.storage_manager.metadata_path(
            metadata.event_id, when or datetime.fromtimestamp(metadata.timestamp)
        )

        try:
            path.write_text(json.dumps(metadata.to_dict(), indent=2), encoding="utf-8")
        except OSError as error:
            raise MetadataError(f"Could not write metadata to {path}") from error

        logger.info(
            "Metadata saved: event_id=%s track=%s zone=%s path=%s",
            metadata.event_id,
            metadata.track_id,
            metadata.zone,
            path,
        )
        return path

    def load(self, event_id: str, when: datetime) -> EventMetadata:
        """Read one event's metadata. Raises MetadataError if missing or malformed."""
        path = self.storage_manager.metadata_path(event_id, when)
        return self._load_path(path)

    def list_for_day(self, when: datetime) -> list[EventMetadata]:
        """All event metadata recorded on the given day, in filename order.

        Malformed files under that day are skipped with a logged warning
        rather than failing the whole listing: one bad file shouldn't hide
        every other event that day.
        """
        day_dir = self.storage_manager.event_dir_for(when)
        return self._load_dir(day_dir)

    def list_all(self) -> list[EventMetadata]:
        """All event metadata under the storage root, in directory-walk order."""
        root_dir = self.storage_manager.root_dir
        if not root_dir.exists():
            return []

        records: list[EventMetadata] = []
        for path in sorted(root_dir.rglob("*.json")):
            records.extend(self._try_load_path(path))
        return records

    def _load_dir(self, day_dir: Path) -> list[EventMetadata]:
        if not day_dir.exists():
            return []

        records: list[EventMetadata] = []
        for path in sorted(day_dir.glob("*.json")):
            records.extend(self._try_load_path(path))
        return records

    def _try_load_path(self, path: Path) -> list[EventMetadata]:
        try:
            return [self._load_path(path)]
        except MetadataError:
            logger.warning("Skipping unreadable metadata file: %s", path)
            return []

    def _load_path(self, path: Path) -> EventMetadata:
        if not path.exists():
            raise MetadataError(f"Metadata file not found: {path}")

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise MetadataError(f"Could not read metadata from {path}") from error

        if not isinstance(data, dict):
            raise MetadataError(f"Metadata in {path} is not a JSON object")

        return EventMetadata.from_dict(data)

"""Read-only HTTP API over the stored surveillance events.

Nothing here records, detects or deletes: it only reads what the pipeline
already wrote, through the existing MetadataStore and StorageManager. That
keeps the API independent of OpenCV, YOLO and the camera, so it can be served
from a machine that has no webcam at all::

    uvicorn src.api.app:app

Or, in tests, with your own storage::

    app = create_app(MetadataStore(StorageManager(tmp_dir)))

There is no database: looking an event up by id means scanning the metadata
files under the storage root. That is fine for a home camera's worth of
events, and can be replaced later without changing these endpoints.
"""

import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from src.config.settings import load_settings
from src.storage.metadata import EventMetadata, MetadataStore
from src.storage.storage_manager import StorageManager

logger = logging.getLogger(__name__)

SNAPSHOT_MEDIA_TYPE = "image/jpeg"
VIDEO_MEDIA_TYPE = "video/mp4"


def default_metadata_store() -> MetadataStore:
    """The store the application uses when none is supplied: data_dir/events."""
    settings = load_settings()
    return MetadataStore(StorageManager(settings.data_dir / "events"))


def create_app(metadata_store: Optional[MetadataStore] = None) -> FastAPI:
    """Build the API. Pass a metadata_store to point it at your own storage."""
    store = metadata_store if metadata_store is not None else default_metadata_store()
    api = FastAPI(title="AI Home Surveillance", version="0.1.0")

    def find_event(event_id: str) -> EventMetadata:
        """Look one event up by id, or fail with 404."""
        for record in store.list_all():
            if record.event_id == event_id:
                return record

        raise HTTPException(status_code=404, detail=f"Event {event_id} not found")

    def serve_file(raw_path: Optional[str], media_type: str, description: str) -> FileResponse:
        """Serve a file an event refers to, refusing anything outside storage."""
        if not raw_path:
            raise HTTPException(status_code=404, detail=f"No {description} for this event")

        path = Path(raw_path)
        if not _is_inside(path, store.storage_manager.root_dir):
            # Metadata is written by this project, so this should never happen;
            # refuse rather than serve an arbitrary file off the disk.
            logger.warning("Refusing to serve %s outside the storage root: %s", description, path)
            raise HTTPException(status_code=404, detail=f"No {description} for this event")

        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"The {description} file is missing")

        return FileResponse(path, media_type=media_type, filename=path.name)

    @api.get("/health")
    def health() -> dict:
        """Liveness check, plus a quick look at what is stored."""
        return {
            "status": "ok",
            "event_count": len(store.list_all()),
            "storage_bytes": store.storage_manager.get_usage_bytes(),
        }

    @api.get("/events", response_model=list[EventMetadata])
    def list_events(zone: Optional[str] = None, limit: Optional[int] = None):
        """Every stored event, newest first. Optionally filtered by zone."""
        events = sorted(store.list_all(), key=lambda record: record.timestamp, reverse=True)

        if zone is not None:
            events = [record for record in events if record.zone == zone]

        if limit is not None:
            if limit < 0:
                raise HTTPException(status_code=400, detail="limit must be 0 or greater")
            events = events[:limit]

        return events

    @api.get("/events/{event_id}", response_model=EventMetadata)
    def get_event(event_id: str):
        """One event's metadata, or 404 if there is no such event."""
        return find_event(event_id)

    @api.get("/events/{event_id}/snapshot")
    def get_event_snapshot(event_id: str) -> FileResponse:
        """The JPEG taken when the event started."""
        return serve_file(find_event(event_id).snapshot_path, SNAPSHOT_MEDIA_TYPE, "snapshot")

    @api.get("/events/{event_id}/video")
    def get_event_video(event_id: str) -> FileResponse:
        """The MP4 recorded while the event was in progress."""
        return serve_file(find_event(event_id).video_path, VIDEO_MEDIA_TYPE, "video")

    return api


def _is_inside(path: Path, root: Path) -> bool:
    """True if path sits under root, once both are made absolute."""
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False

    return True


# The application uvicorn serves. Built lazily by create_app() so tests can
# make their own instance against temporary storage.
app = create_app()

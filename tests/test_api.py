"""API tests. A real MetadataStore over a TemporaryDirectory is served through
FastAPI's TestClient: no server, no camera, no YOLO and no real user files.
"""

import tempfile
import unittest
from datetime import datetime
from pathlib import Path

try:
    from fastapi.testclient import TestClient

    from src.api.app import create_app
except ImportError:  # FastAPI/httpx missing: the rest of the suite still runs.
    TestClient = None
    create_app = None

from src.storage.metadata import EventMetadata, MetadataStore
from src.storage.storage_manager import StorageManager

WHEN = datetime(2026, 8, 28, 10, 0, 0)
LATER = datetime(2026, 8, 28, 11, 0, 0)


@unittest.skipIf(TestClient is None, "FastAPI and httpx are required for the API tests")
class ApiTestCase(unittest.TestCase):
    """An API served from a throwaway storage root holding two events."""

    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp_dir.cleanup)
        self.root_dir = Path(self._temp_dir.name) / "events"

        self.storage_manager = StorageManager(self.root_dir)
        self.store = MetadataStore(self.storage_manager)
        self.day_dir = self.storage_manager.event_dir_for(WHEN)

        self.video_path = self.day_dir / "clip1.mp4"
        self.snapshot_path = self.day_dir / "shot1.jpg"
        self.video_path.write_bytes(b"fake mp4 bytes")
        self.snapshot_path.write_bytes(b"fake jpg bytes")

        self.event = self.save_event(
            "1",
            timestamp=WHEN.timestamp(),
            video_path=str(self.video_path),
            snapshot_path=str(self.snapshot_path),
        )
        # A second, newer event in another zone, with no files of its own.
        self.other_event = self.save_event(
            "2", timestamp=LATER.timestamp(), zone="backyard", when=LATER
        )

        self.client = TestClient(create_app(self.store))
        self.addCleanup(self.client.close)

    def save_event(self, event_id, when=None, **overrides):
        values = dict(
            event_id=event_id,
            event_type="person_entered_zone",
            timestamp=WHEN.timestamp(),
            track_id=1,
            label="person",
            zone="front_door",
            confidence=0.9,
            duration=12.5,
        )
        values.update(overrides)
        metadata = EventMetadata(**values)
        self.store.save(metadata, when=when or WHEN)
        return metadata


class TestHealth(ApiTestCase):
    def test_reports_ok_with_a_summary_of_storage(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["event_count"], 2)
        self.assertGreater(body["storage_bytes"], 0)

    def test_works_on_empty_storage(self):
        empty_root = Path(self._temp_dir.name) / "empty"
        client = TestClient(create_app(MetadataStore(StorageManager(empty_root))))
        self.addCleanup(client.close)

        body = client.get("/health").json()

        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["event_count"], 0)


class TestListEvents(ApiTestCase):
    def test_returns_every_stored_event(self):
        response = self.client.get("/events")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()), 2)

    def test_events_are_newest_first(self):
        body = self.client.get("/events").json()

        self.assertEqual([record["event_id"] for record in body], ["2", "1"])

    def test_each_event_carries_every_field(self):
        body = self.client.get("/events").json()

        self.assertEqual(
            set(body[0]),
            {
                "event_id",
                "event_type",
                "timestamp",
                "track_id",
                "label",
                "zone",
                "confidence",
                "duration",
                "video_path",
                "snapshot_path",
            },
        )

    def test_can_be_filtered_by_zone(self):
        body = self.client.get("/events", params={"zone": "backyard"}).json()

        self.assertEqual([record["event_id"] for record in body], ["2"])

    def test_unknown_zone_returns_an_empty_list(self):
        self.assertEqual(self.client.get("/events", params={"zone": "attic"}).json(), [])

    def test_can_be_limited(self):
        body = self.client.get("/events", params={"limit": 1}).json()

        self.assertEqual([record["event_id"] for record in body], ["2"])

    def test_a_negative_limit_is_rejected(self):
        self.assertEqual(self.client.get("/events", params={"limit": -1}).status_code, 400)

    def test_empty_storage_returns_an_empty_list(self):
        empty_root = Path(self._temp_dir.name) / "empty"
        client = TestClient(create_app(MetadataStore(StorageManager(empty_root))))
        self.addCleanup(client.close)

        self.assertEqual(client.get("/events").json(), [])


class TestGetEvent(ApiTestCase):
    def test_returns_the_matching_event(self):
        response = self.client.get("/events/1")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["event_id"], "1")
        self.assertEqual(body["zone"], "front_door")
        self.assertEqual(body["duration"], 12.5)
        self.assertEqual(body["video_path"], str(self.video_path))

    def test_unknown_event_returns_404(self):
        response = self.client.get("/events/does-not-exist")

        self.assertEqual(response.status_code, 404)
        self.assertIn("not found", response.json()["detail"])


class TestEventFiles(ApiTestCase):
    def test_snapshot_is_served_as_a_jpeg(self):
        response = self.client.get("/events/1/snapshot")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/jpeg")
        self.assertEqual(response.content, b"fake jpg bytes")

    def test_video_is_served_as_an_mp4(self):
        response = self.client.get("/events/1/video")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "video/mp4")
        self.assertEqual(response.content, b"fake mp4 bytes")

    def test_event_without_a_snapshot_returns_404(self):
        self.assertEqual(self.client.get("/events/2/snapshot").status_code, 404)

    def test_event_without_a_video_returns_404(self):
        self.assertEqual(self.client.get("/events/2/video").status_code, 404)

    def test_unknown_event_files_return_404(self):
        self.assertEqual(self.client.get("/events/nope/snapshot").status_code, 404)
        self.assertEqual(self.client.get("/events/nope/video").status_code, 404)

    def test_missing_file_on_disk_returns_404(self):
        self.video_path.unlink()

        response = self.client.get("/events/1/video")

        self.assertEqual(response.status_code, 404)
        self.assertIn("missing", response.json()["detail"])

    def test_a_path_outside_the_storage_root_is_refused(self):
        outside = Path(self._temp_dir.name) / "secret.mp4"
        outside.write_bytes(b"private")
        self.save_event("3", video_path=str(outside))

        response = self.client.get("/events/3/video")

        self.assertEqual(response.status_code, 404)
        self.assertNotIn(b"private", response.content)


class TestReadOnly(ApiTestCase):
    def test_writing_methods_are_not_allowed(self):
        self.assertEqual(self.client.post("/events", json={}).status_code, 405)
        self.assertEqual(self.client.delete("/events/1").status_code, 405)
        self.assertEqual(self.client.put("/events/1", json={}).status_code, 405)

    def test_reading_does_not_change_stored_events(self):
        before = self.storage_manager.get_usage_bytes()

        self.client.get("/events")
        self.client.get("/events/1")
        self.client.get("/events/1/video")

        self.assertEqual(self.storage_manager.get_usage_bytes(), before)
        self.assertEqual(len(self.store.list_all()), 2)


if __name__ == "__main__":
    unittest.main()

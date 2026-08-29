"""Dashboard tests.

The page is plain HTML with vanilla JS, so these tests check what the server
sends: that the route serves the page, that the page asks the existing API for
its data, and that it can render every field the API returns. The JS itself is
not executed here (there is no browser in the test suite).
"""

import tempfile
import unittest
from datetime import datetime
from pathlib import Path

try:
    from fastapi.testclient import TestClient

    from src.api.app import DASHBOARD_FILE, create_app
except ImportError:  # FastAPI/httpx missing: the rest of the suite still runs.
    TestClient = None
    create_app = None
    DASHBOARD_FILE = None

from src.storage.metadata import EventMetadata, MetadataStore
from src.storage.storage_manager import StorageManager

WHEN = datetime(2026, 8, 28, 10, 0, 0)


@unittest.skipIf(TestClient is None, "FastAPI and httpx are required for the dashboard tests")
class DashboardTestCase(unittest.TestCase):
    """A dashboard served from a throwaway storage root."""

    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp_dir.cleanup)
        self.root_dir = Path(self._temp_dir.name) / "events"

        self.storage_manager = StorageManager(self.root_dir)
        self.store = MetadataStore(self.storage_manager)
        self.day_dir = self.storage_manager.event_dir_for(WHEN)

        self.client = TestClient(create_app(self.store))
        self.addCleanup(self.client.close)

    def save_event(self, event_id="1", **overrides):
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
        self.store.save(metadata, when=WHEN)
        return metadata

    def save_event_with_media(self, event_id="1"):
        video_path = self.day_dir / f"clip{event_id}.mp4"
        snapshot_path = self.day_dir / f"shot{event_id}.jpg"
        video_path.write_bytes(b"fake mp4 bytes")
        snapshot_path.write_bytes(b"fake jpg bytes")
        return self.save_event(
            event_id, video_path=str(video_path), snapshot_path=str(snapshot_path)
        )

    def page(self):
        return self.client.get("/").text


class TestDashboardRoute(DashboardTestCase):
    def test_root_serves_the_page_as_html(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])

    def test_page_is_a_complete_html_document(self):
        page = self.page()

        self.assertIn("<!doctype html>", page.lower())
        self.assertIn("AI Home Surveillance", page)
        self.assertIn("</html>", page)

    def test_page_is_served_even_with_no_events_stored(self):
        self.assertEqual(self.client.get("/").status_code, 200)

    def test_page_matches_the_file_on_disk(self):
        self.assertEqual(self.page(), DASHBOARD_FILE.read_text(encoding="utf-8"))


class TestDashboardContent(DashboardTestCase):
    def test_page_loads_its_data_from_the_events_endpoint(self):
        self.assertIn("/events?limit=50", self.page())

    def test_page_links_snapshots_and_videos_through_the_api(self):
        page = self.page()

        self.assertIn("/snapshot", page)
        self.assertIn("/video", page)

    def test_page_renders_every_required_field(self):
        page = self.page()

        for field in ("event_type", "timestamp", "zone", "track_id", "confidence", "duration"):
            with self.subTest(field=field):
                self.assertIn(field, page)

    def test_page_has_an_empty_state_message(self):
        self.assertIn("No events recorded yet.", self.page())

    def test_page_handles_missing_media_gracefully(self):
        page = self.page()

        self.assertIn("No snapshot", page)
        self.assertIn("No video recorded", page)
        self.assertIn("onerror", page)  # a snapshot that 404s is replaced

    def test_page_needs_no_build_system(self):
        page = self.page()

        self.assertNotIn("react", page.lower())
        self.assertNotIn("<script src=", page.lower())  # no external bundles


class TestDashboardUsesTheExistingApi(DashboardTestCase):
    """The page is static; its data comes from the endpoints already tested."""

    def test_events_endpoint_feeds_the_dashboard_newest_first(self):
        self.save_event("1", timestamp=WHEN.timestamp())
        self.save_event("2", timestamp=WHEN.timestamp() + 60)

        body = self.client.get("/events?limit=50").json()

        self.assertEqual([record["event_id"] for record in body], ["2", "1"])

    def test_snapshot_and_video_links_resolve_for_an_event_with_media(self):
        self.save_event_with_media("1")

        self.assertEqual(self.client.get("/events/1/snapshot").status_code, 200)
        self.assertEqual(self.client.get("/events/1/video").status_code, 200)

    def test_snapshot_and_video_links_404_for_an_event_without_media(self):
        self.save_event("1")

        self.assertEqual(self.client.get("/events/1/snapshot").status_code, 404)
        self.assertEqual(self.client.get("/events/1/video").status_code, 404)

    def test_dashboard_route_does_not_change_stored_events(self):
        self.save_event_with_media("1")
        before = self.storage_manager.get_usage_bytes()

        self.client.get("/")

        self.assertEqual(self.storage_manager.get_usage_bytes(), before)
        self.assertEqual(len(self.store.list_all()), 1)

    def test_existing_api_endpoints_still_work(self):
        self.save_event("1")

        self.assertEqual(self.client.get("/health").json()["status"], "ok")
        self.assertEqual(self.client.get("/events/1").json()["event_id"], "1")

    def test_dashboard_is_read_only(self):
        self.assertEqual(self.client.post("/", json={}).status_code, 405)
        self.assertEqual(self.client.delete("/").status_code, 405)


if __name__ == "__main__":
    unittest.main()

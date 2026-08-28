"""Metadata store tests. Everything runs against a throwaway temp directory,
so no test ever touches real user files.
"""

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from src.storage.metadata import EventMetadata, MetadataError, MetadataStore
from src.storage.storage_manager import StorageManager

WHEN = datetime(2026, 8, 5, 10, 30)


def make_metadata(event_id="evt1", **overrides):
    """An EventMetadata with sensible defaults, for tests that only care about one field."""
    values = dict(
        event_id=event_id,
        event_type="person_entered_zone",
        timestamp=WHEN.timestamp(),
        track_id=1,
        label="person",
        zone="front_door",
        confidence=0.92,
    )
    values.update(overrides)
    return EventMetadata(**values)


class MetadataTestCase(unittest.TestCase):
    """Gives every test its own throwaway storage root."""

    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp_dir.cleanup)
        self.root_dir = Path(self._temp_dir.name) / "events"
        self.storage_manager = StorageManager(self.root_dir)
        self.store = MetadataStore(self.storage_manager)


class TestEventMetadata(unittest.TestCase):
    def test_optional_fields_default_to_none(self):
        metadata = make_metadata()

        self.assertIsNone(metadata.duration)
        self.assertIsNone(metadata.video_path)
        self.assertIsNone(metadata.snapshot_path)

    def test_round_trips_through_a_dict(self):
        metadata = make_metadata(duration=12.5, video_path="a.mp4", snapshot_path="a.jpg")

        self.assertEqual(EventMetadata.from_dict(metadata.to_dict()), metadata)

    def test_from_dict_rejects_missing_required_fields(self):
        data = make_metadata().to_dict()
        del data["zone"]

        with self.assertRaises(MetadataError):
            EventMetadata.from_dict(data)


class TestSave(MetadataTestCase):
    def test_writes_json_into_the_day_directory(self):
        path = self.store.save(make_metadata(), when=WHEN)

        self.assertEqual(path, self.root_dir / "2026" / "08" / "05" / "evt1.json")
        self.assertEqual(json.loads(path.read_text())["zone"], "front_door")

    def test_defaults_the_day_to_the_event_timestamp(self):
        path = self.store.save(make_metadata())

        expected_day = datetime.fromtimestamp(WHEN.timestamp()).strftime("%Y/%m/%d")
        self.assertEqual(path.parent, self.root_dir.joinpath(*expected_day.split("/")))

    def test_stores_optional_fields_when_present(self):
        path = self.store.save(
            make_metadata(duration=8.0, video_path="a.mp4", snapshot_path="a.jpg"), when=WHEN
        )

        data = json.loads(path.read_text())
        self.assertEqual(data["duration"], 8.0)
        self.assertEqual(data["video_path"], "a.mp4")
        self.assertEqual(data["snapshot_path"], "a.jpg")

    def test_stores_optional_fields_as_null_when_absent(self):
        path = self.store.save(make_metadata(), when=WHEN)

        self.assertIsNone(json.loads(path.read_text())["duration"])


class TestLoad(MetadataTestCase):
    def test_reads_back_what_was_saved(self):
        metadata = make_metadata(duration=3.0)
        self.store.save(metadata, when=WHEN)

        self.assertEqual(self.store.load("evt1", WHEN), metadata)

    def test_missing_file_raises_a_clear_error(self):
        with self.assertRaises(MetadataError):
            self.store.load("nope", WHEN)

    def test_malformed_json_raises_a_clear_error(self):
        path = self.storage_manager.metadata_path("broken", WHEN)
        path.write_text("{not json", encoding="utf-8")

        with self.assertRaises(MetadataError):
            self.store.load("broken", WHEN)

    def test_json_that_is_not_an_object_raises_a_clear_error(self):
        path = self.storage_manager.metadata_path("listy", WHEN)
        path.write_text("[1, 2, 3]", encoding="utf-8")

        with self.assertRaises(MetadataError):
            self.store.load("listy", WHEN)

    def test_json_missing_a_field_raises_a_clear_error(self):
        data = make_metadata("partial").to_dict()
        del data["confidence"]
        self.storage_manager.metadata_path("partial", WHEN).write_text(json.dumps(data))

        with self.assertRaises(MetadataError):
            self.store.load("partial", WHEN)


class TestListing(MetadataTestCase):
    def test_list_for_day_returns_every_event_that_day(self):
        self.store.save(make_metadata("evt1"), when=WHEN)
        self.store.save(make_metadata("evt2"), when=WHEN)

        found = self.store.list_for_day(WHEN)

        self.assertEqual([record.event_id for record in found], ["evt1", "evt2"])

    def test_list_for_day_is_empty_when_nothing_was_recorded(self):
        self.assertEqual(self.store.list_for_day(WHEN), [])

    def test_list_for_day_skips_malformed_files(self):
        self.store.save(make_metadata("good"), when=WHEN)
        self.storage_manager.metadata_path("bad", WHEN).write_text("{oops")

        found = self.store.list_for_day(WHEN)

        self.assertEqual([record.event_id for record in found], ["good"])

    def test_list_all_spans_multiple_days(self):
        self.store.save(make_metadata("evt1"), when=datetime(2026, 8, 5))
        self.store.save(make_metadata("evt2"), when=datetime(2026, 9, 1))

        found = self.store.list_all()

        self.assertEqual({record.event_id for record in found}, {"evt1", "evt2"})

    def test_list_all_is_empty_when_root_does_not_exist(self):
        self.assertEqual(self.store.list_all(), [])


if __name__ == "__main__":
    unittest.main()

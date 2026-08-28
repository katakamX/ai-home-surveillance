"""StorageManager tests. Everything runs against a throwaway temp directory,
so no test ever touches real user files.
"""

import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from src.storage.storage_manager import StorageManager, StorageManagerError


class StorageManagerTestCase(unittest.TestCase):
    """Gives every test its own throwaway root directory."""

    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp_dir.cleanup)
        self.root_dir = Path(self._temp_dir.name) / "events"
        self.manager = StorageManager(self.root_dir)


class TestEventDirectories(StorageManagerTestCase):
    def test_creates_nothing_on_disk_until_used(self):
        StorageManager(self.root_dir)

        self.assertFalse(self.root_dir.exists())

    def test_event_dir_for_uses_year_month_day_layout(self):
        when = datetime(2026, 8, 5, 10, 30)

        event_dir = self.manager.event_dir_for(when)

        self.assertEqual(event_dir, self.root_dir / "2026" / "08" / "05")
        self.assertTrue(event_dir.is_dir())

    def test_event_dir_for_is_safe_to_call_repeatedly(self):
        when = datetime(2026, 8, 5, 10, 30)

        first = self.manager.event_dir_for(when)
        second = self.manager.event_dir_for(when)

        self.assertEqual(first, second)
        self.assertTrue(second.is_dir())

    def test_accepts_a_string_root_dir(self):
        manager = StorageManager(str(self.root_dir))

        self.assertEqual(manager.root_dir, self.root_dir)


class TestFilePaths(StorageManagerTestCase):
    def test_video_path_lives_under_the_day_directory(self):
        when = datetime(2026, 1, 2, 9, 0)

        path = self.manager.video_path("evt1", when)

        self.assertEqual(path, self.root_dir / "2026" / "01" / "02" / "evt1.mp4")
        self.assertTrue(path.parent.is_dir())

    def test_snapshot_path_lives_under_the_day_directory(self):
        when = datetime(2026, 1, 2, 9, 0)

        path = self.manager.snapshot_path("evt1", when)

        self.assertEqual(path, self.root_dir / "2026" / "01" / "02" / "evt1.jpg")

    def test_metadata_path_lives_under_the_day_directory(self):
        when = datetime(2026, 1, 2, 9, 0)

        path = self.manager.metadata_path("evt1", when)

        self.assertEqual(path, self.root_dir / "2026" / "01" / "02" / "evt1.json")


class TestUsage(StorageManagerTestCase):
    def test_usage_is_zero_when_nothing_exists(self):
        self.assertEqual(self.manager.get_usage_bytes(), 0)

    def test_usage_sums_file_sizes(self):
        when = datetime(2026, 1, 2, 9, 0)
        video_path = self.manager.video_path("evt1", when)
        snapshot_path = self.manager.snapshot_path("evt2", when)
        video_path.write_bytes(b"a" * 100)
        snapshot_path.write_bytes(b"b" * 50)

        self.assertEqual(self.manager.get_usage_bytes(), 150)


class TestRetention(StorageManagerTestCase):
    def _make_day_dir(self, year, month, day):
        day_dir = self.root_dir / f"{year:04d}" / f"{month:02d}" / f"{day:02d}"
        day_dir.mkdir(parents=True)
        (day_dir / "evt.mp4").write_bytes(b"x")
        return day_dir

    def test_returns_empty_list_when_root_does_not_exist(self):
        self.assertEqual(self.manager.delete_old_events(7), [])

    def test_deletes_directories_older_than_retention(self):
        old_dir = self._make_day_dir(2020, 1, 1)
        recent_dir = self._make_day_dir(2026, 8, 27)
        now = datetime(2026, 8, 28)

        deleted = self.manager.delete_old_events(retention_days=7, now=now)

        self.assertEqual(deleted, [old_dir])
        self.assertFalse(old_dir.exists())
        self.assertTrue(recent_dir.exists())

    def test_keeps_directories_within_retention(self):
        recent_dir = self._make_day_dir(2026, 8, 25)
        now = datetime(2026, 8, 28)

        deleted = self.manager.delete_old_events(retention_days=7, now=now)

        self.assertEqual(deleted, [])
        self.assertTrue(recent_dir.exists())

    def test_ignores_directories_that_do_not_parse_as_dates(self):
        junk_dir = self.root_dir / "not-a-year" / "nope" / "nah"
        junk_dir.mkdir(parents=True)
        now = datetime(2026, 8, 28)

        deleted = self.manager.delete_old_events(retention_days=0, now=now)

        self.assertEqual(deleted, [])
        self.assertTrue(junk_dir.exists())

    def test_rejects_negative_retention(self):
        with self.assertRaises(StorageManagerError):
            self.manager.delete_old_events(retention_days=-1)


if __name__ == "__main__":
    unittest.main()

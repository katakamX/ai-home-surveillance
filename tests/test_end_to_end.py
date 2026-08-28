"""End-to-end pipeline tests: synthetic person in, real files out.

Camera, Detector and Tracker are mocked (no webcam and no YOLO model are
needed), but everything downstream is real: a real EventEngine turns synthetic
TrackedObject positions into enter/exit events, and a real VideoRecorder,
StorageManager and MetadataStore write real .mp4, .jpg and .json files into a
TemporaryDirectory.

Recorders are wrapped in MagicMock(wraps=...) so the tests can both count
calls and keep the real OpenCV behaviour behind them.
"""

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

try:
    import cv2
    import numpy
except ImportError:  # OpenCV/numpy missing: these tests need real file writing.
    cv2 = None
    numpy = None

from src.events.events import EVENT_PERSON_ENTERED_ZONE, EVENT_PERSON_EXITED_ZONE, EventEngine, Zone
from src.pipeline.pipeline import SurveillancePipeline
from src.storage.metadata import MetadataStore
from src.storage.recorder import VideoRecorder
from src.storage.storage_manager import StorageManager
from src.tracking.tracker import TrackedObject

FRAME_SIZE = (640, 480)
FPS = 20.0

ZONE = Zone("front_door", (100, 100, 500, 400))
INSIDE_BOX = (250, 200, 350, 300)  # centre (300, 250): inside ZONE
OUTSIDE_BOX = (0, 0, 40, 40)  # centre (20, 20): outside ZONE

# A fixed clock, so durations and the YYYY/MM/DD directory are deterministic.
START_TIME = datetime(2026, 8, 28, 10, 0, 0)


def make_frame():
    """One blank BGR frame of the size the recorder expects."""
    return numpy.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=numpy.uint8)


def tracked_person(box=INSIDE_BOX, track_id=1):
    return TrackedObject(track_id=track_id, box=box, label="person", confidence=0.88)


@unittest.skipIf(cv2 is None, "OpenCV/numpy are required for end-to-end file writing")
class EndToEndTestCase(unittest.TestCase):
    """A whole pipeline writing into a throwaway directory."""

    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp_dir.cleanup)
        self.root_dir = Path(self._temp_dir.name) / "events"

        self.storage_manager = StorageManager(self.root_dir)
        self.store = MetadataStore(self.storage_manager)
        self.day_dir = self.storage_manager.event_dir_for(START_TIME)

        self.recorders = []  # every recorder the pipeline built, in order

        self.camera = MagicMock()
        self.detector = MagicMock()
        self.detector.detect.return_value = []
        self.tracker = MagicMock()
        self.event_engine = EventEngine(zones=[ZONE])

        self.pipeline = SurveillancePipeline(
            camera=self.camera,
            detector=self.detector,
            tracker=self.tracker,
            event_engine=self.event_engine,
            recorder_factory=self._recorder_factory,
            metadata_store=self.store,
        )
        # Cleanups run last-registered-first, so this releases any recording
        # still open before the temporary directory is deleted. On Windows an
        # open VideoWriter keeps a lock on its .mp4 and blocks the cleanup.
        self.addCleanup(self.pipeline.close)

    def _recorder_factory(self):
        """A real VideoRecorder writing into today's storage directory, spied on."""
        recorder = MagicMock(
            wraps=VideoRecorder(self.day_dir, fps=FPS, frame_size=FRAME_SIZE)
        )
        self.recorders.append(recorder)
        return recorder

    def feed(self, tracked_objects, seconds):
        """Run one frame through the pipeline at START_TIME + seconds."""
        self.tracker.update.return_value = tracked_objects
        timestamp = START_TIME.timestamp() + seconds
        return self.pipeline.process_frame(make_frame(), timestamp=timestamp)

    def run_full_visit(self):
        """Person enters, is recorded for 3 frames, then leaves. Returns all events."""
        events = []
        _, enter_events = self.feed([tracked_person()], seconds=0.0)
        events += enter_events
        for index, offset in enumerate((1.0, 2.0), start=1):
            self.feed([tracked_person()], seconds=offset)
        # The track disappears entirely, so EventEngine reports the exit at once.
        _, exit_events = self.feed([], seconds=5.0)
        events += exit_events
        return events


class TestFullVisitProducesFiles(EndToEndTestCase):
    def test_enter_and_exit_events_are_produced_in_order(self):
        events = self.run_full_visit()

        self.assertEqual(
            [event.event_type for event in events],
            [EVENT_PERSON_ENTERED_ZONE, EVENT_PERSON_EXITED_ZONE],
        )

    def test_one_recorder_is_created_started_and_stopped_once(self):
        self.run_full_visit()

        self.assertEqual(len(self.recorders), 1)
        recorder = self.recorders[0]
        recorder.start.assert_called_once_with()
        recorder.save_snapshot.assert_called_once()
        recorder.stop.assert_called_once_with()

    def test_every_frame_of_the_visit_is_written(self):
        self.run_full_visit()

        # Frames at 0.0, 1.0 and 2.0 are written; the frame carrying the exit is not.
        self.assertEqual(self.recorders[0].write.call_count, 3)

    def test_a_real_mp4_and_jpg_land_in_the_day_directory(self):
        self.run_full_visit()

        videos = list(self.day_dir.glob("*.mp4"))
        snapshots = list(self.day_dir.glob("*.jpg"))

        self.assertEqual(len(videos), 1)
        self.assertEqual(len(snapshots), 1)
        self.assertGreater(videos[0].stat().st_size, 0)
        self.assertGreater(snapshots[0].stat().st_size, 0)

    def test_storage_usage_reflects_the_written_files(self):
        self.run_full_visit()

        self.assertGreater(self.storage_manager.get_usage_bytes(), 0)


class TestMetadataForACompletedVisit(EndToEndTestCase):
    def setUp(self):
        super().setUp()
        self.events = self.run_full_visit()
        self.enter_event, self.exit_event = self.events
        self.metadata_path = self.storage_manager.metadata_path(
            str(self.enter_event.event_id), START_TIME
        )

    def test_exactly_one_metadata_file_is_written(self):
        self.assertEqual(len(list(self.day_dir.glob("*.json"))), 1)
        self.assertTrue(self.metadata_path.is_file())

    def test_metadata_file_is_valid_json_with_every_field(self):
        data = json.loads(self.metadata_path.read_text(encoding="utf-8"))

        self.assertEqual(
            set(data),
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

    def test_metadata_describes_the_visit(self):
        record = self.store.load(str(self.enter_event.event_id), START_TIME)

        self.assertEqual(record.event_id, str(self.enter_event.event_id))
        self.assertEqual(record.event_type, EVENT_PERSON_ENTERED_ZONE)
        self.assertEqual(record.timestamp, self.enter_event.timestamp)
        self.assertEqual(record.track_id, 1)
        self.assertEqual(record.label, "person")
        self.assertEqual(record.zone, "front_door")
        self.assertEqual(record.confidence, 0.88)

    def test_duration_is_the_time_between_enter_and_exit(self):
        record = self.store.load(str(self.enter_event.event_id), START_TIME)

        self.assertAlmostEqual(record.duration, 5.0)

    def test_recorded_video_path_points_at_the_real_clip(self):
        record = self.store.load(str(self.enter_event.event_id), START_TIME)
        video_path = Path(record.video_path)

        self.assertTrue(video_path.is_file())
        self.assertEqual(video_path.suffix, ".mp4")
        self.assertEqual(video_path.parent, self.day_dir)
        self.assertEqual([video_path], list(self.day_dir.glob("*.mp4")))

    def test_recorded_snapshot_path_points_at_the_real_image(self):
        record = self.store.load(str(self.enter_event.event_id), START_TIME)
        snapshot_path = Path(record.snapshot_path)

        self.assertTrue(snapshot_path.is_file())
        self.assertEqual(snapshot_path.suffix, ".jpg")
        self.assertEqual(snapshot_path.parent, self.day_dir)
        self.assertEqual([snapshot_path], list(self.day_dir.glob("*.jpg")))

    def test_the_visit_is_listed_by_the_store(self):
        found = self.store.list_for_day(START_TIME)

        self.assertEqual([record.event_id for record in found], [str(self.enter_event.event_id)])


class TestIncompleteVisit(EndToEndTestCase):
    def test_close_stops_the_recorder_but_writes_no_metadata(self):
        self.feed([tracked_person()], seconds=0.0)
        self.feed([tracked_person()], seconds=1.0)

        self.pipeline.close()

        self.recorders[0].stop.assert_called_once_with()
        self.assertEqual(self.store.list_all(), [])
        self.assertEqual(list(self.day_dir.glob("*.json")), [])
        # The clip itself was still written and closed properly.
        self.assertEqual(len(list(self.day_dir.glob("*.mp4"))), 1)

    def test_close_after_a_finished_visit_does_not_stop_twice(self):
        self.run_full_visit()

        self.pipeline.close()

        self.recorders[0].stop.assert_called_once_with()
        self.assertEqual(len(self.store.list_all()), 1)


class TestNoDuplicateRecordings(EndToEndTestCase):
    def test_staying_in_the_zone_does_not_start_a_second_recording(self):
        for offset in range(6):
            self.feed([tracked_person()], seconds=float(offset))

        self.assertEqual(len(self.recorders), 1)
        self.assertEqual(len(list(self.day_dir.glob("*.mp4"))), 1)
        self.assertEqual(len(list(self.day_dir.glob("*.jpg"))), 1)

    def test_a_duplicate_enter_event_is_ignored(self):
        # EventEngine never emits this, so it is injected directly: the pipeline
        # must not leak a second recorder if it ever happens.
        _, events = self.feed([tracked_person()], seconds=0.0)
        enter_event = events[0]

        self.pipeline._handle_event(enter_event, make_frame())

        self.assertEqual(len(self.recorders), 1)
        self.recorders[0].start.assert_called_once_with()

    def test_leaving_and_returning_starts_a_second_recording(self):
        self.run_full_visit()
        self.feed([tracked_person()], seconds=6.0)
        self.feed([], seconds=7.0)

        self.assertEqual(len(self.recorders), 2)
        self.assertEqual(len(list(self.day_dir.glob("*.mp4"))), 2)
        self.assertEqual(len(self.store.list_for_day(START_TIME)), 2)


class TestPipelineRunLoop(EndToEndTestCase):
    def test_run_drives_a_whole_visit_from_the_camera(self):
        frames = [(True, make_frame()) for _ in range(4)] + [(False, None)]
        self.camera.read.side_effect = frames
        # Person present for the first three frames, gone for the fourth.
        self.tracker.update.side_effect = [
            [tracked_person()],
            [tracked_person()],
            [tracked_person()],
            [],
        ]

        self.pipeline.run()

        self.assertEqual(len(self.recorders), 1)
        self.recorders[0].stop.assert_called_once_with()
        self.assertEqual(len(self.store.list_all()), 1)


if __name__ == "__main__":
    unittest.main()

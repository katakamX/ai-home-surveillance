"""Pipeline tests. Camera, Detector, Tracker, EventEngine and VideoRecorder are
all mocks: this only checks the wiring, not any real detection/tracking/OpenCV
behaviour (those already have their own test suites).
"""

import io
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, call

from src.alerts.alert import AlertDispatcher, ConsoleAlertHandler
from src.events.events import EVENT_PERSON_ENTERED_ZONE, EVENT_PERSON_EXITED_ZONE, Event
from src.pipeline.pipeline import SurveillancePipeline, build_recorder_factory
from src.storage.metadata import MetadataError, MetadataStore
from src.storage.recorder import RecorderError
from src.storage.storage_manager import StorageManager


def make_event(event_type, track_id=1, zone="front_door", event_id=1, timestamp=100.0):
    return Event(
        event_id=event_id,
        timestamp=timestamp,
        event_type=event_type,
        track_id=track_id,
        label="person",
        zone=zone,
        confidence=0.9,
    )


def make_pipeline(
    events_per_call=None, recorders=None, metadata_store=None, alert_dispatcher=None
):
    """Build a SurveillancePipeline with every dependency mocked.

    events_per_call: list of event-lists, one per process_frame() call, fed to
    event_engine.update() in order. recorders: list of recorder mocks handed
    out by the recorder_factory in order, one per call.
    """
    camera = MagicMock()
    detector = MagicMock()
    detector.detect.return_value = []
    tracker = MagicMock()
    tracker.update.return_value = []
    event_engine = MagicMock()
    if events_per_call is not None:
        event_engine.update.side_effect = events_per_call

    if recorders is not None:
        recorder_factory = MagicMock(side_effect=recorders)
    else:
        recorder_factory = MagicMock(side_effect=lambda: MagicMock())

    pipeline = SurveillancePipeline(
        camera=camera,
        detector=detector,
        tracker=tracker,
        event_engine=event_engine,
        recorder_factory=recorder_factory,
        metadata_store=metadata_store,
        alert_dispatcher=alert_dispatcher,
    )
    return pipeline, camera, detector, tracker, event_engine, recorder_factory


class TestProcessFrameWiring(unittest.TestCase):
    def test_frame_flows_through_detector_tracker_and_event_engine(self):
        pipeline, camera, detector, tracker, event_engine, _ = make_pipeline(
            events_per_call=[[]]
        )
        detector.detect.return_value = ["detection"]
        tracker.update.return_value = ["tracked"]

        pipeline.process_frame("frame", timestamp=42.0)

        detector.detect.assert_called_once_with("frame")
        tracker.update.assert_called_once_with(["detection"])
        event_engine.update.assert_called_once_with(["tracked"], 42.0)

    def test_timestamp_defaults_when_not_given(self):
        pipeline, *_ = make_pipeline(events_per_call=[[]])

        pipeline.process_frame("frame")

        # No exception, and a real float timestamp was passed through.
        args = pipeline.event_engine.update.call_args.args
        self.assertIsInstance(args[1], float)

    def test_returns_tracked_objects_and_events(self):
        events = [make_event(EVENT_PERSON_ENTERED_ZONE)]
        pipeline, camera, detector, tracker, event_engine, _ = make_pipeline(
            events_per_call=[events]
        )
        tracker.update.return_value = ["tracked"]

        tracked_objects, returned_events = pipeline.process_frame("frame")

        self.assertEqual(tracked_objects, ["tracked"])
        self.assertEqual(returned_events, events)


class TestEnterEventStartsRecording(unittest.TestCase):
    def test_enter_event_starts_and_snapshots_a_new_recorder(self):
        recorder = MagicMock()
        pipeline, *_rest, recorder_factory = make_pipeline(
            events_per_call=[[make_event(EVENT_PERSON_ENTERED_ZONE)]],
            recorders=[recorder],
        )

        pipeline.process_frame("frame1")

        recorder_factory.assert_called_once_with()
        recorder.start.assert_called_once_with()
        recorder.save_snapshot.assert_called_once_with("frame1")

    def test_enter_event_writes_the_triggering_frame_too(self):
        recorder = MagicMock()
        pipeline, *_rest, recorder_factory = make_pipeline(
            events_per_call=[[make_event(EVENT_PERSON_ENTERED_ZONE)]],
            recorders=[recorder],
        )

        pipeline.process_frame("frame1")

        recorder.write.assert_called_once_with("frame1")

    def test_separate_tracks_or_zones_get_separate_recorders(self):
        recorder_a = MagicMock()
        recorder_b = MagicMock()
        events = [
            make_event(EVENT_PERSON_ENTERED_ZONE, track_id=1, zone="front_door"),
            make_event(EVENT_PERSON_ENTERED_ZONE, track_id=2, zone="front_door"),
        ]
        pipeline, *_rest, recorder_factory = make_pipeline(
            events_per_call=[events], recorders=[recorder_a, recorder_b]
        )

        pipeline.process_frame("frame1")

        self.assertEqual(recorder_factory.call_count, 2)
        recorder_a.start.assert_called_once()
        recorder_b.start.assert_called_once()

    def test_duplicate_enter_for_same_key_is_ignored(self):
        recorder = MagicMock()
        events = [
            make_event(EVENT_PERSON_ENTERED_ZONE, event_id=1),
            make_event(EVENT_PERSON_ENTERED_ZONE, event_id=2),
        ]
        pipeline, *_rest, recorder_factory = make_pipeline(
            events_per_call=[events], recorders=[recorder]
        )

        pipeline.process_frame("frame1")

        recorder_factory.assert_called_once_with()
        recorder.start.assert_called_once()


class TestOngoingTrackingWritesFrames(unittest.TestCase):
    def test_frames_are_written_on_every_call_while_tracked(self):
        recorder = MagicMock()
        pipeline, *_rest, recorder_factory = make_pipeline(
            events_per_call=[[make_event(EVENT_PERSON_ENTERED_ZONE)], [], []],
            recorders=[recorder],
        )

        pipeline.process_frame("frame1")
        pipeline.process_frame("frame2")
        pipeline.process_frame("frame3")

        recorder.write.assert_has_calls([call("frame1"), call("frame2"), call("frame3")])
        self.assertEqual(recorder.write.call_count, 3)

    def test_no_recorder_activity_when_nothing_is_tracked(self):
        pipeline, *_rest, recorder_factory = make_pipeline(events_per_call=[[], []])

        pipeline.process_frame("frame1")
        pipeline.process_frame("frame2")

        recorder_factory.assert_not_called()


class TestExitEventStopsRecording(unittest.TestCase):
    def test_exit_event_stops_the_matching_recorder(self):
        recorder = MagicMock()
        pipeline, *_rest, recorder_factory = make_pipeline(
            events_per_call=[
                [make_event(EVENT_PERSON_ENTERED_ZONE)],
                [make_event(EVENT_PERSON_EXITED_ZONE)],
            ],
            recorders=[recorder],
        )

        pipeline.process_frame("frame1")
        pipeline.process_frame("frame2")

        recorder.stop.assert_called_once_with()

    def test_frame_that_carries_the_exit_event_is_not_written_after_stop(self):
        recorder = MagicMock()
        pipeline, *_rest, recorder_factory = make_pipeline(
            events_per_call=[
                [make_event(EVENT_PERSON_ENTERED_ZONE)],
                [make_event(EVENT_PERSON_EXITED_ZONE)],
            ],
            recorders=[recorder],
        )

        pipeline.process_frame("frame1")
        pipeline.process_frame("frame2")

        recorder.write.assert_called_once_with("frame1")

    def test_frames_after_exit_are_not_written_to_the_stopped_recorder(self):
        recorder = MagicMock()
        pipeline, *_rest, recorder_factory = make_pipeline(
            events_per_call=[
                [make_event(EVENT_PERSON_ENTERED_ZONE)],
                [make_event(EVENT_PERSON_EXITED_ZONE)],
                [],
            ],
            recorders=[recorder],
        )

        pipeline.process_frame("frame1")
        pipeline.process_frame("frame2")
        pipeline.process_frame("frame3")

        self.assertEqual(recorder.write.call_count, 1)

    def test_re_entering_the_same_zone_starts_a_fresh_recorder(self):
        first_recorder = MagicMock()
        second_recorder = MagicMock()
        pipeline, *_rest, recorder_factory = make_pipeline(
            events_per_call=[
                [make_event(EVENT_PERSON_ENTERED_ZONE, event_id=1)],
                [make_event(EVENT_PERSON_EXITED_ZONE, event_id=2)],
                [make_event(EVENT_PERSON_ENTERED_ZONE, event_id=3)],
            ],
            recorders=[first_recorder, second_recorder],
        )

        pipeline.process_frame("frame1")
        pipeline.process_frame("frame2")
        pipeline.process_frame("frame3")

        first_recorder.start.assert_called_once()
        first_recorder.stop.assert_called_once()
        second_recorder.start.assert_called_once()
        second_recorder.stop.assert_not_called()
        self.assertEqual(recorder_factory.call_count, 2)

    def test_exit_for_unknown_key_does_not_raise(self):
        pipeline, *_rest = make_pipeline(events_per_call=[[make_event(EVENT_PERSON_EXITED_ZONE)]])

        pipeline.process_frame("frame1")  # should not raise


class TestClose(unittest.TestCase):
    def test_close_stops_every_active_recording(self):
        recorder_a = MagicMock()
        recorder_b = MagicMock()
        events = [
            make_event(EVENT_PERSON_ENTERED_ZONE, track_id=1, zone="front_door"),
            make_event(EVENT_PERSON_ENTERED_ZONE, track_id=2, zone="backyard"),
        ]
        pipeline, *_rest = make_pipeline(events_per_call=[events], recorders=[recorder_a, recorder_b])

        pipeline.process_frame("frame1")
        pipeline.close()

        recorder_a.stop.assert_called_once()
        recorder_b.stop.assert_called_once()

    def test_close_is_safe_with_no_active_recordings(self):
        pipeline, *_rest = make_pipeline(events_per_call=[[]])

        pipeline.process_frame("frame1")
        pipeline.close()  # should not raise

    def test_close_is_safe_to_call_twice(self):
        recorder = MagicMock()
        pipeline, *_rest = make_pipeline(
            events_per_call=[[make_event(EVENT_PERSON_ENTERED_ZONE)]], recorders=[recorder]
        )

        pipeline.process_frame("frame1")
        pipeline.close()
        pipeline.close()

        recorder.stop.assert_called_once()


class TestRun(unittest.TestCase):
    def test_run_reads_and_processes_frames_until_camera_reports_failure(self):
        pipeline, camera, detector, tracker, event_engine, _ = make_pipeline(
            events_per_call=[[], [], []]
        )
        camera.read.side_effect = [(True, "frame1"), (True, "frame2"), (False, None)]

        pipeline.run()

        self.assertEqual(detector.detect.call_count, 2)
        self.assertEqual(camera.read.call_count, 3)

    def test_run_stops_after_max_frames(self):
        pipeline, camera, detector, tracker, event_engine, _ = make_pipeline(
            events_per_call=[[], [], []]
        )
        camera.read.return_value = (True, "frame")

        pipeline.run(max_frames=2)

        self.assertEqual(detector.detect.call_count, 2)

    def test_run_closes_active_recordings_when_the_camera_stops(self):
        recorder = MagicMock()
        pipeline, camera, detector, tracker, event_engine, _ = make_pipeline(
            events_per_call=[[make_event(EVENT_PERSON_ENTERED_ZONE)], []],
            recorders=[recorder],
        )
        camera.read.side_effect = [(True, "frame1"), (False, None)]

        pipeline.run()

        recorder.stop.assert_called_once()

    def test_run_closes_active_recordings_even_if_processing_raises(self):
        recorder = MagicMock()
        pipeline, camera, detector, tracker, event_engine, _ = make_pipeline(
            events_per_call=[[make_event(EVENT_PERSON_ENTERED_ZONE)]],
            recorders=[recorder],
        )
        camera.read.side_effect = [(True, "frame1"), RuntimeError("boom")]

        with self.assertRaises(RuntimeError):
            pipeline.run()

        recorder.stop.assert_called_once()


class TestBuildRecorderFactory(unittest.TestCase):
    def test_factory_builds_a_video_recorder_with_the_given_settings(self):
        from unittest.mock import patch

        with patch("src.pipeline.pipeline.VideoRecorder") as video_recorder_cls:
            factory = build_recorder_factory("data/events", fps=15.0, frame_size=(320, 240))
            factory()

            video_recorder_cls.assert_called_once_with("data/events", fps=15.0, frame_size=(320, 240))

    def test_factory_builds_a_fresh_recorder_on_every_call(self):
        from unittest.mock import patch

        with patch("src.pipeline.pipeline.VideoRecorder") as video_recorder_cls:
            factory = build_recorder_factory("data/events", fps=15.0, frame_size=(320, 240))
            factory()
            factory()

            self.assertEqual(video_recorder_cls.call_count, 2)


def make_recorder(video_path="video.mp4", snapshot_path="snap.jpg"):
    """A recorder mock that reports the paths it wrote, like the real one does."""
    recorder = MagicMock()
    recorder.start.return_value = Path(video_path)
    recorder.save_snapshot.return_value = Path(snapshot_path)
    return recorder


# Marks "use the test case's own store", so None can mean "no store at all".
USE_REAL_STORE = object()

PIPELINE_LOGGER = "src.pipeline.pipeline"

# Marks "use the test case's own dispatcher", so None can mean "no alerts at all".
USE_REAL_DISPATCHER = object()


class MetadataIntegrationTestCase(unittest.TestCase):
    """Pipeline plus a real MetadataStore writing into a throwaway directory."""

    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp_dir.cleanup)
        self.root_dir = Path(self._temp_dir.name) / "events"
        self.store = MetadataStore(StorageManager(self.root_dir))

        # Enter at 100.0, exit at 112.5: a 12.5 second visit.
        self.enter_event = make_event(EVENT_PERSON_ENTERED_ZONE, event_id=7, timestamp=100.0)
        self.exit_event = make_event(EVENT_PERSON_EXITED_ZONE, event_id=8, timestamp=112.5)
        self.when = datetime.fromtimestamp(100.0)

    def run_visit(self, recorder=None, store=USE_REAL_STORE):
        """Drive one full enter -> record -> exit cycle and return the recorder.

        store defaults to this test case's real MetadataStore; pass None to
        build a pipeline with no metadata store at all.
        """
        recorder = recorder or make_recorder()
        pipeline, *_rest = make_pipeline(
            events_per_call=[[self.enter_event], [], [self.exit_event]],
            recorders=[recorder],
            metadata_store=self.store if store is USE_REAL_STORE else store,
        )
        pipeline.process_frame("frame1")
        pipeline.process_frame("frame2")
        pipeline.process_frame("frame3")
        return recorder


class TestCompletedEventMetadata(MetadataIntegrationTestCase):
    def test_exit_saves_one_metadata_record(self):
        self.run_visit()

        self.assertEqual(len(self.store.list_all()), 1)

    def test_metadata_describes_the_completed_visit(self):
        self.run_visit()

        record = self.store.load("7", self.when)

        self.assertEqual(record.event_id, "7")
        self.assertEqual(record.event_type, EVENT_PERSON_ENTERED_ZONE)
        self.assertEqual(record.timestamp, 100.0)
        self.assertEqual(record.track_id, 1)
        self.assertEqual(record.label, "person")
        self.assertEqual(record.zone, "front_door")
        self.assertEqual(record.confidence, 0.9)

    def test_duration_is_the_time_between_enter_and_exit(self):
        self.run_visit()

        self.assertEqual(self.store.load("7", self.when).duration, 12.5)

    def test_metadata_records_the_recorder_paths(self):
        self.run_visit(recorder=make_recorder("clip.mp4", "shot.jpg"))

        record = self.store.load("7", self.when)

        self.assertEqual(record.video_path, str(Path("clip.mp4")))
        self.assertEqual(record.snapshot_path, str(Path("shot.jpg")))

    def test_no_metadata_is_saved_before_the_exit_event(self):
        pipeline, *_rest = make_pipeline(
            events_per_call=[[self.enter_event], []],
            recorders=[make_recorder()],
            metadata_store=self.store,
        )

        pipeline.process_frame("frame1")
        pipeline.process_frame("frame2")

        self.assertEqual(self.store.list_all(), [])

    def test_close_does_not_save_metadata_for_an_unfinished_visit(self):
        pipeline, *_rest = make_pipeline(
            events_per_call=[[self.enter_event]],
            recorders=[make_recorder()],
            metadata_store=self.store,
        )

        pipeline.process_frame("frame1")
        pipeline.close()

        self.assertEqual(self.store.list_all(), [])

    def test_exit_without_a_matching_enter_saves_nothing(self):
        pipeline, *_rest = make_pipeline(
            events_per_call=[[self.exit_event]], metadata_store=self.store
        )

        pipeline.process_frame("frame1")

        self.assertEqual(self.store.list_all(), [])

    def test_pipeline_without_a_metadata_store_still_records(self):
        recorder = self.run_visit(store=None)

        recorder.start.assert_called_once()
        recorder.stop.assert_called_once()
        self.assertEqual(self.store.list_all(), [])


class TestFailuresDoNotStopTheLoop(MetadataIntegrationTestCase):
    """Every failure here is logged and swallowed: the surveillance loop keeps running.

    assertLogs both checks that the failure was reported and keeps the
    expected tracebacks out of the test output.
    """

    def test_failing_recorder_start_skips_the_visit_without_raising(self):
        recorder = make_recorder()
        recorder.start.side_effect = RecorderError("no disk")
        pipeline, *_rest = make_pipeline(
            events_per_call=[[self.enter_event], [self.exit_event]],
            recorders=[recorder],
            metadata_store=self.store,
        )

        with self.assertLogs(PIPELINE_LOGGER, level="ERROR"):
            pipeline.process_frame("frame1")
            pipeline.process_frame("frame2")

        recorder.write.assert_not_called()
        recorder.stop.assert_not_called()
        self.assertEqual(self.store.list_all(), [])

    def test_failing_snapshot_still_records_and_saves_metadata(self):
        recorder = make_recorder()
        recorder.save_snapshot.side_effect = RecorderError("bad frame")

        with self.assertLogs(PIPELINE_LOGGER, level="ERROR"):
            self.run_visit(recorder=recorder)

        record = self.store.load("7", self.when)
        self.assertIsNone(record.snapshot_path)
        self.assertEqual(record.video_path, str(Path("video.mp4")))

    def test_failing_recorder_stop_still_saves_metadata(self):
        recorder = make_recorder()
        recorder.stop.side_effect = RecorderError("cannot close")

        with self.assertLogs(PIPELINE_LOGGER, level="ERROR"):
            self.run_visit(recorder=recorder)

        self.assertEqual(len(self.store.list_all()), 1)

    def test_failing_metadata_save_does_not_raise(self):
        broken_store = MagicMock()
        broken_store.save.side_effect = MetadataError("disk full")

        with self.assertLogs(PIPELINE_LOGGER, level="ERROR"):
            recorder = self.run_visit(store=broken_store)  # should not raise

        recorder.stop.assert_called_once()
        broken_store.save.assert_called_once()


class AlertTestCase(unittest.TestCase):
    """A pipeline with a real AlertDispatcher over a mocked handler."""

    def setUp(self):
        self.handler = MagicMock()
        self.dispatcher = AlertDispatcher([self.handler])

    def sent_alerts(self):
        """Every Alert the handler received, in order."""
        return [call_args.args[0] for call_args in self.handler.send.call_args_list]

    def run_visit(self, dispatcher=USE_REAL_DISPATCHER, recorder=None):
        """Drive one enter -> exit cycle through a pipeline with alerts on.

        dispatcher defaults to this test case's own dispatcher; pass None to
        build a pipeline with no dispatcher at all.
        """
        pipeline, *_rest = make_pipeline(
            events_per_call=[
                [make_event(EVENT_PERSON_ENTERED_ZONE, event_id=7, timestamp=100.0)],
                [make_event(EVENT_PERSON_EXITED_ZONE, event_id=8, timestamp=112.5)],
            ],
            recorders=[recorder or make_recorder()],
            alert_dispatcher=(
                self.dispatcher if dispatcher is USE_REAL_DISPATCHER else dispatcher
            ),
        )
        pipeline.process_frame("frame1")
        pipeline.process_frame("frame2")
        return pipeline


class TestAlertsAreDispatched(AlertTestCase):
    def test_enter_and_exit_each_dispatch_one_alert(self):
        self.run_visit()

        self.assertEqual(
            [alert.event_type for alert in self.sent_alerts()],
            [EVENT_PERSON_ENTERED_ZONE, EVENT_PERSON_EXITED_ZONE],
        )

    def test_enter_alert_carries_the_event_fields(self):
        self.run_visit()

        alert = self.sent_alerts()[0]

        self.assertEqual(alert.event_id, "7")
        self.assertEqual(alert.event_type, EVENT_PERSON_ENTERED_ZONE)
        self.assertEqual(alert.timestamp, 100.0)
        self.assertEqual(alert.zone, "front_door")
        self.assertEqual(alert.track_id, 1)

    def test_exit_alert_carries_the_exit_event_fields(self):
        self.run_visit()

        alert = self.sent_alerts()[1]

        self.assertEqual(alert.event_id, "8")
        self.assertEqual(alert.timestamp, 112.5)

    def test_messages_describe_entering_and_leaving(self):
        self.run_visit()

        enter_alert, exit_alert = self.sent_alerts()

        self.assertIn("entered", enter_alert.message)
        self.assertIn("front_door", enter_alert.message)
        self.assertIn("left", exit_alert.message)

    def test_alerts_reach_a_real_console_handler(self):
        stream = io.StringIO()
        self.run_visit(dispatcher=AlertDispatcher([ConsoleAlertHandler(stream)]))

        self.assertEqual(len(stream.getvalue().strip().splitlines()), 2)

    def test_an_exit_without_a_recording_still_alerts(self):
        pipeline, *_rest = make_pipeline(
            events_per_call=[[make_event(EVENT_PERSON_EXITED_ZONE)]],
            alert_dispatcher=self.dispatcher,
        )

        pipeline.process_frame("frame1")

        self.assertEqual(len(self.sent_alerts()), 1)

    def test_close_does_not_dispatch_alerts(self):
        pipeline, *_rest = make_pipeline(
            events_per_call=[[make_event(EVENT_PERSON_ENTERED_ZONE)]],
            recorders=[make_recorder()],
            alert_dispatcher=self.dispatcher,
        )

        pipeline.process_frame("frame1")
        self.handler.send.reset_mock()
        pipeline.close()

        self.handler.send.assert_not_called()


class TestPipelineWithoutADispatcher(AlertTestCase):
    def test_recording_is_unchanged_when_no_dispatcher_is_given(self):
        recorder = make_recorder()

        self.run_visit(dispatcher=None, recorder=recorder)

        recorder.start.assert_called_once()
        recorder.stop.assert_called_once()
        self.handler.send.assert_not_called()


class TestStageFailuresDoNotStopTheLoop(unittest.TestCase):
    """Detector/Tracker/EventEngine exceptions are logged and swallowed per-stage."""

    def test_failing_detector_skips_the_frame_without_raising(self):
        pipeline, camera, detector, tracker, event_engine, _ = make_pipeline()
        detector.detect.side_effect = RuntimeError("model crashed")

        with self.assertLogs(PIPELINE_LOGGER, level="ERROR"):
            tracked_objects, events = pipeline.process_frame("frame1")

        tracker.update.assert_not_called()
        event_engine.update.assert_not_called()
        self.assertEqual(tracked_objects, [])
        self.assertEqual(events, [])

    def test_failing_tracker_skips_the_rest_of_the_frame_without_raising(self):
        pipeline, camera, detector, tracker, event_engine, _ = make_pipeline()
        detector.detect.return_value = ["detection"]
        tracker.update.side_effect = RuntimeError("tracker crashed")

        with self.assertLogs(PIPELINE_LOGGER, level="ERROR"):
            tracked_objects, events = pipeline.process_frame("frame1")

        event_engine.update.assert_not_called()
        self.assertEqual(tracked_objects, [])
        self.assertEqual(events, [])

    def test_failing_event_engine_does_not_raise(self):
        pipeline, camera, detector, tracker, event_engine, _ = make_pipeline()
        detector.detect.return_value = ["detection"]
        tracker.update.return_value = ["tracked"]
        event_engine.update.side_effect = RuntimeError("event engine crashed")

        with self.assertLogs(PIPELINE_LOGGER, level="ERROR"):
            tracked_objects, events = pipeline.process_frame("frame1")

        self.assertEqual(tracked_objects, ["tracked"])
        self.assertEqual(events, [])

    def test_recording_in_progress_still_gets_the_frame_when_detection_fails(self):
        recorder = MagicMock()
        pipeline, camera, detector, tracker, event_engine, recorder_factory = make_pipeline(
            events_per_call=[[make_event(EVENT_PERSON_ENTERED_ZONE)]], recorders=[recorder]
        )
        pipeline.process_frame("frame1")

        detector.detect.side_effect = RuntimeError("model crashed")
        with self.assertLogs(PIPELINE_LOGGER, level="ERROR"):
            pipeline.process_frame("frame2")

        recorder.write.assert_has_calls([call("frame1"), call("frame2")])

    def test_run_keeps_going_after_a_stage_failure(self):
        pipeline, camera, detector, tracker, event_engine, _ = make_pipeline()
        camera.read.side_effect = [(True, "frame1"), (True, "frame2"), (False, None)]
        detector.detect.side_effect = [RuntimeError("boom"), []]

        with self.assertLogs(PIPELINE_LOGGER, level="ERROR"):
            pipeline.run()  # should not raise

        self.assertEqual(detector.detect.call_count, 2)


class TestActiveRecordingWriteFailures(unittest.TestCase):
    def test_a_failing_write_does_not_raise_and_other_recordings_still_get_the_frame(self):
        broken_recorder = MagicMock()
        broken_recorder.write.side_effect = RecorderError("disk full")
        healthy_recorder = MagicMock()
        events = [
            make_event(EVENT_PERSON_ENTERED_ZONE, track_id=1, zone="front_door"),
            make_event(EVENT_PERSON_ENTERED_ZONE, track_id=2, zone="backyard"),
        ]
        pipeline, *_rest = make_pipeline(
            events_per_call=[events, []], recorders=[broken_recorder, healthy_recorder]
        )
        pipeline.process_frame("frame1")

        with self.assertLogs(PIPELINE_LOGGER, level="ERROR"):
            pipeline.process_frame("frame2")  # should not raise

        healthy_recorder.write.assert_called_with("frame2")


class TestAlertFailuresDoNotStopTheLoop(AlertTestCase):
    """Alerting is best-effort: recording and metadata must survive it failing."""

    def test_a_failing_handler_does_not_break_recording(self):
        self.handler.send.side_effect = RuntimeError("handler down")
        recorder = make_recorder()

        with self.assertLogs("src.alerts.alert", level="ERROR"):
            self.run_visit(recorder=recorder)

        recorder.start.assert_called_once()
        recorder.stop.assert_called_once()

    def test_a_failing_dispatcher_does_not_break_recording_or_metadata(self):
        broken_dispatcher = MagicMock()
        broken_dispatcher.send.side_effect = RuntimeError("dispatcher exploded")
        recorder = make_recorder()

        with self.assertLogs(PIPELINE_LOGGER, level="ERROR"):
            self.run_visit(dispatcher=broken_dispatcher, recorder=recorder)

        recorder.start.assert_called_once()
        recorder.stop.assert_called_once()

    def test_a_failing_dispatcher_still_lets_metadata_be_saved(self):
        broken_dispatcher = MagicMock()
        broken_dispatcher.send.side_effect = RuntimeError("dispatcher exploded")
        store = MagicMock()

        pipeline, *_rest = make_pipeline(
            events_per_call=[
                [make_event(EVENT_PERSON_ENTERED_ZONE, event_id=7)],
                [make_event(EVENT_PERSON_EXITED_ZONE, event_id=8)],
            ],
            recorders=[make_recorder()],
            metadata_store=store,
            alert_dispatcher=broken_dispatcher,
        )

        with self.assertLogs(PIPELINE_LOGGER, level="ERROR"):
            pipeline.process_frame("frame1")
            pipeline.process_frame("frame2")

        store.save.assert_called_once()


if __name__ == "__main__":
    unittest.main()

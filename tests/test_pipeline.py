"""Pipeline tests. Camera, Detector, Tracker, EventEngine and VideoRecorder are
all mocks: this only checks the wiring, not any real detection/tracking/OpenCV
behaviour (those already have their own test suites).
"""

import unittest
from unittest.mock import MagicMock, call

from src.events.events import EVENT_PERSON_ENTERED_ZONE, EVENT_PERSON_EXITED_ZONE, Event
from src.pipeline.pipeline import SurveillancePipeline, build_recorder_factory


def make_event(event_type, track_id=1, zone="front_door", event_id=1):
    return Event(
        event_id=event_id,
        timestamp=100.0,
        event_type=event_type,
        track_id=track_id,
        label="person",
        zone=zone,
        confidence=0.9,
    )


def make_pipeline(events_per_call=None, recorders=None):
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


if __name__ == "__main__":
    unittest.main()

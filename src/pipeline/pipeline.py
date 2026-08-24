"""Pipeline: wires Camera, Detector, Tracker, EventEngine and VideoRecorder together.

EventEngine only ever sees TrackedObject data and emits Event objects; it has
no idea recording exists. This module is the one place that turns those
events into recordings, so OpenCV/VideoRecorder logic never leaks into
src.events.events.

    pipeline = SurveillancePipeline(
        camera=camera,
        detector=detector,
        tracker=tracker,
        event_engine=event_engine,
        recorder_factory=lambda: VideoRecorder(data_dir / "events", fps=20.0, frame_size=(640, 480)),
    )
    pipeline.run()
"""

import logging
import time
from typing import Any, Callable, Optional

from src.camera.camera import Camera
from src.detection.detector import Detector
from src.events.events import EVENT_PERSON_ENTERED_ZONE, EVENT_PERSON_EXITED_ZONE, Event, EventEngine
from src.storage.recorder import VideoRecorder
from src.tracking.tracker import Tracker, TrackedObject

logger = logging.getLogger(__name__)

# A frame is an OpenCV BGR image (a numpy array), same as elsewhere in this project.
Frame = Any

# Builds a fresh VideoRecorder for one event. A new recorder per event keeps
# "one recording per event/track/zone" trivially true: nothing is reused.
RecorderFactory = Callable[[], VideoRecorder]

# (track_id, zone_name): identifies one occupancy, and therefore one recording.
RecordingKey = tuple[int, str]


class SurveillancePipeline:
    """Turns camera frames into tracked-person recordings, one clip per zone visit.

    process_frame() does the real work and is what tests call directly. run()
    is a thin loop around it for driving the pipeline from a live camera.
    """

    def __init__(
        self,
        camera: Camera,
        detector: Detector,
        tracker: Tracker,
        event_engine: EventEngine,
        recorder_factory: RecorderFactory,
    ) -> None:
        self.camera = camera
        self.detector = detector
        self.tracker = tracker
        self.event_engine = event_engine
        self.recorder_factory = recorder_factory
        self._recordings: dict[RecordingKey, VideoRecorder] = {}

    def process_frame(
        self, frame: Frame, timestamp: Optional[float] = None
    ) -> tuple[list[TrackedObject], list[Event]]:
        """Run one frame through detection, tracking and events, and update recordings.

        Order per frame: detect -> track -> event_engine.update -> react to any
        enter/exit events -> write the frame to every recording still active
        (including one just started this frame).
        """
        if timestamp is None:
            timestamp = time.time()

        detections = self.detector.detect(frame)
        tracked_objects = self.tracker.update(detections)
        events = self.event_engine.update(tracked_objects, timestamp)

        for event in events:
            self._handle_event(event, frame)

        for recorder in self._recordings.values():
            recorder.write(frame)

        return tracked_objects, events

    def run(self, max_frames: Optional[int] = None) -> None:
        """Read frames from the camera and process them until it runs dry.

        Stops after max_frames if given, otherwise runs until the camera stops
        returning frames. Always releases any recordings still open on exit.
        """
        frames_read = 0
        try:
            while max_frames is None or frames_read < max_frames:
                success, frame = self.camera.read()
                if not success:
                    break
                self.process_frame(frame)
                frames_read += 1
        finally:
            self.close()

    def close(self) -> None:
        """Stop every recording still in progress. Safe to call more than once."""
        for key in list(self._recordings):
            self._stop_recording(key)

    def _handle_event(self, event: Event, frame: Frame) -> None:
        key = (event.track_id, event.zone)
        if event.event_type == EVENT_PERSON_ENTERED_ZONE:
            self._start_recording(key, frame)
        elif event.event_type == EVENT_PERSON_EXITED_ZONE:
            self._stop_recording(key)

    def _start_recording(self, key: RecordingKey, frame: Frame) -> None:
        if key in self._recordings:
            # EventEngine reports at most one enter per (track, zone) between a
            # matching pair of enter/exit, so this should not happen. Guard
            # against it anyway rather than leaking the old recorder's handle.
            logger.warning("Recording already active for %r; ignoring duplicate enter", key)
            return

        recorder = self.recorder_factory()
        recorder.start()
        recorder.save_snapshot(frame)
        self._recordings[key] = recorder
        logger.info("Recording started for track=%s zone=%s", key[0], key[1])

    def _stop_recording(self, key: RecordingKey) -> None:
        recorder = self._recordings.pop(key, None)
        if recorder is not None:
            recorder.stop()
            logger.info("Recording stopped for track=%s zone=%s", key[0], key[1])


def build_recorder_factory(
    output_dir: Any, fps: float, frame_size: tuple[int, int]
) -> RecorderFactory:
    """Convenience factory: a new VideoRecorder into output_dir on every call."""

    def factory() -> VideoRecorder:
        return VideoRecorder(output_dir, fps=fps, frame_size=frame_size)

    return factory

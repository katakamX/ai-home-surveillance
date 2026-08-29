"""Pipeline: wires Camera, Detector, Tracker, EventEngine and VideoRecorder together.

EventEngine only ever sees TrackedObject data and emits Event objects; it has
no idea recording or metadata exists. This module is the one place that turns
those events into recordings and metadata files, so OpenCV/VideoRecorder and
storage logic never leak into src.events.events.

    pipeline = SurveillancePipeline(
        camera=camera,
        detector=detector,
        tracker=tracker,
        event_engine=event_engine,
        recorder_factory=lambda: VideoRecorder(data_dir / "events", fps=20.0, frame_size=(640, 480)),
        metadata_store=MetadataStore(StorageManager(data_dir / "events")),
        alert_dispatcher=AlertDispatcher([ConsoleAlertHandler()]),
    )
    pipeline.run()

Both metadata_store and alert_dispatcher are optional: without them the
pipeline records exactly as it always did, it just writes no JSON and
notifies nobody.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from src.alerts.alert import Alert, AlertDispatcher
from src.camera.camera import Camera
from src.detection.detector import Detector
from src.events.events import EVENT_PERSON_ENTERED_ZONE, EVENT_PERSON_EXITED_ZONE, Event, EventEngine
from src.storage.metadata import EventMetadata, MetadataError, MetadataStore
from src.storage.recorder import RecorderError, VideoRecorder
from src.tracking.tracker import Tracker, TrackedObject

logger = logging.getLogger(__name__)

# A frame is an OpenCV BGR image (a numpy array), same as elsewhere in this project.
Frame = Any

# Builds a fresh VideoRecorder for one event. A new recorder per event keeps
# "one recording per event/track/zone" trivially true: nothing is reused.
RecorderFactory = Callable[[], VideoRecorder]

# (track_id, zone_name): identifies one occupancy, and therefore one recording.
RecordingKey = tuple[int, str]


def _alert_for(event: Event) -> Alert:
    """Describe one enter/exit event as an Alert.

    EventEngine knows nothing about alerts, so the translation lives here.
    """
    verb = "entered" if event.event_type == EVENT_PERSON_ENTERED_ZONE else "left"
    return Alert(
        event_id=str(event.event_id),
        event_type=event.event_type,
        timestamp=event.timestamp,
        zone=event.zone,
        track_id=event.track_id,
        message=f"{event.label} (track {event.track_id}) {verb} zone {event.zone}",
    )


@dataclass
class _ActiveRecording:
    """One in-progress zone visit: its recorder plus what the metadata will need.

    Held from the enter event until the matching exit event, when it becomes
    one completed EventMetadata record.
    """

    recorder: VideoRecorder
    enter_event: Event
    video_path: Optional[Path] = None
    snapshot_path: Optional[Path] = None


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
        metadata_store: Optional[MetadataStore] = None,
        alert_dispatcher: Optional[AlertDispatcher] = None,
    ) -> None:
        self.camera = camera
        self.detector = detector
        self.tracker = tracker
        self.event_engine = event_engine
        self.recorder_factory = recorder_factory
        self.metadata_store = metadata_store
        self.alert_dispatcher = alert_dispatcher
        self._recordings: dict[RecordingKey, _ActiveRecording] = {}

    def process_frame(
        self, frame: Frame, timestamp: Optional[float] = None
    ) -> tuple[list[TrackedObject], list[Event]]:
        """Run one frame through detection, tracking and events, and update recordings.

        Order per frame: detect -> track -> event_engine.update -> react to any
        enter/exit events -> write the frame to every recording still active
        (including one just started this frame).

        Each stage is isolated: if detection, tracking or the event engine
        raises, the failure is logged and this frame is skipped for that stage
        onward, but recordings already in progress still get their frame and
        the loop keeps running.
        """
        if timestamp is None:
            timestamp = time.time()

        tracked_objects: list[TrackedObject] = []
        events: list[Event] = []

        detections = self._run_stage("Detector", self.detector.detect, frame)
        if detections is not None:
            stage_result = self._run_stage("Tracker", self.tracker.update, detections)
            if stage_result is not None:
                tracked_objects = stage_result
                stage_result = self._run_stage(
                    "EventEngine", self.event_engine.update, tracked_objects, timestamp
                )
                if stage_result is not None:
                    events = stage_result
                    for event in events:
                        self._handle_event(event, frame)

        self._write_active_recordings(frame)

        return tracked_objects, events

    def _run_stage(self, name: str, func: Callable, *args: Any) -> Any:
        """Call one pipeline stage, returning None (and logging) if it raises.

        This is the only place that decides a bad detector/tracker/event
        engine cannot bring surveillance down; the modules themselves stay
        unaware of it.
        """
        try:
            return func(*args)
        except Exception:
            logger.exception("%s failed; skipping this frame", name)
            return None

    def _write_active_recordings(self, frame: Frame) -> None:
        """Write frame to every recording still active, isolating each one."""
        for key, recording in self._recordings.items():
            try:
                recording.recorder.write(frame)
            except RecorderError as error:
                logger.exception(
                    "Could not write frame for event_id=%s track=%s zone=%s: %s",
                    recording.enter_event.event_id,
                    key[0],
                    key[1],
                    error,
                )

    def run(self, max_frames: Optional[int] = None) -> None:
        """Read frames from the camera and process them until it runs dry.

        Stops after max_frames if given, otherwise runs until the camera stops
        returning frames. Always releases any recordings still open on exit.
        """
        logger.info("Pipeline started")
        frames_read = 0
        try:
            while max_frames is None or frames_read < max_frames:
                success, frame = self.camera.read()
                if not success:
                    logger.warning("Camera stopped supplying frames; stopping pipeline")
                    break
                self.process_frame(frame)
                frames_read += 1
        finally:
            self.close()
            logger.info("Pipeline stopped")

    def close(self) -> None:
        """Stop every recording still in progress. Safe to call more than once.

        These visits never saw an exit event, so they are not complete and no
        metadata is written for them.
        """
        for key in list(self._recordings):
            self._stop_recording(key)

    def _handle_event(self, event: Event, frame: Frame) -> None:
        logger.info(
            "Event created: event_id=%s type=%s track=%s zone=%s",
            event.event_id,
            event.event_type,
            event.track_id,
            event.zone,
        )
        key = (event.track_id, event.zone)
        if event.event_type == EVENT_PERSON_ENTERED_ZONE:
            self._start_recording(key, event, frame)
        elif event.event_type == EVENT_PERSON_EXITED_ZONE:
            recording = self._stop_recording(key)
            if recording is not None:
                self._save_metadata(recording, event)

        # Alerting comes last, so recording and metadata are already safe by
        # the time anyone is notified.
        self._send_alert(event)

    def _send_alert(self, event: Event) -> None:
        """Notify the alert dispatcher about one enter or exit event."""
        if self.alert_dispatcher is None:
            return

        try:
            self.alert_dispatcher.send(_alert_for(event))
        except Exception:
            # AlertDispatcher already isolates its handlers; this is the last
            # line of defence so a notification can never stop surveillance.
            logger.exception(
                "Could not send alert for event_id=%s track=%s zone=%s",
                event.event_id,
                event.track_id,
                event.zone,
            )

    def _start_recording(self, key: RecordingKey, event: Event, frame: Frame) -> None:
        if key in self._recordings:
            # EventEngine reports at most one enter per (track, zone) between a
            # matching pair of enter/exit, so this should not happen. Guard
            # against it anyway rather than leaking the old recorder's handle.
            logger.warning("Recording already active for %r; ignoring duplicate enter", key)
            return

        recorder = self.recorder_factory()
        try:
            video_path = recorder.start()
        except RecorderError as error:
            # A camera visit we cannot record is not worth crashing the loop for.
            logger.exception(
                "Could not start recording for event_id=%s track=%s zone=%s: %s",
                event.event_id,
                key[0],
                key[1],
                error,
            )
            return

        recording = _ActiveRecording(recorder=recorder, enter_event=event, video_path=video_path)

        try:
            recording.snapshot_path = recorder.save_snapshot(frame)
        except RecorderError as error:
            # A missing snapshot is not fatal: keep recording video without it.
            logger.exception(
                "Could not save snapshot for event_id=%s track=%s zone=%s: %s",
                event.event_id,
                key[0],
                key[1],
                error,
            )

        self._recordings[key] = recording
        logger.info(
            "Recording started: event_id=%s track=%s zone=%s", event.event_id, key[0], key[1]
        )

    def _stop_recording(self, key: RecordingKey) -> Optional[_ActiveRecording]:
        """Stop the recording for key and return it, or None if there was none."""
        recording = self._recordings.pop(key, None)
        if recording is None:
            return None

        event_id = recording.enter_event.event_id
        try:
            recording.recorder.stop()
        except RecorderError as error:
            # The clip may be truncated, but the visit still happened and is
            # still worth describing in metadata.
            logger.exception(
                "Could not stop recording for event_id=%s track=%s zone=%s: %s",
                event_id,
                key[0],
                key[1],
                error,
            )

        logger.info("Recording stopped: event_id=%s track=%s zone=%s", event_id, key[0], key[1])
        return recording

    def _save_metadata(self, recording: _ActiveRecording, exit_event: Event) -> None:
        """Write one completed zone visit to the metadata store.

        The record is identified by the enter event: it says "this person
        entered this zone at this time and stayed for this long".
        """
        if self.metadata_store is None:
            return

        enter_event = recording.enter_event
        duration = exit_event.timestamp - enter_event.timestamp
        metadata = EventMetadata(
            event_id=str(enter_event.event_id),
            event_type=enter_event.event_type,
            timestamp=enter_event.timestamp,
            track_id=enter_event.track_id,
            label=enter_event.label,
            zone=enter_event.zone,
            confidence=enter_event.confidence,
            duration=duration,
            video_path=str(recording.video_path) if recording.video_path else None,
            snapshot_path=str(recording.snapshot_path) if recording.snapshot_path else None,
        )

        try:
            self.metadata_store.save(metadata)
        except MetadataError as error:
            # Losing a JSON file must not take down a running surveillance loop.
            logger.exception(
                "Could not save metadata for event_id=%s track=%s zone=%s: %s",
                metadata.event_id,
                metadata.track_id,
                metadata.zone,
                error,
            )
        else:
            logger.info(
                "Event completed: event_id=%s track=%s zone=%s duration=%.2fs",
                metadata.event_id,
                metadata.track_id,
                metadata.zone,
                duration,
            )


def build_recorder_factory(
    output_dir: Any, fps: float, frame_size: tuple[int, int]
) -> RecorderFactory:
    """Convenience factory: a new VideoRecorder into output_dir on every call."""

    def factory() -> VideoRecorder:
        return VideoRecorder(output_dir, fps=fps, frame_size=frame_size)

    return factory

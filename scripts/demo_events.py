"""Manual demo: detect, track and raise zone events on the live camera feed.
Press q in the window to quit.

Run it from the project root:

    python scripts/demo_events.py

The first run downloads the YOLO weights into models/.
"""

import logging
import sys
import time
from pathlib import Path

import cv2

# Make the src package importable when this file is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.camera.camera import Camera, CameraError
from src.config.logging_setup import setup_logging
from src.config.settings import load_settings
from src.detection.detector import Detector, DetectorError
from src.events.events import EVENT_PERSON_ENTERED_ZONE, EventEngine, Zone
from src.tracking.tracker import Tracker

GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)

# One test zone, roughly the center of a 640x480 frame. Change this box to
# match whatever resolution your camera actually delivers.
TEST_ZONE = Zone(name="test_zone", box=(200, 100, 440, 380))

logger = logging.getLogger(__name__)


def draw_tracks(frame, tracked_objects) -> None:
    """Draw a box, track ID and confidence score for each tracked person."""
    for tracked in tracked_objects:
        x1, y1, x2, y2 = tracked.box
        cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 2)

        label = f"ID {tracked.track_id} {tracked.label} {tracked.confidence:.2f}"
        cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)


def draw_zone(frame, zone: Zone, occupied: bool) -> None:
    """Draw the test zone, in yellow when empty and green when occupied."""
    x1, y1, x2, y2 = zone.box
    color = GREEN if occupied else YELLOW
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, zone.name, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def print_events(events) -> None:
    for event in events:
        verb = "ENTER" if event.event_type == EVENT_PERSON_ENTERED_ZONE else "EXIT"
        print(f"[{event.timestamp:.2f}] {verb} zone={event.zone} track_id={event.track_id}")


def main() -> None:
    settings = load_settings()
    setup_logging(settings.log_level)

    try:
        detector = Detector(
            model_path=settings.model_path,
            confidence=settings.confidence_threshold,
            person_only=settings.person_only,
        )
    except (DetectorError, ValueError) as error:
        logger.error("%s", error)
        return

    try:
        camera = Camera(settings.camera_source)
    except CameraError as error:
        logger.error("%s", error)
        return

    tracker = Tracker()
    event_engine = EventEngine([TEST_ZONE])

    logger.info("Press 'q' in the video window to quit.")

    try:
        with camera:
            # Seconds since the demo started, from a monotonic clock so the
            # numbers are real elapsed time and never jump if the system
            # clock is adjusted mid-run.
            started_at = time.monotonic()
            while True:
                success, frame = camera.read()
                if not success:
                    logger.warning("No frame received. Stopping.")
                    break

                timestamp = time.monotonic() - started_at

                try:
                    detections = detector.detect(frame)
                except DetectorError as error:
                    logger.error("%s", error)
                    break

                tracked_objects = tracker.update(detections)
                events = event_engine.update(tracked_objects, timestamp)
                print_events(events)

                zone_occupied = any(
                    TEST_ZONE.contains(
                        ((t.box[0] + t.box[2]) // 2, (t.box[1] + t.box[3]) // 2)
                    )
                    for t in tracked_objects
                )
                draw_zone(frame, TEST_ZONE, zone_occupied)
                draw_tracks(frame, tracked_objects)

                cv2.putText(
                    frame,
                    f"People: {len(tracked_objects)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    GREEN,
                    2,
                )
                cv2.imshow("Events", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        logger.info("Interrupted.")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

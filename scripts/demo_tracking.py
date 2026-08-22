"""Manual demo: track people in the live camera feed. Press q in the window to quit.

Run it from the project root:

    python scripts/demo_tracking.py

The first run downloads the YOLO weights into models/.
"""

import logging
import sys
from pathlib import Path

import cv2

# Make the src package importable when this file is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.camera.camera import Camera, CameraError
from src.config.logging_setup import setup_logging
from src.config.settings import load_settings
from src.detection.detector import Detector, DetectorError
from src.tracking.tracker import Tracker

GREEN = (0, 255, 0)


logger = logging.getLogger(__name__)


def draw_tracks(frame, tracked_objects) -> None:
    """Draw a box, track ID and confidence score for each tracked person."""
    for tracked in tracked_objects:
        x1, y1, x2, y2 = tracked.box
        cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 2)

        label = f"ID {tracked.track_id} {tracked.label} {tracked.confidence:.2f}"
        cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)


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

    logger.info("Press 'q' in the video window to quit.")

    try:
        with camera:
            while True:
                success, frame = camera.read()
                if not success:
                    logger.warning("No frame received. Stopping.")
                    break

                try:
                    detections = detector.detect(frame)
                except DetectorError as error:
                    logger.error("%s", error)
                    break

                tracked_objects = tracker.update(detections)

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
                cv2.imshow("Tracking", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        logger.info("Interrupted.")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

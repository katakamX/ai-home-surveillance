"""Manual demo: show the live camera feed. Press q in the window to quit.

Run it from the project root:

    python scripts/demo_camera.py
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

logger = logging.getLogger(__name__)


def main() -> None:
    settings = load_settings()
    setup_logging(settings.log_level)

    try:
        camera = Camera(settings.camera_source)
    except CameraError as error:
        logger.error("%s", error)
        return

    logger.info("Press 'q' in the video window to quit.")

    try:
        with camera:
            while True:
                success, frame = camera.read()
                if not success:
                    logger.warning("No frame received. Stopping.")
                    break

                cv2.imshow("Camera", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        logger.info("Interrupted.")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

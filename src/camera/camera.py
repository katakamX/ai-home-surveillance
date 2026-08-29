"""Camera capture. This is the only module that talks to cv2.VideoCapture."""

import logging
import time
from typing import Any, Optional

try:
    import cv2
except ImportError:  # OpenCV missing: fail with a clear message instead of a stack trace.
    cv2 = None

logger = logging.getLogger(__name__)

# A frame is an OpenCV BGR image (a numpy array). Aliased so signatures stay readable
# without importing numpy just for a type hint.
Frame = Any

# Defaults for reconnect(): a few attempts with a short pause between them, so a
# camera hiccup can recover without hammering the device or blocking too long.
DEFAULT_MAX_RECONNECT_ATTEMPTS = 3
DEFAULT_RECONNECT_DELAY_SECONDS = 1.0


class CameraError(Exception):
    """Raised when a camera source cannot be opened."""


class Camera:
    """A camera you can read frames from.

    The source is either a webcam index (0, 1, ...) or a string such as an RTSP
    URL or a video file path.

    Use it as a context manager so the device is always released::

        with Camera(0) as camera:
            success, frame = camera.read()
    """

    def __init__(
        self,
        source: int | str = 0,
        max_reconnect_attempts: int = DEFAULT_MAX_RECONNECT_ATTEMPTS,
        reconnect_delay_seconds: float = DEFAULT_RECONNECT_DELAY_SECONDS,
    ) -> None:
        if cv2 is None:
            raise CameraError("OpenCV is not installed. Run: pip install -r requirements.txt")

        self.source = source
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay_seconds = reconnect_delay_seconds
        self.capture = None

        capture = self._open_capture()
        if capture is None:
            raise CameraError(f"Could not open camera source: {source!r}")

        self.capture = capture
        logger.info("Camera opened: %r", source)

    def _open_capture(self):
        """Try to open self.source once. Returns the capture, or None on failure."""
        try:
            capture = cv2.VideoCapture(self.source)
        except Exception as error:  # cv2 rejects unusable source types outright
            logger.warning("Could not open camera source %r: %s", self.source, error)
            return None

        if not capture.isOpened():
            capture.release()  # don't leak the handle we just failed to use
            return None

        return capture

    def is_opened(self) -> bool:
        """True while the camera is open and has not been released."""
        return self.capture is not None and self.capture.isOpened()

    def read(self) -> tuple[bool, Optional[Frame]]:
        """Read a single frame.

        Returns (True, frame) or, on any failure, (False, None). It never returns
        a success flag without a frame, and never raises, so callers only need one
        check and a dropped/disconnected camera cannot crash the pipeline.
        """
        if not self.is_opened():
            return False, None

        try:
            success, frame = self.capture.read()
        except Exception as error:
            logger.warning("Camera read failed for %r: %s", self.source, error)
            return False, None

        if not success or frame is None:
            logger.debug("No frame available from %r", self.source)
            return False, None

        return True, frame

    def reconnect(self) -> bool:
        """Try to reopen the camera after a read failure.

        Releases the current device, then retries opening it up to
        max_reconnect_attempts times, pausing reconnect_delay_seconds between
        attempts (no pause after the last one). Returns True once reopened,
        False if every attempt failed. Never raises.
        """
        self.release()

        for attempt in range(1, self.max_reconnect_attempts + 1):
            capture = self._open_capture()
            if capture is not None:
                self.capture = capture
                logger.info("Camera reconnected: %r (attempt %d)", self.source, attempt)
                return True

            logger.warning(
                "Camera reconnect attempt %d/%d failed for %r",
                attempt,
                self.max_reconnect_attempts,
                self.source,
            )
            if attempt < self.max_reconnect_attempts:
                time.sleep(self.reconnect_delay_seconds)

        logger.error(
            "Camera reconnect gave up after %d attempts for %r",
            self.max_reconnect_attempts,
            self.source,
        )
        return False

    def release(self) -> None:
        """Release the device. Safe to call more than once."""
        if self.capture is not None:
            self.capture.release()
            self.capture = None
            logger.info("Camera released: %r", self.source)

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()

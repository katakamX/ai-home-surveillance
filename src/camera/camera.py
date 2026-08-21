"""Camera capture. This is the only module that talks to cv2.VideoCapture."""

import logging
from typing import Any, Optional

try:
    import cv2
except ImportError:  # OpenCV missing: fail with a clear message instead of a stack trace.
    cv2 = None

logger = logging.getLogger(__name__)

# A frame is an OpenCV BGR image (a numpy array). Aliased so signatures stay readable
# without importing numpy just for a type hint.
Frame = Any


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

    def __init__(self, source: int | str = 0) -> None:
        if cv2 is None:
            raise CameraError("OpenCV is not installed. Run: pip install -r requirements.txt")

        self.source = source
        self.capture = None

        try:
            capture = cv2.VideoCapture(source)
        except Exception as error:  # cv2 rejects unusable source types outright
            raise CameraError(f"Could not open camera source: {source!r}") from error

        if not capture.isOpened():
            capture.release()  # don't leak the handle we just failed to use
            raise CameraError(f"Could not open camera source: {source!r}")

        self.capture = capture
        logger.info("Camera opened: %r", source)

    def is_opened(self) -> bool:
        """True while the camera is open and has not been released."""
        return self.capture is not None and self.capture.isOpened()

    def read(self) -> tuple[bool, Optional[Frame]]:
        """Read a single frame.

        Returns (True, frame) or, on any failure, (False, None). It never returns
        a success flag without a frame, so callers only need one check.
        """
        if not self.is_opened():
            return False, None

        success, frame = self.capture.read()
        if not success or frame is None:
            logger.debug("No frame available from %r", self.source)
            return False, None

        return True, frame

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

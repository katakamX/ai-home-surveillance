import cv2


class CameraError(Exception):
    """Raised when the camera cannot be opened or read from."""


class Camera:
    """Wraps an OpenCV VideoCapture so the rest of the app never touches cv2 directly.

    The source can be an integer webcam index (0, 1, ...) or a string such as an
    RTSP URL or a video file path.
    """

    def __init__(self, source=0):
        self.source = source
        self.capture = cv2.VideoCapture(source)

        if not self.capture.isOpened():
            raise CameraError(f"Could not open camera source: {source!r}")

    def is_opened(self) -> bool:
        """Return True while the underlying capture is usable."""
        return self.capture is not None and self.capture.isOpened()

    def read(self):
        """Read one frame.

        Returns (success, frame). On failure success is False and frame is None.
        """
        if not self.is_opened():
            return False, None

        success, frame = self.capture.read()
        if not success:
            return False, None

        return True, frame

    def release(self) -> None:
        """Release the camera. Safe to call more than once."""
        if self.capture is not None:
            self.capture.release()
            self.capture = None

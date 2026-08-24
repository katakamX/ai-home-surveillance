"""Event recording. This is the only module that talks to cv2.VideoWriter.

A VideoRecorder writes one video clip per event, plus optional JPEG snapshots,
into an output directory. Every file is named after the moment it was created,
so clips and snapshots sort chronologically::

    recorder = VideoRecorder("data/events", fps=20.0, frame_size=(640, 480))
    recorder.start()
    recorder.write(frame)
    recorder.stop()

Nothing here knows about cameras, detection or events: it is handed frames and
told when to start and stop.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

try:
    import cv2
except ImportError:  # OpenCV missing: fail with a clear message instead of a stack trace.
    cv2 = None

logger = logging.getLogger(__name__)

# A frame is an OpenCV BGR image (a numpy array), same as src.camera.camera.
Frame = Any

VIDEO_SUFFIX = ".mp4"
VIDEO_CODEC = "mp4v"  # widely available and matches the .mp4 container
SNAPSHOT_SUFFIX = ".jpg"

DEFAULT_FPS = 20.0
DEFAULT_FRAME_SIZE = (640, 480)


class RecorderError(Exception):
    """Raised when a video or snapshot cannot be written."""


def _timestamp() -> str:
    """A filename-safe timestamp, e.g. "20260824_142530_812".

    Milliseconds are included so two files created in the same second do not
    overwrite each other.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


class VideoRecorder:
    """Writes event clips and snapshots to disk.

    Use it as a context manager so the file is always closed::

        with VideoRecorder("data/events") as recorder:
            recorder.start()
            recorder.write(frame)
    """

    def __init__(
        self,
        output_dir: str | Path,
        fps: float = DEFAULT_FPS,
        frame_size: tuple[int, int] = DEFAULT_FRAME_SIZE,
    ) -> None:
        if cv2 is None:
            raise RecorderError("OpenCV is not installed. Run: pip install -r requirements.txt")

        if fps <= 0:
            raise RecorderError(f"fps must be greater than 0, got {fps!r}")

        width, height = frame_size
        if width <= 0 or height <= 0:
            raise RecorderError(f"frame_size must be positive, got {frame_size!r}")

        self.output_dir = Path(output_dir)
        self.fps = float(fps)
        self.frame_size = (int(width), int(height))

        self.writer = None
        self.video_path: Optional[Path] = None

    def is_recording(self) -> bool:
        """True between a successful start() and the next stop()."""
        return self.writer is not None

    def start(self) -> Path:
        """Open a new timestamped video file and return its path.

        Raises RecorderError if a recording is already running, if the output
        directory cannot be created, or if OpenCV refuses to open the file.
        """
        if self.is_recording():
            raise RecorderError(f"Already recording to {self.video_path}. Call stop() first.")

        self._ensure_output_dir()
        path = self.output_dir / f"event_{_timestamp()}{VIDEO_SUFFIX}"

        try:
            fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
            writer = cv2.VideoWriter(str(path), fourcc, self.fps, self.frame_size)
        except Exception as error:  # bad codec, bad path type, ...
            raise RecorderError(f"Could not start recording to {path}") from error

        if not writer.isOpened():
            writer.release()  # don't leak the handle we just failed to use
            raise RecorderError(f"Could not start recording to {path}")

        self.writer = writer
        self.video_path = path
        logger.info("Recording started: %s", path)
        return path

    def write(self, frame: Frame) -> bool:
        """Append one frame to the current recording.

        Returns True if the frame was written. A missing frame, or a call made
        while not recording, returns False instead of raising: a dropped frame
        should never stop the surveillance loop.
        """
        if not self.is_recording():
            logger.debug("write() called while not recording; frame ignored")
            return False

        if frame is None:
            logger.debug("write() called with no frame; frame ignored")
            return False

        try:
            self.writer.write(frame)
        except Exception as error:  # disk full, frame of the wrong shape, ...
            raise RecorderError(f"Could not write frame to {self.video_path}") from error

        return True

    def stop(self) -> Optional[Path]:
        """Close the current recording and return its path.

        Safe to call more than once, and safe to call when nothing was started:
        those calls simply return None.
        """
        if not self.is_recording():
            return None

        path = self.video_path
        writer, self.writer, self.video_path = self.writer, None, None

        try:
            writer.release()
        except Exception as error:
            raise RecorderError(f"Could not close recording {path}") from error

        logger.info("Recording stopped: %s", path)
        return path

    def save_snapshot(self, frame: Frame) -> Path:
        """Write one frame as a timestamped JPEG and return its path.

        Independent of start()/stop(): a snapshot can be taken at any time.
        """
        if frame is None:
            raise RecorderError("Cannot save a snapshot without a frame")

        self._ensure_output_dir()
        path = self.output_dir / f"snapshot_{_timestamp()}{SNAPSHOT_SUFFIX}"

        try:
            written = cv2.imwrite(str(path), frame)
        except Exception as error:  # unwritable path, unsupported frame, ...
            raise RecorderError(f"Could not save snapshot to {path}") from error

        if not written:  # imwrite reports failure by returning False
            raise RecorderError(f"Could not save snapshot to {path}")

        logger.info("Snapshot saved: %s", path)
        return path

    def _ensure_output_dir(self) -> None:
        """Create the output directory if it does not exist yet."""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise RecorderError(f"Could not create output directory {self.output_dir}") from error

    def __enter__(self) -> "VideoRecorder":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()

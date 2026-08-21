"""Person detection. This is the only module that talks to Ultralytics YOLO."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from ultralytics import YOLO
except ImportError:  # Ultralytics missing: fail with a clear message instead of a stack trace.
    YOLO = None

logger = logging.getLogger(__name__)

# "person" is class 0 in the COCO dataset the default YOLO models are trained on.
PERSON_CLASS_ID = 0

# Weights live in models/ (gitignored). Resolved from this file so the default
# works no matter which directory you run from.
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "yolov8n.pt"

# A frame is an OpenCV BGR image (a numpy array). Aliased so signatures stay readable
# without importing numpy just for a type hint.
Frame = Any


class DetectorError(Exception):
    """Raised when the model cannot be loaded, or when inference fails."""


@dataclass
class Detection:
    """One detected object in a single frame. Contains no Ultralytics types."""

    label: str
    confidence: float
    box: tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixels


class Detector:
    """Detects objects in frames, reporting only people by default.

    The model is loaded once here and reused for every call to detect().
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        confidence: float = 0.5,
        person_only: bool = True,
    ) -> None:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {confidence!r}")

        if YOLO is None:
            raise DetectorError(
                "Ultralytics is not installed. Run: pip install -r requirements.txt"
            )

        self.model_path = str(model_path)
        self.confidence = confidence
        self.person_only = person_only

        logger.info("Loading model: %s", self.model_path)
        try:
            self.model = YOLO(self.model_path)
        except Exception as error:
            raise DetectorError(f"Could not load model: {self.model_path}") from error

        logger.info("Model loaded")

    def detect(self, frame: Frame) -> list[Detection]:
        """Run detection on one frame.

        Returns a list of Detection objects, empty when nothing was found. Any
        inference or model-output problem raises DetectorError rather than
        returning a short list that would look like "nobody is there".
        """
        classes = [PERSON_CLASS_ID] if self.person_only else None

        try:
            results = self.model.predict(
                frame,
                conf=self.confidence,
                classes=classes,
                verbose=False,
            )
        except Exception as error:
            raise DetectorError("Inference failed") from error

        return self._to_detections(results)

    def _to_detections(self, results: Any) -> list[Detection]:
        """Convert raw Ultralytics results into plain Detection objects."""
        detections: list[Detection] = []

        try:
            for result in results:
                if result.boxes is None:  # a result can carry no boxes at all
                    continue

                for box in result.boxes:
                    class_id = int(box.cls[0])
                    x1, y1, x2, y2 = (int(value) for value in box.xyxy[0])
                    detections.append(
                        Detection(
                            label=self.model.names[class_id],
                            confidence=float(box.conf[0]),
                            box=(x1, y1, x2, y2),
                        )
                    )
        except Exception as error:
            raise DetectorError("Unexpected model output") from error

        return detections

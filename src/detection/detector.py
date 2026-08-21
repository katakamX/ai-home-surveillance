from dataclasses import dataclass
from pathlib import Path

from ultralytics import YOLO

# "person" is class 0 in the COCO dataset that the default YOLO models are trained on.
PERSON_CLASS_ID = 0

# Weights live in models/ (gitignored). Resolved from this file so the path works
# no matter which directory you run from.
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "yolov8n.pt"


class DetectorError(Exception):
    """Raised when the detection model cannot be loaded."""


@dataclass
class Detection:
    """One detected object in a single frame."""

    label: str
    confidence: float
    box: tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixels


class Detector:
    """Wraps an Ultralytics YOLO model so the rest of the app never touches YOLO directly.

    By default it loads the small yolov8n model and reports only people.
    """

    def __init__(self, model_path=DEFAULT_MODEL_PATH, confidence=0.5, person_only=True):
        self.model_path = str(model_path)
        self.confidence = confidence
        self.person_only = person_only

        try:
            self.model = YOLO(self.model_path)
        except Exception as error:
            raise DetectorError(f"Could not load model: {self.model_path}") from error

    def detect(self, frame) -> list[Detection]:
        """Run detection on one frame and return a list of Detection objects."""
        classes = [PERSON_CLASS_ID] if self.person_only else None

        results = self.model.predict(
            frame,
            conf=self.confidence,
            classes=classes,
            verbose=False,
        )

        detections = []
        for result in results:
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

        return detections

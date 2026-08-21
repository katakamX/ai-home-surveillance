"""Manual check: open webcam 0, detect people, and show the boxes. Press q to quit."""

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.camera.camera import Camera, CameraError
from src.detection.detector import Detector, DetectorError

GREEN = (0, 255, 0)


def draw_detections(frame, detections) -> None:
    """Draw a box and a confidence score for each detection."""
    for detection in detections:
        x1, y1, x2, y2 = detection.box
        cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 2)

        label = f"{detection.label} {detection.confidence:.2f}"
        cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)


def main() -> None:
    print("Loading model (the first run downloads the weights)...")
    try:
        detector = Detector()
    except DetectorError as error:
        print(error)
        return

    try:
        camera = Camera(0)
    except CameraError as error:
        print(error)
        return

    print("Camera opened. Press 'q' to quit.")

    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("Failed to read frame. Stopping.")
                break

            detections = detector.detect(frame)
            draw_detections(frame, detections)

            cv2.putText(
                frame,
                f"People: {len(detections)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                GREEN,
                2,
            )
            cv2.imshow("Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Camera released.")


if __name__ == "__main__":
    main()

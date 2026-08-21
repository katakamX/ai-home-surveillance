"""Manual check: open webcam 0 and show the live feed. Press q to quit."""

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.camera.camera import Camera, CameraError


def main() -> None:
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

            cv2.imshow("Camera", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Camera released.")


if __name__ == "__main__":
    main()

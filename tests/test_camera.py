"""Camera tests. No physical camera is used: cv2 is replaced with a mock.

The whole cv2 module is patched (not just VideoCapture) so these tests also pass
on a machine where OpenCV is not installed.
"""

import unittest
from unittest.mock import MagicMock, patch

from src.camera.camera import Camera, CameraError

RTSP_URL = "rtsp://192.168.1.10:554/stream"


def make_capture(is_opened=True, read_result=(True, "frame")):
    """Build a fake cv2.VideoCapture."""
    capture = MagicMock()
    capture.isOpened.return_value = is_opened
    capture.read.return_value = read_result
    return capture


class TestCameraInitialisation(unittest.TestCase):
    @patch("src.camera.camera.cv2")
    def test_opens_with_integer_source(self, cv2):
        cv2.VideoCapture.return_value = make_capture()

        camera = Camera(0)

        cv2.VideoCapture.assert_called_once_with(0)
        self.assertTrue(camera.is_opened())
        self.assertEqual(camera.source, 0)

    @patch("src.camera.camera.cv2")
    def test_opens_with_rtsp_source(self, cv2):
        cv2.VideoCapture.return_value = make_capture()

        camera = Camera(RTSP_URL)

        cv2.VideoCapture.assert_called_once_with(RTSP_URL)
        self.assertEqual(camera.source, RTSP_URL)

    @patch("src.camera.camera.cv2")
    def test_defaults_to_source_zero(self, cv2):
        cv2.VideoCapture.return_value = make_capture()

        Camera()

        cv2.VideoCapture.assert_called_once_with(0)


class TestCameraFailure(unittest.TestCase):
    @patch("src.camera.camera.cv2")
    def test_raises_when_camera_cannot_be_opened(self, cv2):
        cv2.VideoCapture.return_value = make_capture(is_opened=False)

        with self.assertRaises(CameraError):
            Camera(99)

    @patch("src.camera.camera.cv2")
    def test_releases_capture_when_open_fails(self, cv2):
        capture = make_capture(is_opened=False)
        cv2.VideoCapture.return_value = capture

        with self.assertRaises(CameraError):
            Camera(99)

        capture.release.assert_called_once()

    @patch("src.camera.camera.cv2")
    def test_videocapture_raising_becomes_camera_error(self, cv2):
        cv2.VideoCapture.side_effect = TypeError("bad argument")

        with self.assertRaises(CameraError):
            Camera(object())

    @patch("src.camera.camera.cv2", None)
    def test_raises_when_opencv_is_missing(self):
        with self.assertRaises(CameraError):
            Camera(0)


class TestCameraRead(unittest.TestCase):
    @patch("src.camera.camera.cv2")
    def test_read_returns_frame_on_success(self, cv2):
        cv2.VideoCapture.return_value = make_capture(read_result=(True, "frame"))

        success, frame = Camera(0).read()

        self.assertTrue(success)
        self.assertEqual(frame, "frame")

    @patch("src.camera.camera.cv2")
    def test_read_returns_none_on_failure(self, cv2):
        cv2.VideoCapture.return_value = make_capture(read_result=(False, None))

        success, frame = Camera(0).read()

        self.assertFalse(success)
        self.assertIsNone(frame)

    @patch("src.camera.camera.cv2")
    def test_read_reports_failure_when_frame_is_none_despite_success(self, cv2):
        # OpenCV occasionally reports success with no frame attached.
        cv2.VideoCapture.return_value = make_capture(read_result=(True, None))

        success, frame = Camera(0).read()

        self.assertFalse(success)
        self.assertIsNone(frame)

    @patch("src.camera.camera.cv2")
    def test_read_after_release_returns_failure(self, cv2):
        cv2.VideoCapture.return_value = make_capture()

        camera = Camera(0)
        camera.release()

        self.assertEqual(camera.read(), (False, None))

    @patch("src.camera.camera.cv2")
    def test_read_returns_failure_once_device_closes(self, cv2):
        capture = make_capture()
        cv2.VideoCapture.return_value = capture

        camera = Camera(0)
        capture.isOpened.return_value = False  # device disappeared mid-stream

        self.assertEqual(camera.read(), (False, None))
        capture.read.assert_not_called()


class TestCameraRelease(unittest.TestCase):
    @patch("src.camera.camera.cv2")
    def test_release_closes_the_device(self, cv2):
        capture = make_capture()
        cv2.VideoCapture.return_value = capture

        camera = Camera(0)
        camera.release()

        capture.release.assert_called_once()
        self.assertFalse(camera.is_opened())

    @patch("src.camera.camera.cv2")
    def test_release_is_safe_to_call_twice(self, cv2):
        capture = make_capture()
        cv2.VideoCapture.return_value = capture

        camera = Camera(0)
        camera.release()
        camera.release()

        capture.release.assert_called_once()

    @patch("src.camera.camera.cv2")
    def test_context_manager_releases_on_exit(self, cv2):
        capture = make_capture()
        cv2.VideoCapture.return_value = capture

        with Camera(0) as camera:
            self.assertTrue(camera.is_opened())

        capture.release.assert_called_once()

    @patch("src.camera.camera.cv2")
    def test_context_manager_releases_when_the_body_raises(self, cv2):
        capture = make_capture()
        cv2.VideoCapture.return_value = capture

        with self.assertRaises(RuntimeError):
            with Camera(0):
                raise RuntimeError("boom")

        capture.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import MagicMock, patch

from src.camera.camera import Camera, CameraError


def make_capture(is_opened=True, read_result=(True, "frame")):
    """Build a fake cv2.VideoCapture."""
    capture = MagicMock()
    capture.isOpened.return_value = is_opened
    capture.read.return_value = read_result
    return capture


class TestCamera(unittest.TestCase):
    @patch("src.camera.camera.cv2.VideoCapture")
    def test_opens_with_integer_source(self, video_capture):
        video_capture.return_value = make_capture()

        camera = Camera(0)

        video_capture.assert_called_once_with(0)
        self.assertTrue(camera.is_opened())

    @patch("src.camera.camera.cv2.VideoCapture")
    def test_opens_with_rtsp_source(self, video_capture):
        video_capture.return_value = make_capture()
        url = "rtsp://192.168.1.10:554/stream"

        camera = Camera(url)

        video_capture.assert_called_once_with(url)
        self.assertEqual(camera.source, url)

    @patch("src.camera.camera.cv2.VideoCapture")
    def test_raises_when_camera_cannot_be_opened(self, video_capture):
        video_capture.return_value = make_capture(is_opened=False)

        with self.assertRaises(CameraError):
            Camera(99)

    @patch("src.camera.camera.cv2.VideoCapture")
    def test_read_returns_frame_on_success(self, video_capture):
        video_capture.return_value = make_capture(read_result=(True, "frame"))

        camera = Camera(0)
        success, frame = camera.read()

        self.assertTrue(success)
        self.assertEqual(frame, "frame")

    @patch("src.camera.camera.cv2.VideoCapture")
    def test_read_returns_none_on_failure(self, video_capture):
        video_capture.return_value = make_capture(read_result=(False, None))

        camera = Camera(0)
        success, frame = camera.read()

        self.assertFalse(success)
        self.assertIsNone(frame)

    @patch("src.camera.camera.cv2.VideoCapture")
    def test_read_after_release_returns_failure(self, video_capture):
        video_capture.return_value = make_capture()

        camera = Camera(0)
        camera.release()
        success, frame = camera.read()

        self.assertFalse(success)
        self.assertIsNone(frame)

    @patch("src.camera.camera.cv2.VideoCapture")
    def test_release_is_safe_to_call_twice(self, video_capture):
        capture = make_capture()
        video_capture.return_value = capture

        camera = Camera(0)
        camera.release()
        camera.release()

        capture.release.assert_called_once()
        self.assertFalse(camera.is_opened())


if __name__ == "__main__":
    unittest.main()

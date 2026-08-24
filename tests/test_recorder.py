"""Recorder tests. Nothing is encoded: cv2 is replaced with a mock.

The whole cv2 module is patched (not just VideoWriter) so these tests also pass
on a machine where OpenCV is not installed. Output directories live in a
temporary folder, so no file is ever written into the project.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.storage.recorder import RecorderError, VideoRecorder


def make_writer(is_opened=True):
    """Build a fake cv2.VideoWriter."""
    writer = MagicMock()
    writer.isOpened.return_value = is_opened
    return writer


class RecorderTestCase(unittest.TestCase):
    """Gives every test its own throwaway output directory."""

    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp_dir.cleanup)
        self.output_dir = Path(self._temp_dir.name) / "events"


class TestRecorderInitialisation(RecorderTestCase):
    @patch("src.storage.recorder.cv2")
    def test_stores_settings(self, cv2):
        recorder = VideoRecorder(self.output_dir, fps=15.0, frame_size=(1280, 720))

        self.assertEqual(recorder.output_dir, self.output_dir)
        self.assertEqual(recorder.fps, 15.0)
        self.assertEqual(recorder.frame_size, (1280, 720))
        self.assertFalse(recorder.is_recording())

    @patch("src.storage.recorder.cv2")
    def test_accepts_a_string_output_dir(self, cv2):
        recorder = VideoRecorder(str(self.output_dir))

        self.assertEqual(recorder.output_dir, self.output_dir)

    @patch("src.storage.recorder.cv2")
    def test_creates_nothing_on_disk_until_used(self, cv2):
        VideoRecorder(self.output_dir)

        self.assertFalse(self.output_dir.exists())

    @patch("src.storage.recorder.cv2")
    def test_rejects_non_positive_fps(self, cv2):
        with self.assertRaises(RecorderError):
            VideoRecorder(self.output_dir, fps=0)

    @patch("src.storage.recorder.cv2")
    def test_rejects_non_positive_frame_size(self, cv2):
        with self.assertRaises(RecorderError):
            VideoRecorder(self.output_dir, frame_size=(640, 0))

    @patch("src.storage.recorder.cv2", None)
    def test_raises_when_opencv_is_missing(self):
        with self.assertRaises(RecorderError):
            VideoRecorder(self.output_dir)


class TestRecorderStart(RecorderTestCase):
    @patch("src.storage.recorder.cv2")
    def test_start_opens_a_timestamped_video_file(self, cv2):
        cv2.VideoWriter.return_value = make_writer()

        path = VideoRecorder(self.output_dir).start()

        self.assertEqual(path.parent, self.output_dir)
        self.assertTrue(path.name.startswith("event_"))
        self.assertEqual(path.suffix, ".mp4")

    @patch("src.storage.recorder.cv2")
    def test_start_creates_the_output_directory(self, cv2):
        cv2.VideoWriter.return_value = make_writer()

        VideoRecorder(self.output_dir / "nested").start()

        self.assertTrue((self.output_dir / "nested").is_dir())

    @patch("src.storage.recorder.cv2")
    def test_start_passes_fps_and_frame_size_to_opencv(self, cv2):
        cv2.VideoWriter.return_value = make_writer()
        cv2.VideoWriter_fourcc.return_value = "fourcc"

        recorder = VideoRecorder(self.output_dir, fps=15.0, frame_size=(320, 240))
        path = recorder.start()

        cv2.VideoWriter.assert_called_once_with(str(path), "fourcc", 15.0, (320, 240))

    @patch("src.storage.recorder.cv2")
    def test_start_marks_the_recorder_as_recording(self, cv2):
        cv2.VideoWriter.return_value = make_writer()

        recorder = VideoRecorder(self.output_dir)
        recorder.start()

        self.assertTrue(recorder.is_recording())

    @patch("src.storage.recorder.cv2")
    def test_two_starts_give_different_file_names(self, cv2):
        cv2.VideoWriter.return_value = make_writer()

        recorder = VideoRecorder(self.output_dir)
        first = recorder.start()
        recorder.stop()
        second = recorder.start()

        self.assertNotEqual(first, second)

    @patch("src.storage.recorder.cv2")
    def test_start_while_recording_raises(self, cv2):
        cv2.VideoWriter.return_value = make_writer()

        recorder = VideoRecorder(self.output_dir)
        recorder.start()

        with self.assertRaises(RecorderError):
            recorder.start()

    @patch("src.storage.recorder.cv2")
    def test_start_raises_when_the_file_cannot_be_opened(self, cv2):
        cv2.VideoWriter.return_value = make_writer(is_opened=False)

        with self.assertRaises(RecorderError):
            VideoRecorder(self.output_dir).start()

    @patch("src.storage.recorder.cv2")
    def test_start_releases_the_writer_it_could_not_open(self, cv2):
        writer = make_writer(is_opened=False)
        cv2.VideoWriter.return_value = writer

        recorder = VideoRecorder(self.output_dir)
        with self.assertRaises(RecorderError):
            recorder.start()

        writer.release.assert_called_once()
        self.assertFalse(recorder.is_recording())

    @patch("src.storage.recorder.cv2")
    def test_videowriter_raising_becomes_recorder_error(self, cv2):
        cv2.VideoWriter.side_effect = OSError("no such device")

        with self.assertRaises(RecorderError):
            VideoRecorder(self.output_dir).start()

    @patch("src.storage.recorder.cv2")
    @patch("pathlib.Path.mkdir", side_effect=PermissionError("read-only"))
    def test_start_raises_when_the_directory_cannot_be_created(self, mkdir, cv2):
        cv2.VideoWriter.return_value = make_writer()

        with self.assertRaises(RecorderError):
            VideoRecorder(self.output_dir).start()

        cv2.VideoWriter.assert_not_called()


class TestRecorderWrite(RecorderTestCase):
    @patch("src.storage.recorder.cv2")
    def test_write_sends_the_frame_to_opencv(self, cv2):
        writer = make_writer()
        cv2.VideoWriter.return_value = writer

        recorder = VideoRecorder(self.output_dir)
        recorder.start()

        self.assertTrue(recorder.write("frame"))
        writer.write.assert_called_once_with("frame")

    @patch("src.storage.recorder.cv2")
    def test_write_before_start_is_ignored(self, cv2):
        recorder = VideoRecorder(self.output_dir)

        self.assertFalse(recorder.write("frame"))
        cv2.VideoWriter.assert_not_called()

    @patch("src.storage.recorder.cv2")
    def test_write_after_stop_is_ignored(self, cv2):
        writer = make_writer()
        cv2.VideoWriter.return_value = writer

        recorder = VideoRecorder(self.output_dir)
        recorder.start()
        recorder.stop()

        self.assertFalse(recorder.write("frame"))
        writer.write.assert_not_called()

    @patch("src.storage.recorder.cv2")
    def test_write_without_a_frame_is_ignored(self, cv2):
        writer = make_writer()
        cv2.VideoWriter.return_value = writer

        recorder = VideoRecorder(self.output_dir)
        recorder.start()

        self.assertFalse(recorder.write(None))
        writer.write.assert_not_called()

    @patch("src.storage.recorder.cv2")
    def test_write_failure_becomes_recorder_error(self, cv2):
        writer = make_writer()
        writer.write.side_effect = OSError("disk full")
        cv2.VideoWriter.return_value = writer

        recorder = VideoRecorder(self.output_dir)
        recorder.start()

        with self.assertRaises(RecorderError):
            recorder.write("frame")


class TestRecorderStop(RecorderTestCase):
    @patch("src.storage.recorder.cv2")
    def test_stop_releases_the_writer_and_returns_the_path(self, cv2):
        writer = make_writer()
        cv2.VideoWriter.return_value = writer

        recorder = VideoRecorder(self.output_dir)
        path = recorder.start()

        self.assertEqual(recorder.stop(), path)
        writer.release.assert_called_once()
        self.assertFalse(recorder.is_recording())

    @patch("src.storage.recorder.cv2")
    def test_stop_is_safe_to_call_twice(self, cv2):
        writer = make_writer()
        cv2.VideoWriter.return_value = writer

        recorder = VideoRecorder(self.output_dir)
        recorder.start()
        recorder.stop()

        self.assertIsNone(recorder.stop())
        writer.release.assert_called_once()

    @patch("src.storage.recorder.cv2")
    def test_stop_without_start_returns_none(self, cv2):
        self.assertIsNone(VideoRecorder(self.output_dir).stop())

    @patch("src.storage.recorder.cv2")
    def test_failed_release_becomes_recorder_error_and_still_stops(self, cv2):
        writer = make_writer()
        writer.release.side_effect = OSError("device busy")
        cv2.VideoWriter.return_value = writer

        recorder = VideoRecorder(self.output_dir)
        recorder.start()

        with self.assertRaises(RecorderError):
            recorder.stop()

        self.assertFalse(recorder.is_recording())

    @patch("src.storage.recorder.cv2")
    def test_context_manager_stops_on_exit(self, cv2):
        writer = make_writer()
        cv2.VideoWriter.return_value = writer

        with VideoRecorder(self.output_dir) as recorder:
            recorder.start()

        writer.release.assert_called_once()

    @patch("src.storage.recorder.cv2")
    def test_context_manager_stops_when_the_body_raises(self, cv2):
        writer = make_writer()
        cv2.VideoWriter.return_value = writer

        with self.assertRaises(RuntimeError):
            with VideoRecorder(self.output_dir) as recorder:
                recorder.start()
                raise RuntimeError("boom")

        writer.release.assert_called_once()


class TestRecorderSnapshot(RecorderTestCase):
    @patch("src.storage.recorder.cv2")
    def test_snapshot_writes_a_timestamped_jpeg(self, cv2):
        cv2.imwrite.return_value = True

        path = VideoRecorder(self.output_dir).save_snapshot("frame")

        self.assertEqual(path.parent, self.output_dir)
        self.assertTrue(path.name.startswith("snapshot_"))
        self.assertEqual(path.suffix, ".jpg")
        cv2.imwrite.assert_called_once_with(str(path), "frame")

    @patch("src.storage.recorder.cv2")
    def test_snapshot_creates_the_output_directory(self, cv2):
        cv2.imwrite.return_value = True

        VideoRecorder(self.output_dir).save_snapshot("frame")

        self.assertTrue(self.output_dir.is_dir())

    @patch("src.storage.recorder.cv2")
    def test_snapshot_does_not_need_a_recording(self, cv2):
        cv2.imwrite.return_value = True

        recorder = VideoRecorder(self.output_dir)
        recorder.save_snapshot("frame")

        self.assertFalse(recorder.is_recording())
        cv2.VideoWriter.assert_not_called()

    @patch("src.storage.recorder.cv2")
    def test_snapshot_without_a_frame_raises(self, cv2):
        with self.assertRaises(RecorderError):
            VideoRecorder(self.output_dir).save_snapshot(None)

        cv2.imwrite.assert_not_called()

    @patch("src.storage.recorder.cv2")
    def test_snapshot_raises_when_opencv_reports_failure(self, cv2):
        cv2.imwrite.return_value = False  # imwrite reports failure by returning False

        with self.assertRaises(RecorderError):
            VideoRecorder(self.output_dir).save_snapshot("frame")

    @patch("src.storage.recorder.cv2")
    def test_imwrite_raising_becomes_recorder_error(self, cv2):
        cv2.imwrite.side_effect = OSError("disk full")

        with self.assertRaises(RecorderError):
            VideoRecorder(self.output_dir).save_snapshot("frame")

    @patch("src.storage.recorder.cv2")
    @patch("pathlib.Path.mkdir", side_effect=PermissionError("read-only"))
    def test_snapshot_raises_when_the_directory_cannot_be_created(self, mkdir, cv2):
        with self.assertRaises(RecorderError):
            VideoRecorder(self.output_dir).save_snapshot("frame")

        cv2.imwrite.assert_not_called()


if __name__ == "__main__":
    unittest.main()

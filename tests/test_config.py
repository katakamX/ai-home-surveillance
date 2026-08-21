"""Settings and logging-setup tests.

load_settings() is always given an explicit mapping, so these tests never touch
the real environment or a local .env file.
"""

import logging
import unittest

from src.config.logging_setup import resolve_level
from src.config.settings import PROJECT_ROOT, load_settings


class TestSettingsDefaults(unittest.TestCase):
    def setUp(self):
        self.settings = load_settings({})

    def test_camera_source_defaults_to_the_first_webcam(self):
        self.assertEqual(self.settings.camera_source, 0)

    def test_detection_defaults(self):
        self.assertEqual(self.settings.confidence_threshold, 0.5)
        self.assertTrue(self.settings.person_only)

    def test_directories_default_below_the_project_root(self):
        self.assertEqual(self.settings.model_dir, PROJECT_ROOT / "models")
        self.assertEqual(self.settings.data_dir, PROJECT_ROOT / "data")
        self.assertEqual(self.settings.model_path, PROJECT_ROOT / "models" / "yolov8n.pt")

    def test_log_level_defaults_to_info(self):
        self.assertEqual(self.settings.log_level, "INFO")


class TestCameraSource(unittest.TestCase):
    def test_digits_become_a_webcam_index(self):
        self.assertEqual(load_settings({"CAMERA_SOURCE": "1"}).camera_source, 1)

    def test_rtsp_url_stays_a_string(self):
        url = "rtsp://192.168.1.10:554/stream"
        self.assertEqual(load_settings({"CAMERA_SOURCE": url}).camera_source, url)

    def test_file_path_stays_a_string(self):
        self.assertEqual(load_settings({"CAMERA_SOURCE": "clip.mp4"}).camera_source, "clip.mp4")


class TestPaths(unittest.TestCase):
    def test_model_dir_moves_the_default_model_path(self):
        settings = load_settings({"MODEL_DIR": "weights"})

        self.assertEqual(settings.model_dir, PROJECT_ROOT / "weights")
        self.assertEqual(settings.model_path, PROJECT_ROOT / "weights" / "yolov8n.pt")

    def test_explicit_model_path_wins(self):
        settings = load_settings({"MODEL_DIR": "weights", "MODEL_PATH": "models/big.pt"})

        self.assertEqual(settings.model_path, PROJECT_ROOT / "models" / "big.pt")

    def test_absolute_paths_are_left_alone(self):
        absolute = "C:/weights/yolo.pt" if PROJECT_ROOT.drive else "/weights/yolo.pt"

        settings = load_settings({"MODEL_PATH": absolute})

        self.assertTrue(settings.model_path.is_absolute())

    def test_data_dir_is_configurable(self):
        self.assertEqual(load_settings({"DATA_DIR": "clips"}).data_dir, PROJECT_ROOT / "clips")


class TestValueParsing(unittest.TestCase):
    def test_confidence_is_read_as_a_float(self):
        self.assertEqual(load_settings({"CONFIDENCE_THRESHOLD": "0.8"}).confidence_threshold, 0.8)

    def test_unparsable_confidence_falls_back_to_the_default(self):
        self.assertEqual(load_settings({"CONFIDENCE_THRESHOLD": "high"}).confidence_threshold, 0.5)

    def test_truthy_values_enable_person_only(self):
        for value in ("1", "true", "TRUE", "yes", "on"):
            with self.subTest(value=value):
                self.assertTrue(load_settings({"PERSON_ONLY": value}).person_only)

    def test_other_values_disable_person_only(self):
        for value in ("0", "false", "no", "off"):
            with self.subTest(value=value):
                self.assertFalse(load_settings({"PERSON_ONLY": value}).person_only)

    def test_blank_values_fall_back_to_defaults(self):
        settings = load_settings({"CAMERA_SOURCE": "  ", "MODEL_PATH": "  ", "LOG_LEVEL": ""})

        self.assertEqual(settings.camera_source, 0)
        self.assertEqual(settings.model_path, PROJECT_ROOT / "models" / "yolov8n.pt")
        self.assertEqual(settings.log_level, "INFO")

    def test_surrounding_whitespace_is_ignored(self):
        self.assertEqual(load_settings({"CAMERA_SOURCE": " 2 "}).camera_source, 2)

    def test_log_level_is_upper_cased(self):
        self.assertEqual(load_settings({"LOG_LEVEL": "debug"}).log_level, "DEBUG")


class TestLoggingSetup(unittest.TestCase):
    def test_known_level_names_resolve(self):
        self.assertEqual(resolve_level("DEBUG"), logging.DEBUG)
        self.assertEqual(resolve_level("warning"), logging.WARNING)

    def test_unknown_level_falls_back_to_info(self):
        self.assertEqual(resolve_level("chatty"), logging.INFO)

    def test_non_level_attribute_falls_back_to_info(self):
        # "shutdown" is a real attribute of the logging module, but not a level.
        self.assertEqual(resolve_level("shutdown"), logging.INFO)


if __name__ == "__main__":
    unittest.main()

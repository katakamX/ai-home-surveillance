import unittest
from unittest.mock import patch

from src.config.settings import load_settings
from src.main import main


class TestMain(unittest.TestCase):
    def test_main_is_callable(self):
        self.assertTrue(callable(main))

    @patch("src.main.setup_logging")
    @patch("src.main.load_settings")
    def test_main_logs_startup_and_shutdown(self, load, _setup_logging):
        load.return_value = load_settings({})

        with self.assertLogs("src.main", level="INFO") as captured:
            main()

        output = "\n".join(captured.output)
        self.assertIn("AI Home Surveillance System started.", output)
        self.assertIn("AI Home Surveillance System stopped.", output)

    @patch("src.main.setup_logging")
    @patch("src.main.load_settings")
    def test_main_configures_logging_from_settings(self, load, setup_logging):
        load.return_value = load_settings({"LOG_LEVEL": "debug"})

        with self.assertLogs("src.main", level="INFO"):
            main()

        setup_logging.assert_called_once_with("DEBUG")


if __name__ == "__main__":
    unittest.main()

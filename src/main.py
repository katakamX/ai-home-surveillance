"""Application entry point.

No pipeline is wired up yet. This starts logging, reports the effective
configuration, and exits. Use the scripts in scripts/ to try the camera and
detector on a laptop.
"""

import logging

from src.config.logging_setup import setup_logging
from src.config.settings import load_settings

logger = logging.getLogger(__name__)


def main() -> None:
    settings = load_settings()
    setup_logging(settings.log_level)

    logger.info("AI Home Surveillance System started.")
    logger.info("Camera source:        %s", settings.camera_source)
    logger.info("Model path:           %s", settings.model_path)
    logger.info("Confidence threshold: %s", settings.confidence_threshold)
    logger.info("Person only:          %s", settings.person_only)
    logger.info("No capture pipeline yet. Try: python scripts/demo_detection.py")
    logger.info("AI Home Surveillance System stopped.")


if __name__ == "__main__":
    main()

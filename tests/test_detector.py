import unittest
from unittest.mock import MagicMock, patch

from src.detection.detector import Detection, Detector, DetectorError


class FakeBox:
    """Stands in for one Ultralytics box (values are wrapped in lists like the real thing)."""

    def __init__(self, class_id, confidence, xyxy):
        self.cls = [class_id]
        self.conf = [confidence]
        self.xyxy = [xyxy]


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


def make_model(boxes=()):
    """Build a fake YOLO model that returns the given boxes."""
    model = MagicMock()
    model.names = {0: "person", 15: "cat"}
    model.predict.return_value = [FakeResult(list(boxes))]
    return model


class TestDetector(unittest.TestCase):
    @patch("src.detection.detector.YOLO")
    def test_raises_when_model_cannot_be_loaded(self, yolo):
        yolo.side_effect = FileNotFoundError("missing weights")

        with self.assertRaises(DetectorError):
            Detector("models/does-not-exist.pt")

    @patch("src.detection.detector.YOLO")
    def test_detect_returns_structured_results(self, yolo):
        yolo.return_value = make_model([FakeBox(0, 0.93, [10.4, 20.6, 100.2, 200.9])])

        detections = Detector().detect("frame")

        self.assertEqual(len(detections), 1)
        self.assertIsInstance(detections[0], Detection)
        self.assertEqual(detections[0].label, "person")
        self.assertAlmostEqual(detections[0].confidence, 0.93)
        self.assertEqual(detections[0].box, (10, 20, 100, 200))

    @patch("src.detection.detector.YOLO")
    def test_person_only_filters_to_the_person_class(self, yolo):
        model = make_model()
        yolo.return_value = model

        Detector(person_only=True).detect("frame")

        self.assertEqual(model.predict.call_args.kwargs["classes"], [0])

    @patch("src.detection.detector.YOLO")
    def test_person_only_disabled_keeps_every_class(self, yolo):
        model = make_model()
        yolo.return_value = model

        Detector(person_only=False).detect("frame")

        self.assertIsNone(model.predict.call_args.kwargs["classes"])

    @patch("src.detection.detector.YOLO")
    def test_confidence_is_passed_to_the_model(self, yolo):
        model = make_model()
        yolo.return_value = model

        Detector(confidence=0.75).detect("frame")

        self.assertEqual(model.predict.call_args.kwargs["conf"], 0.75)

    @patch("src.detection.detector.YOLO")
    def test_detect_returns_empty_list_when_nothing_is_found(self, yolo):
        yolo.return_value = make_model()

        self.assertEqual(Detector().detect("frame"), [])

    @patch("src.detection.detector.YOLO")
    def test_detect_handles_several_boxes(self, yolo):
        yolo.return_value = make_model(
            [
                FakeBox(0, 0.9, [0, 0, 10, 10]),
                FakeBox(0, 0.6, [20, 20, 30, 30]),
            ]
        )

        detections = Detector().detect("frame")

        self.assertEqual(len(detections), 2)
        self.assertEqual([d.confidence for d in detections], [0.9, 0.6])


if __name__ == "__main__":
    unittest.main()

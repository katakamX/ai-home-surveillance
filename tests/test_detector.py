"""Detector tests. YOLO is replaced with a mock, so no PyTorch, GPU, model
weights or internet access are needed to run them."""

import unittest
from unittest.mock import MagicMock, patch

from src.detection.detector import Detection, Detector, DetectorError


class FakeBox:
    """Stands in for one Ultralytics box (values are wrapped in sequences, as they are in the real thing)."""

    def __init__(self, class_id, confidence, xyxy):
        self.cls = [class_id]
        self.conf = [confidence]
        self.xyxy = [xyxy]


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


def make_model(boxes=(), results=None):
    """Build a fake YOLO model returning the given boxes (or whole results)."""
    model = MagicMock()
    model.names = {0: "person", 15: "cat"}
    model.predict.return_value = [FakeResult(list(boxes))] if results is None else results
    return model


class TestDetectorInitialisation(unittest.TestCase):
    @patch("src.detection.detector.YOLO")
    def test_model_is_loaded_once_and_reused(self, yolo):
        yolo.return_value = make_model()

        detector = Detector()
        detector.detect("frame")
        detector.detect("frame")

        yolo.assert_called_once()
        self.assertEqual(detector.model.predict.call_count, 2)

    @patch("src.detection.detector.YOLO")
    def test_model_path_is_stored_as_a_string(self, yolo):
        yolo.return_value = make_model()

        detector = Detector("models/custom.pt")

        self.assertEqual(detector.model_path, "models/custom.pt")
        yolo.assert_called_once_with("models/custom.pt")


class TestDetectorFailure(unittest.TestCase):
    @patch("src.detection.detector.YOLO")
    def test_raises_when_model_cannot_be_loaded(self, yolo):
        yolo.side_effect = FileNotFoundError("missing weights")

        with self.assertRaises(DetectorError):
            Detector("models/does-not-exist.pt")

    @patch("src.detection.detector.YOLO", None)
    def test_raises_when_ultralytics_is_missing(self):
        with self.assertRaises(DetectorError):
            Detector()

    @patch("src.detection.detector.YOLO")
    def test_inference_failure_raises_instead_of_returning_nothing(self, yolo):
        model = make_model()
        model.predict.side_effect = RuntimeError("CUDA out of memory")
        yolo.return_value = model

        with self.assertRaises(DetectorError):
            Detector().detect("frame")


class TestConfidenceThreshold(unittest.TestCase):
    @patch("src.detection.detector.YOLO")
    def test_confidence_is_passed_to_the_model(self, yolo):
        model = make_model()
        yolo.return_value = model

        Detector(confidence=0.75).detect("frame")

        self.assertEqual(model.predict.call_args.kwargs["conf"], 0.75)

    @patch("src.detection.detector.YOLO")
    def test_default_confidence_is_half(self, yolo):
        model = make_model()
        yolo.return_value = model

        Detector().detect("frame")

        self.assertEqual(model.predict.call_args.kwargs["conf"], 0.5)

    @patch("src.detection.detector.YOLO")
    def test_confidence_above_one_is_rejected(self, yolo):
        yolo.return_value = make_model()

        with self.assertRaises(ValueError):
            Detector(confidence=50)

    @patch("src.detection.detector.YOLO")
    def test_negative_confidence_is_rejected(self, yolo):
        yolo.return_value = make_model()

        with self.assertRaises(ValueError):
            Detector(confidence=-0.1)


class TestPersonOnlyFiltering(unittest.TestCase):
    @patch("src.detection.detector.YOLO")
    def test_person_only_filters_to_the_person_class(self, yolo):
        model = make_model()
        yolo.return_value = model

        Detector(person_only=True).detect("frame")

        self.assertEqual(model.predict.call_args.kwargs["classes"], [0])

    @patch("src.detection.detector.YOLO")
    def test_person_only_is_the_default(self, yolo):
        model = make_model()
        yolo.return_value = model

        Detector().detect("frame")

        self.assertEqual(model.predict.call_args.kwargs["classes"], [0])

    @patch("src.detection.detector.YOLO")
    def test_person_only_disabled_keeps_every_class(self, yolo):
        model = make_model()
        yolo.return_value = model

        Detector(person_only=False).detect("frame")

        self.assertIsNone(model.predict.call_args.kwargs["classes"])


class TestDetectionResults(unittest.TestCase):
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
    def test_detection_contains_only_plain_python_types(self, yolo):
        yolo.return_value = make_model([FakeBox(0, 0.9, [1.0, 2.0, 3.0, 4.0])])

        detection = Detector().detect("frame")[0]

        self.assertIsInstance(detection.label, str)
        self.assertIsInstance(detection.confidence, float)
        self.assertIsInstance(detection.box, tuple)
        self.assertTrue(all(isinstance(value, int) for value in detection.box))

    @patch("src.detection.detector.YOLO")
    def test_detect_returns_empty_list_when_nothing_is_found(self, yolo):
        yolo.return_value = make_model()

        self.assertEqual(Detector().detect("frame"), [])

    @patch("src.detection.detector.YOLO")
    def test_detect_handles_several_boxes(self, yolo):
        yolo.return_value = make_model(
            [FakeBox(0, 0.9, [0, 0, 10, 10]), FakeBox(0, 0.6, [20, 20, 30, 30])]
        )

        detections = Detector().detect("frame")

        self.assertEqual(len(detections), 2)
        self.assertEqual([d.confidence for d in detections], [0.9, 0.6])
        self.assertEqual(detections[1].box, (20, 20, 30, 30))

    @patch("src.detection.detector.YOLO")
    def test_detect_collects_boxes_from_every_result(self, yolo):
        yolo.return_value = make_model(
            results=[
                FakeResult([FakeBox(0, 0.9, [0, 0, 10, 10])]),
                FakeResult([FakeBox(0, 0.8, [5, 5, 15, 15])]),
            ]
        )

        self.assertEqual(len(Detector().detect("frame")), 2)


class TestMalformedModelOutput(unittest.TestCase):
    @patch("src.detection.detector.YOLO")
    def test_result_without_boxes_is_skipped(self, yolo):
        yolo.return_value = make_model(results=[FakeResult(None)])

        self.assertEqual(Detector().detect("frame"), [])

    @patch("src.detection.detector.YOLO")
    def test_box_with_too_few_coordinates_raises(self, yolo):
        yolo.return_value = make_model([FakeBox(0, 0.9, [1, 2, 3])])

        with self.assertRaises(DetectorError):
            Detector().detect("frame")

    @patch("src.detection.detector.YOLO")
    def test_unknown_class_id_raises(self, yolo):
        yolo.return_value = make_model([FakeBox(77, 0.9, [1, 2, 3, 4])])

        with self.assertRaises(DetectorError):
            Detector().detect("frame")

    @patch("src.detection.detector.YOLO")
    def test_non_numeric_coordinates_raise(self, yolo):
        yolo.return_value = make_model([FakeBox(0, 0.9, ["a", "b", "c", "d"])])

        with self.assertRaises(DetectorError):
            Detector().detect("frame")


if __name__ == "__main__":
    unittest.main()

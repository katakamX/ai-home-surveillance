"""Tracker tests. Detections are synthetic, plain Detection objects: no
camera, GPU, YOLO or model weights are needed to run them."""

import unittest

from src.detection.detector import Detection
from src.tracking.tracker import Tracker, TrackedObject, _iou


def person(box, confidence=0.9, label="person"):
    return Detection(label=label, confidence=confidence, box=box)


class TestIou(unittest.TestCase):
    def test_identical_boxes_have_iou_one(self):
        self.assertEqual(_iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)

    def test_disjoint_boxes_have_iou_zero(self):
        self.assertEqual(_iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)

    def test_partial_overlap_is_between_zero_and_one(self):
        iou = _iou((0, 0, 10, 10), (5, 5, 15, 15))
        self.assertGreater(iou, 0.0)
        self.assertLess(iou, 1.0)


class TestTrackerBasics(unittest.TestCase):
    def test_new_detection_starts_a_track(self):
        tracker = Tracker()
        result = tracker.update([person((0, 0, 10, 10))])

        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], TrackedObject)
        self.assertEqual(result[0].track_id, 1)
        self.assertEqual(result[0].box, (0, 0, 10, 10))
        self.assertEqual(result[0].label, "person")
        self.assertEqual(result[0].confidence, 0.9)

    def test_empty_detections_return_no_tracks(self):
        tracker = Tracker()
        self.assertEqual(tracker.update([]), [])

    def test_track_ids_increase_for_each_new_person(self):
        tracker = Tracker()
        result = tracker.update([person((0, 0, 10, 10)), person((50, 50, 60, 60))])

        track_ids = {tracked.track_id for tracked in result}
        self.assertEqual(track_ids, {1, 2})


class TestTrackerContinuity(unittest.TestCase):
    def test_same_person_keeps_the_same_track_id_across_frames(self):
        tracker = Tracker()
        first = tracker.update([person((0, 0, 10, 10))])

        # The person moves slightly: boxes overlap heavily between frames.
        second = tracker.update([person((1, 1, 11, 11))])

        self.assertEqual(first[0].track_id, second[0].track_id)

    def test_track_survives_a_single_missed_frame(self):
        tracker = Tracker(max_missed_frames=2)
        tracker.update([person((0, 0, 10, 10))])

        result = tracker.update([])  # detector missed the person for one frame

        self.assertEqual(len(result), 1)

    def test_track_survives_a_burst_of_missed_frames(self):
        # A real detector loses a marginally-visible person for several frames
        # in a row, not just one, so the default persistence has to cover a
        # burst rather than a single blink.
        tracker = Tracker()
        first = tracker.update([person((0, 0, 10, 10))])

        for _ in range(6):
            tracker.update([])
        result = tracker.update([person((1, 1, 11, 11))])

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].track_id, first[0].track_id)

    def test_track_is_dropped_after_too_many_missed_frames(self):
        tracker = Tracker(max_missed_frames=1)
        tracker.update([person((0, 0, 10, 10))])

        tracker.update([])
        result = tracker.update([])

        self.assertEqual(result, [])

    def test_reappearing_person_gets_a_new_track_id(self):
        tracker = Tracker(max_missed_frames=1)
        first = tracker.update([person((0, 0, 10, 10))])

        tracker.update([])
        tracker.update([])  # track dropped here

        second = tracker.update([person((0, 0, 10, 10))])

        self.assertNotEqual(first[0].track_id, second[0].track_id)

    def test_track_updates_confidence_and_box_each_frame(self):
        tracker = Tracker()
        tracker.update([person((0, 0, 10, 10), confidence=0.5)])
        result = tracker.update([person((2, 2, 12, 12), confidence=0.95)])

        self.assertEqual(result[0].box, (2, 2, 12, 12))
        self.assertEqual(result[0].confidence, 0.95)


class TestTrackerMultiplePeople(unittest.TestCase):
    def test_two_people_are_tracked_independently(self):
        tracker = Tracker()
        first = tracker.update([person((0, 0, 10, 10)), person((100, 100, 110, 110))])
        second = tracker.update([person((1, 1, 11, 11)), person((101, 101, 111, 111))])

        first_ids = {tracked.box: tracked.track_id for tracked in first}
        second_ids = {tracked.box: tracked.track_id for tracked in second}

        self.assertEqual(len(second), 2)
        # Each box moved only slightly, so both should keep their track_id.
        self.assertEqual(set(first_ids.values()), set(second_ids.values()))

    def test_non_overlapping_boxes_never_merge_into_one_track(self):
        tracker = Tracker()
        tracker.update([person((0, 0, 10, 10)), person((100, 100, 110, 110))])
        result = tracker.update([person((0, 0, 10, 10)), person((100, 100, 110, 110))])

        track_ids = {tracked.track_id for tracked in result}
        self.assertEqual(len(track_ids), 2)

    def test_far_apart_new_detection_does_not_steal_existing_track(self):
        tracker = Tracker()
        tracker.update([person((0, 0, 10, 10))])

        # A second, unrelated person appears far away.
        result = tracker.update([person((0, 0, 10, 10)), person((200, 200, 210, 210))])

        self.assertEqual(len(result), 2)
        track_ids = {tracked.track_id for tracked in result}
        self.assertEqual(track_ids, {1, 2})


if __name__ == "__main__":
    unittest.main()

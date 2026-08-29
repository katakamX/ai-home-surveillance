"""Regression tests for the false enter/exit storm seen on a real webcam.

A person the detector only marginally sees is not lost once and for all: YOLO
drops them for a burst of frames, finds them again, drops them again. On the
real camera that turned one person standing in one place into 41 track IDs and
35 separate "visits" in two minutes, each with its own clip and metadata file.

The cause was a pair of assumptions that only hold when detection is perfect:
the tracker gave up on an unmatched track after 5 frames, and the event engine
treated a track missing from the frame as an instant departure. Between them,
one dropout ended the visit, and the next detection -- of the same person, who
had not moved -- arrived as a new track_id and opened a new one.

Everything here is synthetic: the detections are plain Detection objects, so no
camera, GPU, YOLO or model weights are involved, and the pattern is fixed
rather than timed, so the tests are deterministic.
"""

import unittest

from src.detection.detector import Detection
from src.events.events import (
    EVENT_PERSON_ENTERED_ZONE,
    EVENT_PERSON_EXITED_ZONE,
    EventEngine,
    Zone,
)
from src.tracking.tracker import Tracker

DOOR = Zone(name="door", box=(0, 0, 100, 100))
INSIDE_DOOR = (40, 40, 60, 60)  # centre (50, 50): inside DOOR


def person(box=INSIDE_DOOR):
    return Detection(label="person", confidence=0.9, box=box)


def replay(visibility, tracker=None, engine=None):
    """Feed a seen/not-seen pattern through a real Tracker and EventEngine.

    visibility is one bool per frame: True on the frames the detector saw the
    person, False on the frames it missed them. The person never moves, so any
    event at all beyond the first enter is the churn this module guards against.

    Returns (event_types, track_ids_seen).
    """
    tracker = tracker if tracker is not None else Tracker()
    engine = engine if engine is not None else EventEngine([DOOR])

    event_types = []
    track_ids = set()

    for frame_number, seen in enumerate(visibility):
        tracked_objects = tracker.update([person()] if seen else [])
        track_ids.update(obj.track_id for obj in tracked_objects)
        for event in engine.update(tracked_objects, timestamp=float(frame_number)):
            event_types.append(event.event_type)

    return event_types, track_ids


class TestDropoutsDoNotEndAVisit(unittest.TestCase):
    def test_a_dropout_longer_than_the_old_limit_keeps_one_visit(self):
        # Six missed frames: past the old limit of five, and still well under a
        # second at the frame rates this pipeline reaches on a laptop CPU.
        events, track_ids = replay([True] * 10 + [False] * 6 + [True] * 10)

        self.assertEqual(events, [EVENT_PERSON_ENTERED_ZONE])
        self.assertEqual(len(track_ids), 1)

    def test_repeated_marginal_dropouts_are_one_visit_not_a_storm(self):
        # Bursts of the length measured on the real camera, one after another,
        # while the person stands still. This is the reproduction of the bug.
        visibility = []
        for burst in (1, 3, 2, 5, 4, 7, 2, 6, 3, 5):
            visibility += [True] * 4 + [False] * burst
        visibility += [True] * 4

        events, track_ids = replay(visibility)

        self.assertEqual(events, [EVENT_PERSON_ENTERED_ZONE])
        self.assertEqual(len(track_ids), 1)

    def test_the_person_keeps_one_track_id_across_every_dropout(self):
        _, track_ids = replay([True] * 5 + [False] * 8 + [True] * 5 + [False] * 9 + [True] * 5)

        self.assertEqual(track_ids, {1})


class TestRealDeparturesStillWork(unittest.TestCase):
    """The fix must not make the system blind to somebody actually leaving."""

    def test_a_person_who_really_leaves_gets_exactly_one_exit(self):
        events, _ = replay([True] * 10 + [False] * 40)

        self.assertEqual(events, [EVENT_PERSON_ENTERED_ZONE, EVENT_PERSON_EXITED_ZONE])

    def test_the_exit_is_not_repeated_however_long_they_stay_away(self):
        events, _ = replay([True] * 5 + [False] * 200)

        self.assertEqual(events.count(EVENT_PERSON_EXITED_ZONE), 1)

    def test_returning_after_a_real_departure_is_exactly_one_new_enter(self):
        events, _ = replay([True] * 10 + [False] * 40 + [True] * 10)

        self.assertEqual(
            events,
            [EVENT_PERSON_ENTERED_ZONE, EVENT_PERSON_EXITED_ZONE, EVENT_PERSON_ENTERED_ZONE],
        )

    def test_two_real_visits_produce_two_enters_and_two_exits(self):
        events, _ = replay(([True] * 8 + [False] * 40) * 2)

        self.assertEqual(
            events,
            [
                EVENT_PERSON_ENTERED_ZONE,
                EVENT_PERSON_EXITED_ZONE,
                EVENT_PERSON_ENTERED_ZONE,
                EVENT_PERSON_EXITED_ZONE,
            ],
        )


class TestTheOldSettingsStillChurn(unittest.TestCase):
    """Guards the fix itself: the same input, with the persistence and the
    instant-exit behaviour that shipped before, reproduces the storm. If this
    ever stops churning, the tests above have stopped proving anything.
    """

    def test_short_persistence_and_instant_exit_reproduce_the_churn(self):
        visibility = []
        for burst in (6, 7, 6, 8, 6):
            visibility += [True] * 4 + [False] * burst
        visibility += [True] * 4

        events, track_ids = replay(
            visibility,
            tracker=Tracker(max_missed_frames=5),
            engine=EventEngine([DOOR], exit_frames=1),
        )

        self.assertGreater(events.count(EVENT_PERSON_ENTERED_ZONE), 1)
        self.assertGreater(events.count(EVENT_PERSON_EXITED_ZONE), 1)
        self.assertGreater(len(track_ids), 1)

    def test_the_same_input_is_a_single_visit_with_the_current_defaults(self):
        visibility = []
        for burst in (6, 7, 6, 8, 6):
            visibility += [True] * 4 + [False] * burst
        visibility += [True] * 4

        events, track_ids = replay(visibility)

        self.assertEqual(events, [EVENT_PERSON_ENTERED_ZONE])
        self.assertEqual(len(track_ids), 1)


if __name__ == "__main__":
    unittest.main()

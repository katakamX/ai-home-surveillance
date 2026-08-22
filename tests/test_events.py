"""Event layer tests. Tracked objects are synthetic: no camera, GPU, YOLO or
model weights are needed to run them."""

import unittest

from src.events.events import (
    EVENT_PERSON_ENTERED_ZONE,
    EVENT_PERSON_EXITED_ZONE,
    Event,
    EventEngine,
    Zone,
)
from src.tracking.tracker import TrackedObject


DOOR = Zone(name="door", box=(0, 0, 100, 100))
DRIVEWAY = Zone(name="driveway", box=(200, 200, 300, 300))


def tracked(track_id, box, confidence=0.9, label="person"):
    return TrackedObject(track_id=track_id, box=box, label=label, confidence=confidence)


def inside_door(track_id=1, confidence=0.9):
    return tracked(track_id, (40, 40, 60, 60), confidence=confidence)


def outside_all_zones(track_id=1):
    return tracked(track_id, (500, 500, 520, 520))


class TestZone(unittest.TestCase):
    def test_point_inside_is_contained(self):
        self.assertTrue(DOOR.contains((50, 50)))

    def test_point_outside_is_not_contained(self):
        self.assertFalse(DOOR.contains((500, 50)))

    def test_point_on_the_edge_is_contained(self):
        self.assertTrue(DOOR.contains((100, 100)))


class TestEnterEvents(unittest.TestCase):
    def test_person_entering_a_zone_generates_one_enter_event(self):
        engine = EventEngine([DOOR])
        events = engine.update([inside_door()], timestamp=1.0)

        self.assertEqual(len(events), 1)
        self.assertIsInstance(events[0], Event)
        self.assertEqual(events[0].event_type, EVENT_PERSON_ENTERED_ZONE)
        self.assertEqual(events[0].track_id, 1)
        self.assertEqual(events[0].zone, "door")
        self.assertEqual(events[0].label, "person")
        self.assertEqual(events[0].confidence, 0.9)
        self.assertEqual(events[0].timestamp, 1.0)

    def test_staying_in_a_zone_does_not_repeat_the_enter_event(self):
        engine = EventEngine([DOOR])
        engine.update([inside_door()], timestamp=1.0)

        self.assertEqual(engine.update([inside_door()], timestamp=2.0), [])
        self.assertEqual(engine.update([inside_door()], timestamp=3.0), [])

    def test_person_outside_every_zone_generates_no_event(self):
        engine = EventEngine([DOOR])
        self.assertEqual(engine.update([outside_all_zones()], timestamp=1.0), [])

    def test_no_tracked_objects_generates_no_event(self):
        engine = EventEngine([DOOR])
        self.assertEqual(engine.update([], timestamp=1.0), [])

    def test_event_ids_are_unique(self):
        engine = EventEngine([DOOR, DRIVEWAY])
        first = engine.update([inside_door(track_id=1)], timestamp=1.0)
        second = engine.update(
            [inside_door(track_id=1), tracked(2, (250, 250, 260, 260))], timestamp=2.0
        )

        self.assertNotEqual(first[0].event_id, second[0].event_id)


class TestExitEvents(unittest.TestCase):
    def test_leaving_a_zone_generates_one_exit_event(self):
        engine = EventEngine([DOOR], exit_frames=2)
        engine.update([inside_door()], timestamp=1.0)

        engine.update([outside_all_zones()], timestamp=1.5)
        events = engine.update([outside_all_zones()], timestamp=2.0)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, EVENT_PERSON_EXITED_ZONE)
        self.assertEqual(events[0].track_id, 1)
        self.assertEqual(events[0].zone, "door")
        self.assertEqual(events[0].timestamp, 2.0)

    def test_exit_event_is_not_repeated(self):
        engine = EventEngine([DOOR], exit_frames=1)
        engine.update([inside_door()], timestamp=1.0)
        engine.update([outside_all_zones()], timestamp=2.0)

        self.assertEqual(engine.update([outside_all_zones()], timestamp=3.0), [])

    def test_disappearing_from_the_frame_counts_as_an_exit(self):
        engine = EventEngine([DOOR])
        engine.update([inside_door()], timestamp=1.0)

        events = engine.update([], timestamp=2.0)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, EVENT_PERSON_EXITED_ZONE)

    def test_exit_event_keeps_the_last_seen_label_and_confidence(self):
        engine = EventEngine([DOOR])
        engine.update([inside_door(confidence=0.77)], timestamp=1.0)

        events = engine.update([], timestamp=2.0)

        self.assertEqual(events[0].label, "person")
        self.assertEqual(events[0].confidence, 0.77)

    def test_re_entering_generates_a_new_enter_event(self):
        engine = EventEngine([DOOR])
        engine.update([inside_door()], timestamp=1.0)
        engine.update([], timestamp=2.0)

        events = engine.update([inside_door()], timestamp=3.0)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, EVENT_PERSON_ENTERED_ZONE)


class TestBoundaryJitter(unittest.TestCase):
    """Regression tests for the false enter/exit pairs seen on the real webcam,
    where a person standing on the zone edge had their box center wobble in and
    out of the zone from one frame to the next."""

    def just_outside_door(self, track_id=1):
        # Center at (105, 50): a few pixels past the door zone's x2 = 100.
        return tracked(track_id, (100, 40, 110, 60))

    def test_single_frame_flicker_out_does_not_emit_an_exit(self):
        engine = EventEngine([DOOR])
        engine.update([inside_door()], timestamp=1.0)

        events = engine.update([self.just_outside_door()], timestamp=1.03)

        self.assertEqual(events, [])

    def test_alternating_in_out_frames_emit_no_events_after_the_enter(self):
        engine = EventEngine([DOOR])
        entered = engine.update([inside_door()], timestamp=1.0)
        self.assertEqual(len(entered), 1)

        # The exact pattern observed on camera: one frame out, one frame in.
        for frame in range(10):
            objects = [self.just_outside_door() if frame % 2 else inside_door()]
            self.assertEqual(engine.update(objects, timestamp=1.0 + frame * 0.03), [])

    def test_sustained_exit_still_emits_exactly_one_exit(self):
        engine = EventEngine([DOOR], exit_frames=3)
        engine.update([inside_door()], timestamp=1.0)

        emitted = []
        for frame in range(10):
            emitted.extend(engine.update([outside_all_zones()], timestamp=2.0 + frame * 0.03))

        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0].event_type, EVENT_PERSON_EXITED_ZONE)

    def test_real_re_entry_after_a_sustained_exit_emits_one_enter(self):
        engine = EventEngine([DOOR], exit_frames=2)
        engine.update([inside_door()], timestamp=1.0)
        for frame in range(5):
            engine.update([outside_all_zones()], timestamp=2.0 + frame * 0.03)

        events = engine.update([inside_door()], timestamp=3.0)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, EVENT_PERSON_ENTERED_ZONE)

    def test_flicker_resets_the_exit_countdown(self):
        engine = EventEngine([DOOR], exit_frames=3)
        engine.update([inside_door()], timestamp=1.0)

        engine.update([self.just_outside_door()], timestamp=1.03)
        engine.update([self.just_outside_door()], timestamp=1.06)
        engine.update([inside_door()], timestamp=1.09)  # back in: countdown resets
        engine.update([self.just_outside_door()], timestamp=1.12)
        events = engine.update([self.just_outside_door()], timestamp=1.15)

        self.assertEqual(events, [])

    def test_track_disappearing_exits_immediately_without_waiting(self):
        engine = EventEngine([DOOR], exit_frames=10)
        engine.update([inside_door()], timestamp=1.0)

        events = engine.update([], timestamp=1.03)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, EVENT_PERSON_EXITED_ZONE)


class TestMultipleZonesAndPeople(unittest.TestCase):
    def test_each_person_gets_their_own_enter_event(self):
        engine = EventEngine([DOOR])
        events = engine.update(
            [inside_door(track_id=1), tracked(2, (10, 10, 30, 30))], timestamp=1.0
        )

        self.assertEqual(len(events), 2)
        self.assertEqual({event.track_id for event in events}, {1, 2})

    def test_one_person_in_overlapping_zones_enters_both(self):
        overlapping = Zone(name="porch", box=(0, 0, 200, 200))
        engine = EventEngine([DOOR, overlapping])

        events = engine.update([inside_door()], timestamp=1.0)

        self.assertEqual({event.zone for event in events}, {"door", "porch"})

    def test_zones_are_tracked_independently(self):
        engine = EventEngine([DOOR, DRIVEWAY])
        engine.update([inside_door(track_id=1)], timestamp=1.0)

        # Track 1 stays put; a second person appears in the driveway.
        events = engine.update(
            [inside_door(track_id=1), tracked(2, (250, 250, 260, 260))], timestamp=2.0
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].zone, "driveway")
        self.assertEqual(events[0].track_id, 2)

    def test_moving_between_zones_emits_an_exit_and_an_enter(self):
        engine = EventEngine([DOOR, DRIVEWAY], exit_frames=1)
        engine.update([inside_door(track_id=1)], timestamp=1.0)

        events = engine.update([tracked(1, (250, 250, 260, 260))], timestamp=2.0)

        types = {(event.event_type, event.zone) for event in events}
        self.assertEqual(
            types,
            {
                (EVENT_PERSON_EXITED_ZONE, "door"),
                (EVENT_PERSON_ENTERED_ZONE, "driveway"),
            },
        )


if __name__ == "__main__":
    unittest.main()

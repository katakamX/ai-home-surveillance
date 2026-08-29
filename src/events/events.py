"""Surveillance events. Turns tracked people into zone enter/exit events.

Independent of OpenCV, YOLO and the camera: it only consumes TrackedObject
results from src.tracking.tracker, so it can be unit tested with synthetic
data and reused unchanged on a Jetson.
"""

from dataclasses import dataclass
from itertools import count

from src.tracking.tracker import TrackedObject

EVENT_PERSON_ENTERED_ZONE = "person_entered_zone"
EVENT_PERSON_EXITED_ZONE = "person_exited_zone"

# A person must be seen outside a zone for this many consecutive frames before
# an exit is reported. Detection boxes wobble a few pixels every frame, so
# without this a person standing on the zone edge produces a stream of false
# enter/exit pairs.
DEFAULT_EXIT_FRAMES = 5

_event_id_counter = count(1)


@dataclass
class Zone:
    """A rectangular area of interest, in the same pixel coordinates as boxes."""

    name: str
    box: tuple[int, int, int, int]  # (x1, y1, x2, y2)

    def contains(self, point: tuple[int, int]) -> bool:
        x, y = point
        x1, y1, x2, y2 = self.box
        return x1 <= x <= x2 and y1 <= y <= y2


@dataclass
class Event:
    """One surveillance event."""

    event_id: int
    timestamp: float
    event_type: str
    track_id: int
    label: str
    zone: str
    confidence: float


def _center(box: tuple[int, int, int, int]) -> tuple[int, int]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)


class EventEngine:
    """Converts tracked objects into person_entered_zone / person_exited_zone events.

    Call update() once per frame with that frame's TrackedObject list and the
    current time. It remembers which (track_id, zone) pairs are occupied, so a
    person crossing into a zone produces exactly one enter event and leaving
    produces exactly one exit event.

    Entering is reported immediately, but leaving must be confirmed by
    exit_frames consecutive frames of not being inside the zone. That
    hysteresis absorbs the few-pixel box wobble at the zone edge that would
    otherwise produce rapid false enter/exit pairs, and it applies just as much
    to a track that is missing from the frame altogether: the detector drops a
    marginally-visible person for a burst of frames at a time, and treating
    that as an immediate departure is what produced storms of false
    enter/exit pairs on a real camera. A person who is really gone stays
    un-seen, so the countdown runs out and they get exactly one exit.
    """

    def __init__(self, zones: list[Zone], exit_frames: int = DEFAULT_EXIT_FRAMES) -> None:
        self.zones = zones
        self.exit_frames = exit_frames
        self._occupied: set[tuple[int, str]] = set()  # (track_id, zone_name)
        # Consecutive frames each occupied pair has been seen outside its zone.
        self._frames_outside: dict[tuple[int, str], int] = {}
        # Last label/confidence seen per track, so an exit event can describe a
        # person who is no longer in the frame.
        self._last_seen: dict[int, tuple[str, float]] = {}

    def update(self, tracked_objects: list[TrackedObject], timestamp: float) -> list[Event]:
        events: list[Event] = []
        inside_now: set[tuple[int, str]] = set()

        for tracked in tracked_objects:
            self._last_seen[tracked.track_id] = (tracked.label, tracked.confidence)
            point = _center(tracked.box)
            for zone in self.zones:
                if not zone.contains(point):
                    continue

                key = (tracked.track_id, zone.name)
                inside_now.add(key)
                self._frames_outside.pop(key, None)

                if key not in self._occupied:
                    self._occupied.add(key)
                    events.append(
                        self._make_event(
                            EVENT_PERSON_ENTERED_ZONE, tracked, zone.name, timestamp
                        )
                    )

        for key in sorted(self._occupied - inside_now):
            track_id, zone_name = key
            # An occupied pair can stop being inside for two reasons: the person
            # is still on screen but stepped outside, or the track is missing
            # from this frame entirely. For a frame or two those look identical
            # and both settle on their own -- edge wobble on one side, a
            # detector dropout on the other -- so both wait out exit_frames
            # before an exit is reported. Exiting an absent track at once is
            # what turned every brief dropout into a false exit/enter pair.
            self._frames_outside[key] = self._frames_outside.get(key, 0) + 1
            if self._frames_outside[key] < self.exit_frames:
                continue

            self._occupied.discard(key)
            self._frames_outside.pop(key, None)
            events.append(self._make_exit_event(track_id, zone_name, timestamp))

        self._forget_gone_tracks()
        return events

    def _make_exit_event(self, track_id: int, zone_name: str, timestamp: float) -> Event:
        label, confidence = self._last_seen.get(track_id, ("person", 0.0))
        return Event(
            event_id=next(_event_id_counter),
            timestamp=timestamp,
            event_type=EVENT_PERSON_EXITED_ZONE,
            track_id=track_id,
            label=label,
            zone=zone_name,
            confidence=confidence,
        )

    def _forget_gone_tracks(self) -> None:
        """Drop remembered info for tracks that no longer occupy any zone."""
        still_tracked = {track_id for track_id, _ in self._occupied}
        self._last_seen = {
            track_id: info
            for track_id, info in self._last_seen.items()
            if track_id in still_tracked
        }

    def _make_event(
        self, event_type: str, tracked: TrackedObject, zone_name: str, timestamp: float
    ) -> Event:
        return Event(
            event_id=next(_event_id_counter),
            timestamp=timestamp,
            event_type=event_type,
            track_id=tracked.track_id,
            label=tracked.label,
            zone=zone_name,
            confidence=tracked.confidence,
        )

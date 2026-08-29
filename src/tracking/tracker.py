"""Multi-object tracking. Links detections across frames into stable tracks.

Uses greedy IoU (intersection-over-union) matching: each existing track is
paired with the detection whose box overlaps it most, frame after frame. This
needs no camera, no GPU and no ML model, so it works the same on a laptop and
on a Jetson Nano.
"""

from dataclasses import dataclass

from src.detection.detector import Detection

# A track that goes unmatched for more than this many consecutive frames is
# dropped, so a person who leaves the frame does not linger forever.
#
# Sized from real webcam measurements rather than picked by feel. A person the
# detector only marginally sees (YOLO confidence sitting near the threshold)
# is not lost once: the detections drop out in bursts. In a measured marginal
# run, bursts of up to 12 frames covered ~95% of all dropouts, and at the
# ~12 FPS this pipeline reaches on a laptop CPU that is roughly one second:
# long enough to ride out the flicker, short enough that somebody who really
# walked away is not held on to. The old value of 5 was under half of that, so
# a single burst ended the track, the next detection became a *new* track_id,
# and every flicker turned into a spurious exit/enter pair downstream.
DEFAULT_MAX_MISSED_FRAMES = 12

# Boxes must overlap by at least this fraction to be considered the same
# object from one frame to the next.
DEFAULT_IOU_THRESHOLD = 0.3


@dataclass
class TrackedObject:
    """One tracked person, current as of the latest update() call."""

    track_id: int
    box: tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixels
    label: str
    confidence: float


class _Track:
    """Internal bookkeeping for one track. Not exposed outside this module."""

    def __init__(self, track_id: int, detection: Detection) -> None:
        self.track_id = track_id
        self.box = detection.box
        self.label = detection.label
        self.confidence = detection.confidence
        self.missed_frames = 0

    def update(self, detection: Detection) -> None:
        self.box = detection.box
        self.label = detection.label
        self.confidence = detection.confidence
        self.missed_frames = 0

    def mark_missed(self) -> None:
        self.missed_frames += 1

    def as_tracked_object(self) -> TrackedObject:
        return TrackedObject(
            track_id=self.track_id,
            box=self.box,
            label=self.label,
            confidence=self.confidence,
        )


class Tracker:
    """Assigns a stable track_id to each person across consecutive frames.

    Call update() once per frame with that frame's Detection objects. Track
    IDs are never reused: once assigned, a track_id belongs to that person for
    the rest of the run, even after the track is dropped.
    """

    def __init__(
        self,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        max_missed_frames: int = DEFAULT_MAX_MISSED_FRAMES,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames
        self._tracks: list[_Track] = []
        self._next_track_id = 1

    def update(self, detections: list[Detection]) -> list[TrackedObject]:
        """Match this frame's detections to existing tracks and return the result.

        Matching is greedy: the highest-overlap (track, detection) pair is
        confirmed first, then the next-highest among what remains, and so on.
        Unmatched detections start new tracks; unmatched tracks are kept alive
        for a few frames (in case of a missed detection) before being dropped.
        """
        matches, unmatched_tracks, unmatched_detections = self._match(detections)

        for track_index, detection_index in matches:
            self._tracks[track_index].update(detections[detection_index])

        for track_index in unmatched_tracks:
            self._tracks[track_index].mark_missed()

        for detection_index in unmatched_detections:
            self._tracks.append(_Track(self._next_track_id, detections[detection_index]))
            self._next_track_id += 1

        self._tracks = [
            track for track in self._tracks if track.missed_frames <= self.max_missed_frames
        ]

        return [track.as_tracked_object() for track in self._tracks]

    def _match(
        self, detections: list[Detection]
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Greedily pair tracks with detections by descending IoU.

        Returns (matches, unmatched_track_indices, unmatched_detection_indices),
        where matches is a list of (track_index, detection_index) pairs.
        """
        candidates = []
        for track_index, track in enumerate(self._tracks):
            for detection_index, detection in enumerate(detections):
                iou = _iou(track.box, detection.box)
                if iou >= self.iou_threshold:
                    candidates.append((iou, track_index, detection_index))

        candidates.sort(key=lambda candidate: candidate[0], reverse=True)

        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        matches: list[tuple[int, int]] = []

        for _, track_index, detection_index in candidates:
            if track_index in matched_tracks or detection_index in matched_detections:
                continue
            matches.append((track_index, detection_index))
            matched_tracks.add(track_index)
            matched_detections.add(detection_index)

        unmatched_tracks = [
            index for index in range(len(self._tracks)) if index not in matched_tracks
        ]
        unmatched_detections = [
            index for index in range(len(detections)) if index not in matched_detections
        ]

        return matches, unmatched_tracks, unmatched_detections


def _iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    """Intersection-over-union of two (x1, y1, x2, y2) boxes, in [0.0, 1.0]."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    intersection = inter_width * inter_height

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union

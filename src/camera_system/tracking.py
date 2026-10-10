"""Small deterministic IoU tracker for the Phase 4 baseline."""

from dataclasses import dataclass
from math import isfinite

from .contracts import Region


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


@dataclass
class _Track:
    track_id: str
    label: str | None
    box: tuple[float, float, float, float]
    missed: int = 0
    last_seen: float = 0.0
    velocity: tuple[float, float] = (0.0, 0.0)


class IoUTracker:
    def __init__(
        self,
        threshold: float = 0.3,
        max_missed: int = 5,
        max_age_seconds: float = 2.0,
        motion: bool = False,
    ):
        if not isfinite(threshold) or not 0 < threshold <= 1:
            raise ValueError("tracker IoU threshold must be in (0, 1]")
        if not isfinite(max_age_seconds) or max_age_seconds <= 0:
            raise ValueError("tracker max age must be positive and finite")
        if type(max_missed) is not int or max_missed < 0:
            raise ValueError("tracker max_missed must be a nonnegative integer")
        self.threshold, self.max_missed = threshold, max_missed
        self._tracks: dict[str, _Track] = {}
        self._next_id = 1
        self.max_age_seconds = max_age_seconds
        self._timestamp = None
        self.motion = motion

    def _matching_box(self, track, timestamp):
        if not self.motion:
            return track.box
        dt = timestamp - track.last_seen
        dx, dy = (v * dt for v in track.velocity)
        x1, y1, x2, y2 = track.box
        return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)

    def update(self, regions: list[Region], timestamp: float | None = None) -> list[Region]:
        timestamp = (
            (0 if self._timestamp is None else self._timestamp + 1)
            if timestamp is None
            else timestamp
        )
        if (
            not isfinite(timestamp)
            or timestamp < 0
            or (self._timestamp is not None and timestamp <= self._timestamp)
        ):
            raise ValueError("tracking timestamps must be finite, nonnegative and increasing")
        self._timestamp = timestamp
        self._tracks = {
            key: track
            for key, track in self._tracks.items()
            if timestamp - track.last_seen <= self.max_age_seconds
        }
        candidates = sorted(
            (
                (iou(self._matching_box(track, timestamp), region.box), track_id, index)
                for track_id, track in self._tracks.items()
                for index, region in enumerate(regions)
                if track.label is None or region.label is None or track.label == region.label
            ),
            reverse=True,
        )
        matches, used_tracks, used_regions = {}, set(), set()
        for score, track_id, index in candidates:
            if score < self.threshold or track_id in used_tracks or index in used_regions:
                continue
            matches[index] = track_id
            used_tracks.add(track_id)
            used_regions.add(index)
        for track_id, track in list(self._tracks.items()):
            if track_id not in used_tracks:
                track.missed += 1
                if track.missed > self.max_missed:
                    del self._tracks[track_id]
        output = []
        for index, region in enumerate(regions):
            track_id = matches.get(index)
            if track_id is None:
                track_id = f"track-{self._next_id}"
                self._next_id += 1
            old = self._tracks.get(track_id)
            velocity = (0.0, 0.0)
            if self.motion and old:
                dt = timestamp - old.last_seen
                velocity = tuple(
                    (
                        (region.box[axis] + region.box[axis + 2])
                        - (old.box[axis] + old.box[axis + 2])
                    )
                    / (2 * dt)
                    for axis in (0, 1)
                )
            self._tracks[track_id] = _Track(
                track_id, region.label, region.box, last_seen=timestamp, velocity=velocity
            )
            output.append(
                Region(region.id, region.box, region.label, track_id, region.detection_score)
            )
        return output

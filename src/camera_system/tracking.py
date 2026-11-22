"""Deterministic multi-object trackers (IoU, motion-IoU, and Kalman)."""

from dataclasses import dataclass
from math import isfinite

import numpy as np

from .contracts import Region

try:  # Optional SciPy Hungarian solver; falls back to greedy assignment when absent.
    from scipy.optimize import linear_sum_assignment as _linear_sum_assignment
except ImportError:  # pragma: no cover - optional dependency
    _linear_sum_assignment = None


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


def _validate_tracker_params(threshold: float, max_missed: int, max_age_seconds: float) -> None:
    if not isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("tracker IoU threshold must be in (0, 1]")
    if not isfinite(max_age_seconds) or max_age_seconds <= 0:
        raise ValueError("tracker max age must be positive and finite")
    if type(max_missed) is not int or max_missed < 0:
        raise ValueError("tracker max_missed must be a nonnegative integer")


def _resolve_timestamp(previous: float | None, timestamp: float | None) -> float:
    resolved = (0.0 if previous is None else previous + 1.0) if timestamp is None else timestamp
    if not isfinite(resolved) or resolved < 0 or (previous is not None and resolved <= previous):
        raise ValueError("tracking timestamps must be finite, nonnegative and increasing")
    return float(resolved)


class IoUTracker:
    def __init__(
        self,
        threshold: float = 0.3,
        max_missed: int = 5,
        max_age_seconds: float = 2.0,
        motion: bool = False,
    ):
        _validate_tracker_params(threshold, max_missed, max_age_seconds)
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
        timestamp = _resolve_timestamp(self._timestamp, timestamp)
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


_KALMAN_H = np.array(
    [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ],
    dtype=float,
)


@dataclass
class _KalmanTrack:
    track_id: str
    label: str | None
    box: tuple[float, float, float, float]
    state: np.ndarray
    covariance: np.ndarray
    missed: int = 0
    last_seen: float = 0.0
    last_updated: float = 0.0


def _box_to_measurement(box: tuple[float, float, float, float]) -> np.ndarray:
    x1, y1, x2, y2 = box
    w = max(float(x2 - x1), 1.0)
    h = max(float(y2 - y1), 1.0)
    return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5, w, h], dtype=float)


def _state_to_box(state: np.ndarray) -> tuple[float, float, float, float]:
    cx, cy = float(state[0]), float(state[1])
    w = max(float(state[4]), 1.0)
    h = max(float(state[5]), 1.0)
    return (cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5)


class KalmanTracker:
    """2D constant-velocity Kalman filter tracker preserving raw detection boxes."""

    def __init__(
        self,
        threshold: float = 0.3,
        max_missed: int = 5,
        max_age_seconds: float = 2.0,
        process_noise: float = 25.0,
        measurement_noise: float = 4.0,
    ):
        _validate_tracker_params(threshold, max_missed, max_age_seconds)
        if not isfinite(process_noise) or process_noise <= 0:
            raise ValueError("tracker process_noise must be positive and finite")
        if not isfinite(measurement_noise) or measurement_noise <= 0:
            raise ValueError("tracker measurement_noise must be positive and finite")
        self.threshold = threshold
        self.max_missed = max_missed
        self.max_age_seconds = max_age_seconds
        self.process_noise = float(process_noise)
        self.measurement_noise = float(measurement_noise)
        self._tracks: dict[str, _KalmanTrack] = {}
        self._next_id = 1
        self._timestamp: float | None = None

    def _predict_track(
        self, track: _KalmanTrack, timestamp: float
    ) -> tuple[np.ndarray, np.ndarray]:
        dt = max(timestamp - track.last_updated, 1e-4)
        transition = np.eye(6, dtype=float)
        transition[0, 2] = dt
        transition[1, 3] = dt
        q_pos = self.process_noise * (dt**2)
        q_vel = self.process_noise * 4.0 * dt
        q_size = self.process_noise * 0.25 * dt
        noise = np.diag([q_pos, q_pos, q_vel, q_vel, q_size, q_size])
        pred_state = transition @ track.state
        pred_state[4] = max(float(pred_state[4]), 1.0)
        pred_state[5] = max(float(pred_state[5]), 1.0)
        pred_cov = transition @ track.covariance @ transition.T + noise
        return pred_state, pred_cov

    def _assign(
        self,
        track_items: list[tuple[str, _KalmanTrack]],
        predicted_boxes: list[tuple[float, float, float, float]],
        regions: list[Region],
    ) -> tuple[dict[int, str], set[str]]:
        if not track_items or not regions:
            return {}, set()
        if _linear_sum_assignment is not None:
            n_tracks = len(track_items)
            n_regions = len(regions)
            cost = np.full((n_tracks, n_regions + n_tracks), 1e6, dtype=float)
            scores = np.zeros((n_tracks, n_regions), dtype=float)
            for r_idx, (track_id, track) in enumerate(track_items):
                pred_box = predicted_boxes[r_idx]
                cost[r_idx, n_regions + r_idx] = 1.0  # allow leaving track unmatched
                for c_idx, region in enumerate(regions):
                    if (
                        track.label is not None
                        and region.label is not None
                        and track.label != region.label
                    ):
                        continue
                    score = iou(pred_box, region.box)
                    scores[r_idx, c_idx] = score
                    if score >= self.threshold:
                        cost[r_idx, c_idx] = 1.0 - score
            row_ind, col_ind = _linear_sum_assignment(cost)
            matches: dict[int, str] = {}
            used_tracks: set[str] = set()
            for r_idx, c_idx in zip(row_ind, col_ind, strict=True):
                if c_idx < n_regions and scores[r_idx, c_idx] >= self.threshold:
                    track_id = track_items[r_idx][0]
                    matches[int(c_idx)] = track_id
                    used_tracks.add(track_id)
            return matches, used_tracks

        candidates = sorted(
            (
                (iou(predicted_boxes[t_idx], region.box), track_id, r_idx)
                for t_idx, (track_id, track) in enumerate(track_items)
                for r_idx, region in enumerate(regions)
                if track.label is None or region.label is None or track.label == region.label
            ),
            reverse=True,
        )
        matches = {}
        used_tracks = set()
        used_regions = set()
        for score, track_id, r_idx in candidates:
            if score < self.threshold or track_id in used_tracks or r_idx in used_regions:
                continue
            matches[r_idx] = track_id
            used_tracks.add(track_id)
            used_regions.add(r_idx)
        return matches, used_tracks

    def update(self, regions: list[Region], timestamp: float | None = None) -> list[Region]:
        timestamp = _resolve_timestamp(self._timestamp, timestamp)
        self._timestamp = timestamp
        self._tracks = {
            key: track
            for key, track in self._tracks.items()
            if timestamp - track.last_seen <= self.max_age_seconds
        }
        track_items = list(self._tracks.items())
        predicted = [self._predict_track(track, timestamp) for _, track in track_items]
        predicted_boxes = [_state_to_box(state) for state, _ in predicted]
        pred_by_id = {
            track_id: pred for (track_id, _), pred in zip(track_items, predicted, strict=True)
        }
        matches, used_tracks = self._assign(track_items, predicted_boxes, regions)

        for track_id, track in list(self._tracks.items()):
            if track_id not in used_tracks:
                track.missed += 1
                if track.missed > self.max_missed:
                    del self._tracks[track_id]
                else:
                    pred_state, pred_cov = pred_by_id[track_id]
                    track.state = pred_state
                    track.covariance = pred_cov
                    track.box = _state_to_box(pred_state)
                    track.last_updated = timestamp

        meas_noise = np.eye(4, dtype=float) * self.measurement_noise
        eye6 = np.eye(6, dtype=float)
        output: list[Region] = []
        for index, region in enumerate(regions):
            track_id = matches.get(index)
            z = _box_to_measurement(region.box)
            if track_id is None:
                track_id = f"track-{self._next_id}"
                self._next_id += 1
                state = np.array([z[0], z[1], 0.0, 0.0, z[2], z[3]], dtype=float)
                cov = np.diag([4.0, 4.0, 2500.0, 2500.0, 4.0, 4.0])
            else:
                pred_state, pred_cov = pred_by_id[track_id]
                innovation = z - (_KALMAN_H @ pred_state)
                innovation_cov = _KALMAN_H @ pred_cov @ _KALMAN_H.T + meas_noise
                gain = np.linalg.solve(innovation_cov.T, (_KALMAN_H @ pred_cov)).T
                state = pred_state + (gain @ innovation)
                state[4] = max(float(state[4]), 1.0)
                state[5] = max(float(state[5]), 1.0)
                ikh = eye6 - (gain @ _KALMAN_H)
                cov = ikh @ pred_cov @ ikh.T + gain @ meas_noise @ gain.T
            self._tracks[track_id] = _KalmanTrack(
                track_id=track_id,
                label=region.label,
                box=region.box,
                state=state,
                covariance=cov,
                missed=0,
                last_seen=timestamp,
                last_updated=timestamp,
            )
            output.append(
                Region(region.id, region.box, region.label, track_id, region.detection_score)
            )
        return output


def make_tracker(
    name: str = "iou",
    threshold: float = 0.3,
    max_missed: int = 5,
    max_age_seconds: float = 2.0,
    motion: bool = False,
):
    """Instantiate one of the supported multi-object trackers."""
    normalized = (name or "iou").strip().lower()
    if normalized == "iou":
        return IoUTracker(
            threshold=threshold,
            max_missed=max_missed,
            max_age_seconds=max_age_seconds,
            motion=motion,
        )
    if normalized == "iou-motion":
        return IoUTracker(
            threshold=threshold,
            max_missed=max_missed,
            max_age_seconds=max_age_seconds,
            motion=True,
        )
    if normalized == "kalman":
        return KalmanTracker(
            threshold=threshold,
            max_missed=max_missed,
            max_age_seconds=max_age_seconds,
        )
    raise ValueError(f"unsupported tracker {name!r}; expected 'iou', 'iou-motion', or 'kalman'")

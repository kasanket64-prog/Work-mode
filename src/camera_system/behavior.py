"""Conservative track-window observations in image coordinates."""

from collections import deque
from math import hypot, isfinite


class BehaviorAnalyzer:
    def __init__(self, window=3.0, max_gap=1.5, stationary_fraction=0.01):
        if not all(isfinite(v) and v > 0 for v in (window, max_gap, stationary_fraction)):
            raise ValueError("behavior settings must be positive and finite")
        self.window, self.max_gap, self.stationary_fraction = window, max_gap, stationary_fraction
        self.history = {}
        self.last_time = None

    def update(self, regions, timestamp, size):
        if (
            not isfinite(timestamp)
            or timestamp < 0
            or (self.last_time is not None and timestamp <= self.last_time)
        ):
            raise ValueError("behavior timestamps must increase")
        if len(size) != 2 or any(not isfinite(v) or v <= 0 for v in size):
            raise ValueError("image size must be positive")
        self.last_time = timestamp
        visible = {
            r.track_id: r
            for r in regions
            if r.track_id and r.label in {"person", "car", "bus", "truck", "motorcycle", "bicycle"}
        }
        # Missing observations reset continuity; no inferred entry/exit from disappearance.
        self.history = {key: samples for key, samples in self.history.items() if key in visible}
        result = []
        for key, region in visible.items():
            samples = self.history.setdefault(key, deque(maxlen=256))
            if samples and (timestamp - samples[-1][0] > self.max_gap or samples[-1][3] != size):
                samples.clear()
            x1, y1, x2, y2 = region.box
            samples.append((timestamp, (x1 + x2) / 2, (y1 + y2) / 2, size))
            while samples and timestamp - samples[0][0] > self.window:
                samples.popleft()
            span = timestamp - samples[0][0]
            state = {
                "track_id": key,
                "label": region.label,
                "status": "insufficient_evidence",
                "value": "More continuous observations needed",
                "samples": len(samples),
                "span_seconds": round(span, 3),
            }
            if len(samples) >= 3 and span >= 1:
                diagonal = hypot(*size)
                extent = (
                    hypot(
                        max(s[1] for s in samples) - min(s[1] for s in samples),
                        max(s[2] for s in samples) - min(s[2] for s in samples),
                    )
                    / diagonal
                )
                dx, dy = samples[-1][1] - samples[0][1], samples[-1][2] - samples[0][2]
                direction = (
                    ("right" if dx > 0 else "left")
                    if abs(dx) >= abs(dy)
                    else ("down" if dy > 0 else "up")
                )
                state.update(
                    status="estimated",
                    value="Little image displacement"
                    if extent <= self.stationary_fraction
                    else f"Moving {direction} in image",
                    displacement_fraction=round(extent, 5),
                    net_image_diagonals_per_second=round(hypot(dx, dy) / diagonal / span, 5),
                    limitation="Camera motion, occlusion and ID switches affect this estimate; not physical speed or action recognition.",
                )
            result.append(state)
        return result

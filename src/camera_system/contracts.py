"""Model-independent contracts, also usable by future detector/tracker stages."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class Region:
    id: str
    box: tuple[float, float, float, float]
    label: str | None = None
    track_id: str | None = None
    detection_score: float | None = None


@dataclass(frozen=True)
class Relation:
    subject_id: str
    predicate: str
    object_id: str
    score: float


def parse_regions(data: object, size: tuple[int, int]) -> list[Region]:
    if not isinstance(data, list) or len(data) < 2:
        raise ValueError("regions must be a list containing at least two objects")
    regions = []
    ids = set()
    width, height = size
    for entry in data:
        if not isinstance(entry, dict):
            raise ValueError("each region must be an object")
        if set(entry) - {"id", "box", "label", "track_id", "detection_score"}:
            raise ValueError("unknown region field; use id, box, label, track_id, detection_score")
        rid = entry.get("id")
        if not isinstance(rid, str) or not rid.strip() or rid in ids:
            raise ValueError("region IDs must be nonempty unique strings")
        box = entry.get("box")
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError(f"{rid}: box must be [x1, y1, x2, y2]")
        if any(type(v) not in (int, float) or not isfinite(v) for v in box):
            raise ValueError(f"{rid}: box coordinates must be finite numbers")
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"{rid}: box is empty, inverted, or outside {width}x{height}")
        for field in ("label", "track_id"):
            value = entry.get(field)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{rid}: {field} must be a nonempty string or null")
        score = entry.get("detection_score")
        if score is not None and (
            type(score) not in (int, float) or not isfinite(score) or not 0 <= score <= 1
        ):
            raise ValueError(f"{rid}: detection_score must be in [0, 1] or null")
        ids.add(rid)
        regions.append(
            Region(
                rid, tuple(float(v) for v in box), entry.get("label"), entry.get("track_id"), score
            )
        )
    return regions

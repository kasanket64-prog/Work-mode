"""Timestamped relation evidence and a conservative candidate-event baseline."""

import tomllib
from collections import deque
from dataclasses import dataclass, fields
from math import hypot, isfinite
from pathlib import Path

from .contracts import Region, Relation


@dataclass(frozen=True)
class ReasoningSettings:
    window_seconds: float = 3.0
    min_hits: int = 3
    min_duration_seconds: float = 0.5
    relation_threshold: float = 0.6
    max_gap_seconds: float = 1.5
    unattended_seconds: float = 30.0
    stationary_fraction: float = 0.02
    person_near_fraction: float = 0.15
    zones: tuple = ()
    zone_margin_fraction: float = 0.01
    dwell_seconds: float = 0.0
    pair_distance_change_fraction: float = 0.0

    def __post_init__(self):
        if (
            type(self.pair_distance_change_fraction) not in (int, float)
            or not isfinite(self.pair_distance_change_fraction)
            or not 0 <= self.pair_distance_change_fraction <= 1
        ):
            raise ValueError("pair_distance_change_fraction must be in [0, 1]")
        if (
            type(self.dwell_seconds) not in (int, float)
            or not isfinite(self.dwell_seconds)
            or self.dwell_seconds < 0
        ):
            raise ValueError("dwell_seconds must be finite and nonnegative; zero disables it")
        for name in ("window_seconds", "max_gap_seconds", "unattended_seconds"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("relation_threshold", "stationary_fraction", "person_near_fraction"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if type(self.min_hits) is not int or not 1 <= self.min_hits <= 256:
            raise ValueError("min_hits must be an integer in [1, 256]")
        if (
            not isfinite(self.min_duration_seconds)
            or not 0 <= self.min_duration_seconds <= self.window_seconds
        ):
            raise ValueError("min_duration_seconds must fit inside window_seconds")
        if not isfinite(self.zone_margin_fraction) or not 0 <= self.zone_margin_fraction < 0.5:
            raise ValueError("zone_margin_fraction must be in [0, 0.5)")
        if not isinstance(self.zones, (list, tuple)) or len(self.zones) > 20:
            raise ValueError("zones must contain at most 20 rectangles")
        ids = set()
        for zone in self.zones:
            if not isinstance(zone, dict) or set(zone) != {"id", "box", "labels"}:
                raise ValueError("each zone needs id, box and labels")
            if not isinstance(zone["id"], str) or not zone["id"].strip() or zone["id"] in ids:
                raise ValueError("zone IDs must be nonempty and unique")
            ids.add(zone["id"])
            box = zone["box"]
            if (
                not isinstance(box, (list, tuple))
                or len(box) != 4
                or any(type(v) not in (int, float) or not isfinite(v) for v in box)
            ):
                raise ValueError("zone box needs four finite normalized coordinates")
            x1, y1, x2, y2 = box
            if (
                not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1)
                or min(x2 - x1, y2 - y1) <= 2 * self.zone_margin_fraction
            ):
                raise ValueError(
                    "zone must fit within [0, 1] and be wider than its boundary margin"
                )
            if (
                not isinstance(zone["labels"], list)
                or not zone["labels"]
                or any(not isinstance(v, str) or not v.strip() for v in zone["labels"])
            ):
                raise ValueError("zone labels must be a nonempty list of object class names")


def load_reasoning(path: Path | None) -> ReasoningSettings:
    if path is None:
        return ReasoningSettings()
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    if set(data) - {f.name for f in fields(ReasoningSettings)}:
        raise ValueError("unknown reasoning configuration field")
    return ReasoningSettings(**data)


def center(region):
    x1, y1, x2, y2 = region.box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


class Reasoner:
    def __init__(self, settings: ReasoningSettings):
        self.settings = settings
        self.history = {}
        self.bags = {}
        self.previous_time = None
        self.event_count = 0
        self.zone_states = {}
        self.previous_size = None
        self.dwell_states = {}
        self.pair_states = {}

    def update(self, timestamp: float, regions: list[Region], relations: list[Relation], size):
        cfg = self.settings
        if (
            not isfinite(timestamp)
            or timestamp < 0
            or (self.previous_time is not None and timestamp <= self.previous_time)
        ):
            raise ValueError("reasoning timestamps must be finite, nonnegative and increasing")
        if len(size) != 2 or any(not isfinite(v) or v <= 0 for v in size):
            raise ValueError("image dimensions must be positive and finite")
        if (
            self.previous_time is not None and timestamp - self.previous_time > cfg.max_gap_seconds
        ) or (self.previous_size is not None and size != self.previous_size):
            self.history.clear()
            self.bags.clear()
            self.zone_states.clear()
            self.dwell_states.clear()
            self.pair_states.clear()
        self.previous_size = size
        self.previous_time = timestamp
        visible = {r.track_id: r for r in regions if r.track_id is not None}
        # Missing endpoints are not observations and cannot confirm an event.
        current = {}
        for relation in relations:
            if (
                relation.subject_id in visible
                and relation.object_id in visible
                and isfinite(relation.score)
                and cfg.relation_threshold <= relation.score <= 1
            ):
                key = (relation.subject_id, relation.predicate, relation.object_id)
                current[key] = max(current.get(key, 0), relation.score)
        for key in list(self.history):
            samples = self.history[key]
            while samples and timestamp - samples[0][0] > cfg.window_seconds:
                samples.popleft()
            if not samples or timestamp - samples[-1][0] > cfg.max_gap_seconds:
                del self.history[key]
        for key, score in current.items():
            self.history.setdefault(key, deque(maxlen=256)).append((timestamp, score))
        confirmed = []
        for key in current:
            samples = self.history[key]
            if (
                len(samples) >= cfg.min_hits
                and timestamp - samples[0][0] >= cfg.min_duration_seconds
            ):
                confirmed.append(
                    {
                        "subject_id": key[0],
                        "predicate": key[1],
                        "object_id": key[2],
                        "mean_score": sum(s for _, s in samples) / len(samples),
                        "support_count": len(samples),
                        "span_seconds": timestamp - samples[0][0],
                    }
                )
        diagonal = hypot(*size)
        if diagonal <= 0:
            raise ValueError("image dimensions must be positive")
        bag_regions = {
            key: r for key, r in visible.items() if r.label in {"backpack", "handbag", "suitcase"}
        }
        people = {key: r for key, r in visible.items() if r.label == "person"}
        # Require continuously visible bags; occlusion clears timers and prior association.
        self.bags = {key: state for key, state in self.bags.items() if key in bag_regions}
        events = []
        pairs = {}
        if cfg.pair_distance_change_fraction:
            vehicles = {
                k: r
                for k, r in visible.items()
                if r.label in {"car", "truck", "bus", "motorcycle", "bicycle"}
            }
            # ponytail: all visible pairs; spatial indexing if object counts grow substantially.
            for person_id, person in people.items():
                for vehicle_id, vehicle in vehicles.items():
                    key = (person_id, vehicle_id, vehicle.label)
                    distance = (
                        hypot(
                            center(person)[0] - center(vehicle)[0],
                            center(person)[1] - center(vehicle)[1],
                        )
                        / diagonal
                    )
                    state = self.pair_states.get(
                        key, {"samples": deque(maxlen=256), "emitted": None}
                    )
                    samples = state["samples"]
                    while samples and timestamp - samples[0][0] > cfg.window_seconds:
                        samples.popleft()
                    samples.append((timestamp, distance))
                    pairs[key] = state
                    if (
                        len(samples) < max(3, cfg.min_hits)
                        or timestamp - samples[0][0] < cfg.min_duration_seconds
                    ):
                        continue
                    change = distance - samples[0][1]
                    direction = "closing" if change < 0 else "separating"
                    deltas = [b[1] - a[1] for a, b in zip(samples, list(samples)[1:])]
                    consistent = (
                        all(d <= 0 for d in deltas) if change < 0 else all(d >= 0 for d in deltas)
                    )
                    if (
                        consistent
                        and abs(change) >= cfg.pair_distance_change_fraction
                        and min(distance, samples[0][1]) <= cfg.person_near_fraction
                        and state["emitted"] != direction
                    ):
                        self.event_count += 1
                        events.append(
                            {
                                "event_id": f"event-{self.event_count}",
                                "type": f"person_vehicle_{direction}_candidate",
                                "timestamp_seconds": timestamp,
                                "object_track_id": vehicle_id,
                                "subject_track_id": person_id,
                                "started_seconds": samples[0][0],
                                "observed_duration_seconds": timestamp - samples[0][0],
                                "support_count": len(samples),
                                "distance_change_fraction": change,
                                "start_distance_fraction": samples[0][1],
                                "end_distance_fraction": distance,
                                "basis": "monotonic_image_center_distance_change",
                                "limitation": "Relative image motion only; either object or the camera may be moving. Does not establish following or intent.",
                            }
                        )
                        state["emitted"] = direction
        self.pair_states = pairs
        self.dwell_states = {
            key: state for key, state in self.dwell_states.items() if key in visible
        }
        if cfg.dwell_seconds:
            for key, region in visible.items():
                if region.label not in {"person", "car", "truck", "bus", "motorcycle", "bicycle"}:
                    self.dwell_states.pop(key, None)
                    continue
                location = center(region)
                state = self.dwell_states.get(key)
                if (
                    state is None
                    or state["label"] != region.label
                    or hypot(location[0] - state["anchor"][0], location[1] - state["anchor"][1])
                    / diagonal
                    > cfg.stationary_fraction
                ):
                    state = {
                        "anchor": location,
                        "since": timestamp,
                        "label": region.label,
                        "hits": 0,
                        "emitted": False,
                    }
                    self.dwell_states[key] = state
                state["hits"] += 1
                if (
                    not state["emitted"]
                    and state["hits"] >= cfg.min_hits
                    and timestamp - state["since"] >= cfg.dwell_seconds
                ):
                    self.event_count += 1
                    events.append(
                        {
                            "event_id": f"event-{self.event_count}",
                            "type": "person_dwell_candidate"
                            if region.label == "person"
                            else "vehicle_dwell_candidate",
                            "timestamp_seconds": timestamp,
                            "object_track_id": key,
                            "object_label": region.label,
                            "started_seconds": state["since"],
                            "observed_duration_seconds": timestamp - state["since"],
                            "support_count": state["hits"],
                            "evidence_box": list(region.box),
                            "basis": "continuous_track_center_stayed_within_image_radius",
                            "limitation": "Image displacement only; camera motion and tracking errors affect this candidate. Does not establish intent, standing, or parking.",
                        }
                    )
                    state["emitted"] = True
        for key, bag in bag_regions.items():
            location = center(bag)
            state = self.bags.setdefault(
                key,
                {
                    "anchor": location,
                    "since": timestamp,
                    "owner": None,
                    "emitted": False,
                    "away_since": None,
                },
            )
            associations = [
                r
                for r in confirmed
                if r["object_id"] == key
                and r["subject_id"] in people
                and r["predicate"] in {"holding", "carrying"}
            ]
            if associations:
                state["owner"] = max(associations, key=lambda r: r["mean_score"])["subject_id"]
            moved = (
                hypot(location[0] - state["anchor"][0], location[1] - state["anchor"][1]) / diagonal
                > cfg.stationary_fraction
            )
            nearby = any(
                hypot(center(p)[0] - location[0], center(p)[1] - location[1]) / diagonal
                <= cfg.person_near_fraction
                for p in people.values()
            )
            if moved or nearby or associations or state["owner"] is None:
                state.update(anchor=location, since=timestamp, emitted=False, away_since=None)
            elif state["away_since"] is None:
                state["away_since"] = timestamp
            elif timestamp - state["away_since"] >= cfg.unattended_seconds and not state["emitted"]:
                self.event_count += 1
                events.append(
                    {
                        "event_id": f"event-{self.event_count}",
                        "type": "possible_unattended_bag",
                        "timestamp_seconds": timestamp,
                        "object_track_id": key,
                        "previous_person_track_id": state["owner"],
                        "observed_unattended_seconds": timestamp - state["away_since"],
                        "basis": "prior_confirmed_holding_or_carrying; stationary_bag; no_nearby_person",
                    }
                )
                state["emitted"] = True
        self.zone_states = {
            key: state for key, state in self.zone_states.items() if key[1] in visible
        }
        for zone in cfg.zones:
            for key, region in visible.items():
                if region.label not in zone["labels"]:
                    self.zone_states.pop((zone["id"], key), None)
                    continue
                # Bottom-center is an image anchor, not a calibrated ground position.
                x = (region.box[0] + region.box[2]) / (2 * size[0])
                y = region.box[3] / size[1]
                x1, y1, x2, y2 = zone["box"]
                margin = cfg.zone_margin_fraction
                inside = x1 + margin <= x <= x2 - margin and y1 + margin <= y <= y2 - margin
                outside = x < x1 - margin or x > x2 + margin or y < y1 - margin or y > y2 + margin
                state = self.zone_states.setdefault(
                    (zone["id"], key),
                    {"stable": None, "pending": None, "hits": 0, "since": timestamp},
                )
                if not inside and not outside:
                    state.update(pending=None, hits=0)
                    continue
                if state["pending"] != inside:
                    state.update(pending=inside, hits=0, since=timestamp)
                state["hits"] += 1
                if (
                    state["hits"] < cfg.min_hits
                    or timestamp - state["since"] < cfg.min_duration_seconds
                ):
                    continue
                if state["stable"] is not None and state["stable"] != inside:
                    self.event_count += 1
                    events.append(
                        {
                            "event_id": f"event-{self.event_count}",
                            "type": "zone_entry" if inside else "zone_exit",
                            "timestamp_seconds": timestamp,
                            "object_track_id": key,
                            "zone_id": zone["id"],
                            "transition_started_seconds": state["since"],
                            "support_count": state["hits"],
                            "basis": "continuous_track_bottom_center_crossed_image_zone",
                            "limitation": "Image-zone transition only; camera motion and tracking errors can cause false events.",
                        }
                    )
                state["stable"] = inside
        return confirmed, events

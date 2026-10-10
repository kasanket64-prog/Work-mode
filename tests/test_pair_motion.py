import pytest

from camera_system.contracts import Region
from camera_system.reasoning import Reasoner, ReasoningSettings
from camera_system.review import evaluate


def scene(x):
    return [
        Region("p", (x, 20, x + 10, 40), "person", "p1"),
        Region("v", (70, 20, 90, 40), "car", "v1"),
    ]


def test_pair_closing_and_evaluation_require_both_tracks():
    engine = Reasoner(ReasoningSettings(pair_distance_change_fraction=0.03))
    assert not engine.update(0, scene(45), [], (100, 100))[1]
    assert not engine.update(0.5, scene(55), [], (100, 100))[1]
    events = engine.update(1, scene(65), [], (100, 100))[1]
    assert len(events) == 1
    event = events[0]
    assert event["type"] == "person_vehicle_closing_candidate"
    assert event["subject_track_id"] == "p1" and event["object_track_id"] == "v1"
    assert not engine.update(1.5, scene(70), [], (100, 100))[1]
    result = evaluate(
        [
            {"type": "frame", "timestamp_seconds": 0},
            {"type": "frame", "timestamp_seconds": 1, "events": events},
            {"type": "frame", "timestamp_seconds": 2},
        ],
        {"duration_seconds": 2, "events": [dict(event, subject_track_id="other")]},
    )
    assert result["events"]["true_positive"] == 0
    engine.update(2, [], [], (100, 100))
    assert not engine.update(2.5, scene(65), [], (100, 100))[1]


@pytest.mark.parametrize("positions,expected", [([65, 55, 45], True), ([45, 65, 60], False)])
def test_separation_and_reversal(positions, expected):
    engine = Reasoner(ReasoningSettings(pair_distance_change_fraction=0.03))
    events = []
    for i, x in enumerate(positions):
        events.extend(engine.update(i / 2, scene(x), [], (100, 100))[1])
    assert bool(events) == expected
    if expected:
        assert events[0]["type"] == "person_vehicle_separating_candidate"

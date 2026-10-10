import pytest

from camera_system.contracts import Region
from camera_system.reasoning import Reasoner, ReasoningSettings
from camera_system.replay import replay
from camera_system.review import evaluate

ZONE = {"id": "area", "box": [0.3, 0.2, 0.7, 0.8], "labels": ["person"]}


def person(x):
    return [Region("p", (x - 5, 20, x + 5, 50), "person", "track-1")]


def test_zone_confirmation_jitter_exit_and_evaluation_identity():
    engine = Reasoner(ReasoningSettings(zones=[ZONE]))
    for t in (0, 0.3, 0.6):
        assert not engine.update(t, person(10), [], (100, 100))[1]
    assert not engine.update(0.9, person(50), [], (100, 100))[1]
    assert not engine.update(1.2, person(30), [], (100, 100))[1]  # margin resets pending
    for t in (1.5, 1.8):
        assert not engine.update(t, person(50), [], (100, 100))[1]
    entry = engine.update(2.1, person(50), [], (100, 100))[1]
    assert len(entry) == 1 and entry[0]["type"] == "zone_entry"
    assert not engine.update(2.4, person(50), [], (100, 100))[1]
    for t in (2.7, 3):
        assert not engine.update(t, person(90), [], (100, 100))[1]
    exit_events = engine.update(3.3, person(90), [], (100, 100))[1]
    assert len(exit_events) == 1 and exit_events[0]["type"] == "zone_exit"
    rows = [
        {"type": "frame", "timestamp_seconds": 0},
        {"type": "frame", "timestamp_seconds": 2.1, "events": entry},
        {"type": "frame", "timestamp_seconds": 3},
    ]
    result = evaluate(rows, {"duration_seconds": 3, "events": [dict(entry[0], zone_id="other")]})
    assert result["events"]["true_positive"] == 0


@pytest.mark.parametrize("reset", ["missing", "gap", "resolution"])
def test_discontinuity_and_first_sighting_do_not_emit(reset):
    engine = Reasoner(ReasoningSettings(zones=[ZONE], min_hits=1, min_duration_seconds=0))
    assert not engine.update(0, person(10), [], (100, 100))[1]
    if reset == "missing":
        engine.update(0.1, [], [], (100, 100))
    assert not engine.update(
        3 if reset == "gap" else 0.2,
        person(50),
        [],
        (101, 100) if reset == "resolution" else (100, 100),
    )[1]


@pytest.mark.parametrize(
    "zone",
    [dict(ZONE, box=[0, 0, 2, 1]), dict(ZONE, labels=[]), dict(ZONE, box=[0.3, 0.2, 0.31, 0.8])],
)
def test_invalid_zone_configuration(zone):
    with pytest.raises(ValueError):
        ReasoningSettings(zones=[zone])


def test_replay_preserves_source_and_recomputes_events():
    from dataclasses import asdict

    rows = [{"type": "session", "relation_endpoint_kind": "track_id"}]
    rows.extend(
        {
            "type": "frame",
            "timestamp_seconds": t,
            "regions": [asdict(r) for r in person(x)],
            "image_size": [100, 100],
            "relations": [],
            "events": [{"old": True}],
        }
        for t, x in [(0, 10), (0.5, 50)]
    )
    replayed = list(
        replay(rows, ReasoningSettings(zones=[ZONE], min_hits=1, min_duration_seconds=0))
    )
    assert replayed[0]["reasoning_replayed"]
    assert replayed[-1]["events"][0]["type"] == "zone_entry"
    assert rows[-1]["events"] == [{"old": True}]

import pytest

from camera_system.contracts import Region, Relation
from camera_system.reasoning import Reasoner, ReasoningSettings
from camera_system.review import HTML, evaluate


def scene(person=True):
    bag = Region("b", (10, 10, 20, 20), "backpack", "bag")
    owner = Region("p", (10, 10, 30, 50), "person", "person")
    return [bag, owner] if person else [bag]


def test_temporal_support_event_and_deduplication():
    reasoner = Reasoner(ReasoningSettings(unattended_seconds=2))
    relation = Relation("person", "holding", "bag", 0.9)
    assert reasoner.update(0, scene(), [relation], (100, 100)) == ([], [])
    assert reasoner.update(0.3, scene(), [relation], (100, 100)) == ([], [])
    confirmed, events = reasoner.update(0.6, scene(), [relation], (100, 100))
    assert len(confirmed) == 1 and not events
    assert reasoner.update(1.6, scene(False), [], (100, 100)) == ([], [])
    assert not reasoner.update(2.7, scene(False), [], (100, 100))[1]
    _, events = reasoner.update(3.7, scene(False), [], (100, 100))
    assert len(events) == 1
    assert events[0]["type"] == "possible_unattended_bag"
    assert not reasoner.update(4, scene(False), [], (100, 100))[1]


@pytest.mark.parametrize("interruption", ["gap", "occlusion", "movement"])
def test_discontinuity_prevents_alert(interruption):
    reasoner = Reasoner(ReasoningSettings(min_hits=1, min_duration_seconds=0, unattended_seconds=1))
    relation = Relation("person", "carrying", "bag", 0.8)
    reasoner.update(0, scene(), [relation], (100, 100))
    if interruption == "gap":
        assert not reasoner.update(10, scene(False), [], (100, 100))[1]
    elif interruption == "occlusion":
        reasoner.update(0.5, [], [], (100, 100))
        assert not reasoner.update(1.1, scene(False), [], (100, 100))[1]
    else:
        moved = [Region("b", (70, 70, 80, 80), "backpack", "bag")]
        assert not reasoner.update(1.1, moved, [], (100, 100))[1]


def test_no_owner_no_event_and_monotonic_time():
    reasoner = Reasoner(ReasoningSettings(unattended_seconds=0.2))
    reasoner.update(0, scene(False), [], (100, 100))
    assert not reasoner.update(1, scene(False), [], (100, 100))[1]
    with pytest.raises(ValueError):
        reasoner.update(1, scene(False), [], (100, 100))


def test_evaluation_one_to_one_mapping_and_relation_metrics():
    event = {
        "type": "possible_unattended_bag",
        "object_track_id": "track-1",
        "timestamp_seconds": 5,
    }
    rows = [
        {"type": "frame", "frame_index": 0, "timestamp_seconds": 0},
        {
            "type": "frame",
            "frame_index": 1,
            "timestamp_seconds": 5,
            "events": [event, event],
            "confirmed_relations": [
                {"subject_id": "p", "predicate": "holding", "object_id": "track-1"}
            ],
        }
    ]
    rows.append({"type": "frame", "frame_index": 2, "timestamp_seconds": 10})
    truth = {
        "duration_seconds": 10,
        "track_mapping": {"track-1": "bag"},
        "events": [dict(event, object_track_id="bag")],
        "annotated_frame_indices": [1],
        "relations": [
            {"frame_index": 1, "subject_id": "p", "predicate": "holding", "object_id": "bag"}
        ],
    }
    result = evaluate(rows, truth)
    assert result["events"]["true_positive"] == 1
    assert result["events"]["false_positive"] == 1
    assert result["false_alarms_per_hour"] == 360
    assert result["confirmed_relations"]["f1"] == 1
    assert "textContent" in HTML


def test_expired_evidence_does_not_confirm():
    reasoner = Reasoner(ReasoningSettings())
    relation = Relation("person", "holding", "bag", 0.9)
    reasoner.update(0, scene(), [relation], (100, 100))
    reasoner.update(0.3, scene(), [relation], (100, 100))
    assert not reasoner.update(4, scene(), [relation], (100, 100))[0]

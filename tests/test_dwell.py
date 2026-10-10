import pytest

from camera_system.contracts import Region
from camera_system.reasoning import Reasoner, ReasoningSettings


@pytest.mark.parametrize("label", ["person", "car"])
def test_dwell_confirmation_reset_and_no_duplicate(label):
    engine = Reasoner(ReasoningSettings(dwell_seconds=1))
    region = Region("r", (10, 10, 20, 30), label, "track-1")
    for t in (0, 0.5):
        assert not engine.update(t, [region], [], (100, 100))[1]
    events = engine.update(1, [region], [], (100, 100))[1]
    assert len(events) == 1 and events[0]["observed_duration_seconds"] == 1
    assert events[0]["type"] == (
        "person_dwell_candidate" if label == "person" else "vehicle_dwell_candidate"
    )
    assert not engine.update(1.5, [region], [], (100, 100))[1]
    engine.update(2, [], [], (100, 100))
    assert not engine.update(2.5, [region], [], (100, 100))[1]
    assert not engine.update(5, [region], [], (100, 100))[1]
    assert not engine.update(5.5, [region], [], (200, 200))[1]


def test_motion_resets_dwell_and_invalid_duration():
    engine = Reasoner(ReasoningSettings(dwell_seconds=1))
    for t in (0, 0.5, 1, 1.5):
        region = Region("r", (t * 20, 10, t * 20 + 10, 30), "car", "track-1")
        assert not engine.update(t, [region], [], (100, 100))[1]
    for duration in (-1, float("nan"), True):
        with pytest.raises(ValueError):
            ReasoningSettings(dwell_seconds=duration)

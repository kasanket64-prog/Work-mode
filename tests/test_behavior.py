import pytest

from camera_system.behavior import BehaviorAnalyzer
from camera_system.contracts import Region


def test_continuity_motion_and_resolution_reset():
    analyzer = BehaviorAnalyzer()

    def observation(t, x=0, size=(100, 100)):
        return analyzer.update([Region("a", (x, 0, x + 10, 20), "car", "track-1")], t, size)[0]

    assert observation(0)["status"] == "insufficient_evidence"
    observation(0.5)
    assert observation(1)["value"] == "Little image displacement"
    assert observation(1.5, 30)["value"] == "Moving right in image"
    assert observation(2, 30, (200, 100))["status"] == "insufficient_evidence"
    analyzer.update([], 2.5, (200, 100))
    assert observation(3)["samples"] == 1
    assert observation(7)["samples"] == 1
    with pytest.raises(ValueError):
        observation(7)

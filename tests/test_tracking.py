import pytest

from camera_system.contracts import Region
from camera_system.tracking import IoUTracker, iou


def region(rid, box, label="person"):
    return Region(rid, box, label)


def test_iou_and_stable_ids():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1
    tracker = IoUTracker(threshold=0.2, max_missed=1)
    first = tracker.update([region("a", (0, 0, 10, 10))])
    second = tracker.update([region("b", (1, 0, 11, 10))])
    assert first[0].track_id == second[0].track_id == "track-1"
    assert second[0].id == "b"


def test_labels_prevent_cross_class_match():
    tracker = IoUTracker(threshold=0.1)
    person = tracker.update([region("p", (0, 0, 10, 10), "person")])[0]
    horse = tracker.update([region("h", (0, 0, 10, 10), "horse")])[0]
    assert person.track_id != horse.track_id


def test_missed_expiry_creates_new_id():
    tracker = IoUTracker(threshold=0.2, max_missed=1)
    first = tracker.update([region("a", (0, 0, 10, 10))])[0].track_id
    assert tracker.update([]) == []
    assert tracker.update([]) == []
    current = tracker.update([region("b", (0, 0, 10, 10))])[0].track_id
    assert current != first


@pytest.mark.parametrize("kwargs", [dict(threshold=-0.1), dict(threshold=1.1), dict(max_missed=-1)])
def test_bad_tracker_config(kwargs):
    with pytest.raises(ValueError):
        IoUTracker(**kwargs)


def test_reordered_detections_occlusion_and_expiry():
    tracker = IoUTracker(max_age_seconds=2)
    left = region("a", (0, 0, 10, 10))
    right = region("b", (50, 0, 60, 10))
    first = tracker.update([left, right], 0)
    second = tracker.update([right, left], 0.5)
    assert second[0].track_id == first[1].track_id
    assert second[1].track_id == first[0].track_id
    tracker.update([], 1)
    assert tracker.update([left], 1.5)[0].track_id == first[0].track_id
    assert tracker.update([left], 10)[0].track_id != first[0].track_id


def test_zero_threshold_and_backward_time_rejected():
    with pytest.raises(ValueError):
        IoUTracker(threshold=0)
    tracker = IoUTracker()
    tracker.update([], 2)
    with pytest.raises(ValueError):
        tracker.update([], 1)


def test_motion_bridges_gap_using_elapsed_time_and_keeps_observed_boxes():
    tracker = IoUTracker(motion=True)
    first = tracker.update([region("a", (0, 0, 10, 10))], 0)[0]
    tracker.update([region("b", (4, 0, 14, 10))], 0.5)
    tracker.update([], 1.0)
    observed = (12, 0, 22, 10)
    result = tracker.update([region("c", observed)], 1.5)[0]
    assert result.track_id == first.track_id
    assert result.box == observed
    assert tracker.update([region("d", (40, 0, 50, 10))], 5)[0].track_id != first.track_id


def test_motion_keeps_ids_for_two_objects_crossing_between_observations():
    tracker = IoUTracker(motion=True)
    first = tracker.update([region("a", (0, 0, 10, 10)), region("b", (20, 0, 30, 10))], 0)
    tracker.update([region("a", (4, 0, 14, 10)), region("b", (16, 0, 26, 10))], 0.5)
    result = tracker.update([region("b", (4, 0, 14, 10)), region("a", (16, 0, 26, 10))], 2)
    assert result[0].track_id == first[1].track_id
    assert result[1].track_id == first[0].track_id

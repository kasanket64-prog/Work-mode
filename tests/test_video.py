import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from camera_system.contracts import Region, Relation
from camera_system.video import FrameSource
from camera_system.video_cli import main


def fake_cv(monkeypatch, count=4, fps=10.0, pts=None):
    capture = Mock()
    capture.isOpened.return_value = True
    state = {"index": -1}

    def read():
        state["index"] += 1
        if state["index"] >= count:
            return False, None
        return True, np.array([[[10, 20, 30]]], dtype=np.uint8)

    def get(prop):
        if prop == 1:
            return fps
        return (pts or [100 * i for i in range(count)])[state["index"]]

    capture.read.side_effect = read
    capture.get.side_effect = get
    cv = SimpleNamespace(
        VideoCapture=Mock(return_value=capture),
        CAP_PROP_FPS=1,
        CAP_PROP_POS_MSEC=2,
        COLOR_BGR2RGB=3,
        cvtColor=lambda image, _: image[:, :, ::-1].copy(),
    )
    monkeypatch.setitem(sys.modules, "cv2", cv)
    return capture


def test_source_stride_color_timestamps_and_release(tmp_path, monkeypatch):
    path = tmp_path / "video.avi"
    path.touch()
    capture = fake_cv(monkeypatch, count=5)
    with FrameSource(str(path), 2) as source:
        frames = list(source)
    assert [f.index for f in frames] == [0, 2, 4]
    assert [f.timestamp_seconds for f in frames] == [0, 0.2, 0.4]
    assert frames[0].image.getpixel((0, 0)) == (30, 20, 10)
    assert source.frames_skipped == 2
    capture.release.assert_called_once()


def test_timestamp_fallback(tmp_path, monkeypatch):
    path = tmp_path / "video.avi"
    path.touch()
    fake_cv(monkeypatch, count=3, pts=[0, 0, float("nan")])
    with FrameSource(str(path)) as source:
        frames = list(source)
    assert [f.timestamp_seconds for f in frames] == [0, 0.1, 0.2]
    assert frames[1].timestamp_source == "estimated_from_fps"


def test_invalid_timestamps_release(tmp_path, monkeypatch):
    path = tmp_path / "video.avi"
    path.touch()
    capture = fake_cv(monkeypatch, count=2, fps=0, pts=[0, 0])
    with pytest.raises(RuntimeError, match="FPS"):
        with FrameSource(str(path)) as source:
            list(source)
    capture.release.assert_called_once()


def test_camera_failure_releases(monkeypatch):
    capture = fake_cv(monkeypatch, count=1)
    with pytest.raises(RuntimeError, match="camera stopped"):
        with FrameSource(0) as source:
            frames = iter(source)
            frame = next(frames)
            assert frame.timestamp_seconds >= 0
            assert frame.timestamp_source == "host_monotonic_after_read"
            next(frames)
    capture.release.assert_called_once()


@pytest.mark.parametrize("interrupt", [False, True])
@pytest.mark.parametrize("relation_every", [1, 2])
def test_runner_reuses_models_and_preserves_partial_output(
    tmp_path, monkeypatch, interrupt, relation_every
):
    path = tmp_path / "video.avi"
    path.touch()
    config = tmp_path / "config.toml"
    config.write_text('device = "cpu"')
    det_config = tmp_path / "det.toml"
    det_config.write_text("max_detections = 2")
    output = tmp_path / "result.jsonl"
    capture = fake_cv(monkeypatch)
    detector = Mock()
    detector.return_value.device = "cpu"
    detector.return_value.weights_sha256 = "hash"
    regions = [Region("0", (0, 0, 1, 1)), Region("1", (0, 0, 1, 1))]
    detector.return_value.detect.side_effect = [regions, KeyboardInterrupt()] if interrupt else None
    detector.return_value.detect.return_value = regions
    predictor = Mock()
    predictor.return_value.device = "cpu"
    predictor.return_value.predict.return_value = [Relation("0", "beside", "1", 0.9)]
    predictor.return_value.rejected_relations = []
    monkeypatch.setattr("camera_system.video_cli.YoloDetector", detector)
    monkeypatch.setattr("camera_system.video_cli.RelateAnythingPredictor", predictor)
    args = [
        "--video",
        str(path),
        "--config",
        str(config),
        "--detector-config",
        str(det_config),
        "--output",
        str(output),
        "--max-frames",
        "2",
        "--relation-every",
        str(relation_every),
    ]
    assert main(args) == (130 if interrupt else 0)
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert records[0]["type"] == "session"
    assert records[-1]["frames_processed"] == (1 if interrupt else 2)
    assert records[-1]["status"] == ("interrupted" if interrupt else "completed")
    frames = [r for r in records if r["type"] == "frame"]
    assert records[0]["relation_every"] == relation_every
    assert predictor.return_value.predict.call_count == sum(
        i % relation_every == 0 for i in range(len(frames))
    )
    for i, frame in enumerate(frames):
        assert len(frame["regions"]) == 2
        if i % relation_every:
            assert frame["relation_skip_reason"] == "scheduled_skip"
            assert frame["relations"] == []
            assert frame["confirmed_relations"] == []
        else:
            assert frame["relation_skip_reason"] is None
            assert len(frame["relations"]) == 1
    detector.assert_called_once()
    predictor.assert_called_once()
    capture.release.assert_called_once()
    original = output.read_bytes()
    assert main(args) == 2
    assert output.read_bytes() == original


def test_unopened_capture_is_released(monkeypatch):
    capture = fake_cv(monkeypatch)
    capture.isOpened.return_value = False
    with pytest.raises(RuntimeError, match="cannot open"):
        with FrameSource(0):
            pass
    capture.release.assert_called_once()


def test_network_stream_timeout_clock_and_failure(monkeypatch):
    capture = fake_cv(monkeypatch, count=1)
    cv = sys.modules["cv2"]
    cv.CAP_FFMPEG = 10
    cv.CAP_PROP_OPEN_TIMEOUT_MSEC = 11
    cv.CAP_PROP_READ_TIMEOUT_MSEC = 12
    with FrameSource("https://example.com/live.m3u8", network=True) as source:
        frames = iter(source)
        assert next(frames).timestamp_source == "host_monotonic_after_read"
        with pytest.raises(RuntimeError, match="stream ended"):
            next(frames)
        assert source.end_reason == "stream_read_failure"
    cv.VideoCapture.assert_called_once_with(
        "https://example.com/live.m3u8", 10, [11, 10000, 12, 10000]
    )
    capture.release.assert_called_once()
    for url in ["file:///secret", "https://user:password@example.com/live", "https:///missing"]:
        with pytest.raises(ValueError):
            FrameSource(url, network=True)

"""Tests for Tier 3-4 live FrameSource resilience, KalmanTracker, and calibration benchmark."""

import json
import sys
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from camera_system.benchmark import main as benchmark_main
from camera_system.benchmark import run_benchmark
from camera_system.contracts import Region
from camera_system.demo import make_demo
from camera_system.reasoning import ReasoningSettings
from camera_system.tracking import IoUTracker, KalmanTracker, make_tracker
from camera_system.video import FrameSource
from camera_system.video_cli import main as video_main


def _make_fake_cv2(capture_factory):
    return SimpleNamespace(
        CAP_FFMPEG=1900,
        CAP_PROP_OPEN_TIMEOUT_MSEC=53,
        CAP_PROP_READ_TIMEOUT_MSEC=54,
        CAP_PROP_FPS=5,
        CAP_PROP_POS_MSEC=0,
        COLOR_BGR2RGB=4,
        cvtColor=lambda img, _code: img[:, :, ::-1],
        VideoCapture=capture_factory,
    )


def test_frame_source_credentials_env_not_leaked(monkeypatch):
    opened_sources = []

    class FakeCapture:
        def __init__(self, target, *_args):
            opened_sources.append(target)

        def isOpened(self):
            return True

        def get(self, _prop):
            return 30.0

        def read(self):
            return False, None

        def release(self):
            pass

    monkeypatch.setitem(sys.modules, "cv2", _make_fake_cv2(FakeCapture))
    monkeypatch.setenv("CAM_CREDS", "operator:s3cr3t-pass!")
    src = FrameSource("rtsp://camera.local/live", network=True, credentials_env="CAM_CREDS")
    assert "s3cr3t-pass!" not in src.source
    assert "operator" not in src.source
    with src:
        pass
    assert opened_sources == ["rtsp://operator:s3cr3t-pass%21@camera.local/live"]


def test_frame_source_credentials_validation(monkeypatch):
    monkeypatch.delenv("MISSING_CREDS", raising=False)
    with pytest.raises(ValueError, match="username:password"):
        FrameSource("rtsp://camera.local/live", network=True, credentials_env="MISSING_CREDS")
    with pytest.raises(ValueError, match="without embedded credentials"):
        FrameSource("rtsp://user:pass@camera.local/live", network=True)


def test_frame_source_threaded_live_buffer_drops_stale_frames(tmp_path, monkeypatch):
    clip = tmp_path / "live.mp4"
    clip.write_bytes(b"fake")
    four_read = Event()
    allow_eof = Event()

    class FastCapture:
        def __init__(self, *_args):
            self.calls = 0

        def isOpened(self):
            return True

        def get(self, prop):
            if prop == 5:  # FPS
                return 10.0
            return float(self.calls * 100)

        def read(self):
            idx = self.calls
            self.calls += 1
            if idx < 4:
                frame = np.full((8, 8, 3), idx, dtype=np.uint8)
                if idx == 3:
                    four_read.set()
                return True, frame
            allow_eof.wait(timeout=2)
            return False, None

        def release(self):
            allow_eof.set()

    monkeypatch.setitem(sys.modules, "cv2", _make_fake_cv2(FastCapture))
    src = FrameSource(str(clip), stride=1, threaded=True)
    with src:
        assert four_read.wait(timeout=2)
        it = iter(src)
        latest = next(it)
        assert latest.index == 3
        assert src.dropped_frames == 3
        allow_eof.set()
        assert list(it) == []
    assert src.end_reason == "video_eof_or_decode_failure"


def test_frame_source_reconnect_recovery(monkeypatch):
    attempt = {"n": 0}
    release_calls = {"n": 0}

    class FlakyStreamCapture:
        def __init__(self, *_args):
            self.id = attempt["n"]
            attempt["n"] += 1
            self.step = 0

        def isOpened(self):
            return True

        def get(self, _prop):
            return 15.0

        def read(self):
            self.step += 1
            if self.id == 0:
                if self.step == 1:
                    return True, np.zeros((6, 6, 3), dtype=np.uint8)
                return False, None
            if self.step == 1:
                return True, np.ones((6, 6, 3), dtype=np.uint8)
            return False, None

        def release(self):
            release_calls["n"] += 1

    monkeypatch.setitem(sys.modules, "cv2", _make_fake_cv2(FlakyStreamCapture))
    src = FrameSource(
        "http://stream.local/feed.mjpg",
        network=True,
        max_reconnects=1,
        reconnect_delay=0.01,
    )
    with src:
        it = iter(src)
        f0 = next(it)
        f1 = next(it)
        with pytest.raises(RuntimeError, match="stream ended"):
            next(it)
    assert f0.index == 0 and f1.index == 1
    assert src.reconnects == 1
    assert release_calls["n"] >= 2


def test_kalman_tracker_bridges_occlusion_and_preserves_boxes():
    kalman = make_tracker("kalman", threshold=0.3, max_missed=2, max_age_seconds=5.0)
    iou_plain = make_tracker("iou", threshold=0.3, max_missed=2, max_age_seconds=5.0)

    # Two overlapping observations establish vx=+20px/s, then t=1.5 reappears past static overlap.
    seq = [
        (0.0, [Region("r0", (10.0, 10.0, 30.0, 50.0), "person")]),
        (0.5, [Region("r1", (20.0, 10.0, 40.0, 50.0), "person")]),
        (1.0, []),  # Brief occlusion
        (1.5, [Region("r3", (40.0, 10.0, 60.0, 50.0), "person")]),
    ]
    k_ids = []
    i_ids = []
    for ts, regs in seq:
        k_out = kalman.update(regs, timestamp=ts)
        i_out = iou_plain.update(regs, timestamp=ts)
        if k_out:
            k_ids.append(k_out[0].track_id)
            assert k_out[0].box == regs[0].box
        if i_out:
            i_ids.append(i_out[0].track_id)

    assert k_ids == ["track-1", "track-1", "track-1"]
    assert i_ids == ["track-1", "track-1", "track-2"]


def test_kalman_tracker_crossing_tracks_and_factory_validation():
    tracker = KalmanTracker(threshold=0.2, max_missed=2, max_age_seconds=5.0)
    # Two objects moving in opposite directions with overlapping init and brief occlusion.
    f0 = tracker.update(
        [
            Region("a", (0.0, 0.0, 20.0, 20.0), "car"),
            Region("b", (80.0, 40.0, 100.0, 60.0), "person"),
        ],
        timestamp=0.0,
    )
    f1 = tracker.update(
        [
            Region("a", (10.0, 0.0, 30.0, 20.0), "car"),
            Region("b", (70.0, 40.0, 90.0, 60.0), "person"),
        ],
        timestamp=0.5,
    )
    f2 = tracker.update(
        [
            Region("b", (60.0, 40.0, 80.0, 60.0), "person"),
            Region("a", (20.0, 0.0, 40.0, 20.0), "car"),
        ],
        timestamp=1.0,
    )
    assert tracker.update([], timestamp=1.5) == []
    f3 = tracker.update(
        [
            Region("b", (40.0, 40.0, 60.0, 60.0), "person"),
            Region("a", (40.0, 0.0, 60.0, 20.0), "car"),
        ],
        timestamp=2.0,
    )
    assert [r.track_id for r in f0] == ["track-1", "track-2"]
    assert [r.track_id for r in f1] == ["track-1", "track-2"]
    assert [r.track_id for r in f2] == ["track-2", "track-1"]
    assert [r.track_id for r in f3] == ["track-2", "track-1"]

    assert isinstance(make_tracker("iou-motion"), IoUTracker)
    with pytest.raises(ValueError, match="unsupported tracker"):
        make_tracker("unknown")


def test_video_cli_accepts_tracker_kalman(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    output = tmp_path / "session.jsonl"

    class StubFrameSource:
        def __init__(self, *_a, **_kw):
            self.fps = 1.0
            self.frames_read = 0
            self.frames_skipped = 0
            self.dropped_frames = None
            self.reconnects = 0
            self.end_reason = "video_eof_or_decode_failure"

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            pass

        def __iter__(self):
            from PIL import Image

            self.frames_read = 2
            img = Image.new("RGB", (64, 64))
            yield SimpleNamespace(index=0, timestamp_seconds=0.0, timestamp_source="fps", image=img)
            yield SimpleNamespace(index=1, timestamp_seconds=1.0, timestamp_source="fps", image=img)

    monkeypatch.setattr("camera_system.video_cli.FrameSource", StubFrameSource)
    monkeypatch.setattr(
        "camera_system.video_cli.YoloDetector",
        lambda *_a, **_kw: SimpleNamespace(
            weights_sha256="test-sha",
            device="cpu",
            detect=lambda _img: [Region("d0", (5.0, 5.0, 25.0, 25.0), "person")],
        ),
    )
    monkeypatch.setattr(
        "camera_system.video_cli.RelateAnythingPredictor",
        lambda *_a, **_kw: SimpleNamespace(
            device="cpu",
            predict=lambda *_a: [],
            rejected_relations=[],
        ),
    )

    rc = video_main(["--video", str(video), "--output", str(output), "--tracker", "kalman"])
    assert rc == 0
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["tracker"]["algorithm"] == "kalman"
    frame_rows = [r for r in rows if r.get("type") == "frame"]
    assert len(frame_rows) == 2
    assert frame_rows[0]["regions"][0]["track_id"] == "track-1"
    assert frame_rows[1]["regions"][0]["track_id"] == "track-1"


def test_benchmark_report_error_attribution_and_cli(tmp_path):
    rows, truth = make_demo()
    report = run_benchmark(rows, truth, settings=ReasoningSettings())
    assert report["summary"]["events"]["f1"] == 1.0
    assert report["per_event_type"]["possible_unattended_bag"]["true_positive"] == 1
    assert len(report["threshold_sweep"]) == 5

    # Inject a low-score relation failure (FN) and an unexpected prediction (FP) to verify attribution
    modified_rows = [dict(r) for r in rows]
    for r in modified_rows:
        if r.get("type") == "frame" and r["frame_index"] == 10:
            r["events"] = [
                {
                    "type": "possible_unattended_bag",
                    "object_track_id": "track-99",
                    "timestamp_seconds": 10.0,
                }
            ]
    truth_with_missed = {
        **truth,
        "events": [
            *truth["events"],
            {"type": "zone_dwell", "object_track_id": "track-2", "timestamp_seconds": 15.0},
        ],
    }
    err_report = run_benchmark(modified_rows, truth_with_missed)
    assert err_report["error_attribution"]["false_positives"][0]["reason"] == "unexpected_event"
    assert err_report["error_attribution"]["false_negatives"][0]["reason"] == (
        "track_missing_near_timestamp"
    )

    sess_path = tmp_path / "sess.jsonl"
    truth_path = tmp_path / "truth.json"
    out_dir = tmp_path / "bench"
    sess_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    truth_path.write_text(json.dumps(truth), encoding="utf-8")
    assert benchmark_main([str(sess_path), "--truth", str(truth_path), "--output-dir", str(out_dir)]) == 0
    assert (out_dir / "report.json").is_file()
    md = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "Calibration Benchmark Report" in md
    assert "Relation Threshold Replay Sweep" in md
    assert benchmark_main([str(sess_path), "--truth", str(truth_path), "--output-dir", str(out_dir)]) == 2

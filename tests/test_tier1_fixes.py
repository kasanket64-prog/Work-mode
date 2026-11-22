"""Regression tests for Tier 1 audit fixes."""

import json
import shutil
import subprocess
import sys
from email.message import Message
from importlib.resources import files
from io import BytesIO
from pathlib import Path
from threading import Thread
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest
from PIL import Image

from camera_system import plates, replay
from camera_system.app import Jobs, make_app
from camera_system.config import load_settings
from camera_system.contracts import Region, Relation
from camera_system.demo import make_demo
from camera_system.detection import load_detector_settings
from camera_system.people import PersonAnalyzer, PoseEstimator
from camera_system.reasoning import load_reasoning
from camera_system.review import HTML, evaluate, link_event_frames, read_records
from camera_system.review import main as review_main
from camera_system.video import FrameSource
from camera_system.video_cli import main as video_main


def _fake_cv(monkeypatch, count=4, fps=5.0):
    capture = Mock()
    capture.isOpened.return_value = True
    state = {"index": -1}

    def read():
        state["index"] += 1
        if state["index"] >= count:
            return False, None
        return True, np.zeros((16, 16, 3), dtype=np.uint8)

    def get(prop):
        if prop == 1:
            return fps
        return state["index"] * (1000.0 / fps)

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


def test_bom_handling_across_all_loaders(tmp_path):
    cfg = tmp_path / "settings.toml"
    cfg.write_text('device = "cpu"\n', encoding="utf-8-sig")
    assert load_settings(cfg).device == "cpu"

    det = tmp_path / "detector.toml"
    det.write_text("max_detections = 12\n", encoding="utf-8-sig")
    assert load_detector_settings(det).max_detections == 12

    reas = tmp_path / "reasoning.toml"
    reas.write_text("min_hits = 2\n", encoding="utf-8-sig")
    assert load_reasoning(reas).min_hits == 2

    rows, truth = make_demo()
    session = tmp_path / "session.jsonl"
    session.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8-sig")
    truth_file = tmp_path / "truth.json"
    truth_file.write_text(json.dumps(truth), encoding="utf-8-sig")
    metrics_file = tmp_path / "metrics.json"

    loaded = read_records(session)
    assert loaded[0]["type"] == "session"
    assert review_main([str(session), "--truth", str(truth_file), "--output", str(metrics_file)]) == 0
    assert json.loads(metrics_file.read_text(encoding="utf-8"))["events"]["true_positive"] == 1


def test_video_preload_fps_rejects_sparse_before_loading_models_and_auto_stride(
    tmp_path, monkeypatch
):
    video = tmp_path / "slow.mp4"
    video.write_bytes(b"stub")
    cfg = tmp_path / "cfg.toml"
    cfg.write_text('device = "cpu"\n')
    det_cfg = tmp_path / "det.toml"
    det_cfg.write_text("max_detections = 4\n")

    _fake_cv(monkeypatch, count=20, fps=5.0)
    detector = Mock()
    detector.return_value.device = "cpu"
    detector.return_value.weights_sha256 = "h"
    detector.return_value.detect.return_value = [
        Region("0", (0, 0, 5, 5), "person"),
        Region("1", (6, 6, 12, 12), "car"),
    ]
    predictor = Mock()
    predictor.return_value.device = "cpu"
    predictor.return_value.predict.return_value = []
    predictor.return_value.rejected_relations = []
    monkeypatch.setattr("camera_system.video_cli.YoloDetector", detector)
    monkeypatch.setattr("camera_system.video_cli.RelateAnythingPredictor", predictor)

    rejected_out = tmp_path / "rejected.jsonl"
    assert (
        video_main(
            [
                "--video",
                str(video),
                "--config",
                str(cfg),
                "--detector-config",
                str(det_cfg),
                "--stride",
                "15",
                "--output",
                str(rejected_out),
            ]
        )
        == 2
    )
    detector.assert_not_called()
    predictor.assert_not_called()
    assert not rejected_out.exists()

    auto_out = tmp_path / "auto.jsonl"
    assert (
        video_main(
            [
                "--video",
                str(video),
                "--config",
                str(cfg),
                "--detector-config",
                str(det_cfg),
                "--stride",
                "15",
                "--auto-stride",
                "--max-frames",
                "2",
                "--output",
                str(auto_out),
            ]
        )
        == 0
    )
    rows = [json.loads(line) for line in auto_out.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["stride"] == 7
    assert rows[0]["requested_stride"] == 15
    assert rows[0]["auto_stride"] is True
    assert rows[-1]["status"] == "completed"


def test_video_rejects_bool_camera_index():
    with pytest.raises(ValueError):
        FrameSource(True)


def test_plates_cli_validation_summary_and_interrupt(tmp_path, monkeypatch, capsys):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"stub")
    _fake_cv(monkeypatch, count=5, fps=10.0)

    monkeypatch.setitem(sys.modules, "fast_alpr", None)
    missing_out = tmp_path / "missing_dep.jsonl"
    assert plates.main([str(video), "--output", str(missing_out)]) == 2
    assert ".[plates]" in capsys.readouterr().err

    alpr_cls = Mock()
    alpr_cls.return_value.predict.return_value = []
    monkeypatch.setitem(sys.modules, "fast_alpr", SimpleNamespace(ALPR=alpr_cls))

    out = tmp_path / "plates.jsonl"
    assert plates.main([str(video), "--stride", "1", "--max-frames", "2", "--output", str(out)]) == 0
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["analysis"] == "plate_candidates"
    assert rows[-1]["type"] == "summary"
    assert rows[-1]["status"] == "completed"
    assert rows[-1]["end_reason"] == "max_frames"
    assert rows[-1]["frames_processed"] == 2
    assert "elapsed_seconds" in rows[-1]

    alpr_cls.reset_mock()
    original_bytes = out.read_bytes()
    assert plates.main([str(video), "--output", str(out)]) == 2
    assert out.read_bytes() == original_bytes
    alpr_cls.assert_not_called()

    _fake_cv(monkeypatch, count=5, fps=10.0)
    alpr_cls.return_value.predict.side_effect = KeyboardInterrupt()
    interrupted = tmp_path / "interrupted.jsonl"
    assert plates.main([str(video), "--stride", "1", "--output", str(interrupted)]) == 130
    summary = json.loads(interrupted.read_text(encoding="utf-8").splitlines()[-1])
    assert summary["status"] == "interrupted"
    assert summary["end_reason"] == "keyboard_interrupt"


def test_unattended_bag_links_full_evidence_interval():
    rows, _ = make_demo()
    linked = link_event_frames(rows)
    events = [e for r in linked if r.get("type") == "frame" for e in r.get("events", [])]
    assert len(events) == 1
    assert events[0]["started_seconds"] == 3.0
    assert events[0]["review_frame_indices"] == list(range(3, 34))


def test_pose_estimator_guards_and_custom_interaction_threshold(monkeypatch):
    monkeypatch.setitem(sys.modules, "ultralytics", None)
    with pytest.raises(RuntimeError, match="detection"):
        PoseEstimator("cpu")

    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=Mock()))
    with pytest.raises(RuntimeError, match="CUDA requested"):
        PoseEstimator("cuda")

    pose = Mock()
    pose.predict.return_value = []
    with pytest.raises(ValueError):
        PersonAnalyzer(pose, relation_threshold=1.5)
    analyzer = PersonAnalyzer(pose, relation_threshold=0.8)
    regions = [
        Region("p", (0, 0, 40, 80), "person", "p1"),
        Region("b", (10, 10, 20, 20), "backpack", "b1"),
    ]
    rels = [
        Relation("p1", "holding", "b1", 0.7),
        Relation("p1", "carrying", "b1", 0.85),
    ]
    obs = analyzer.analyze(Image.new("RGB", (100, 100)), regions, rels)[0]
    assert [i["predicate"] for i in obs["interactions"]] == ["carrying"]


def test_replay_refreshes_person_interactions_and_cli_errors(tmp_path, capsys):
    rows = [
        {"type": "session", "relation_endpoint_kind": "track_id"},
        {
            "type": "frame",
            "frame_index": 0,
            "timestamp_seconds": 0.0,
            "image_size": [100, 100],
            "regions": [
                {"id": "p", "box": [0, 0, 40, 80], "label": "person", "track_id": "p1"},
                {"id": "b", "box": [10, 10, 20, 20], "label": "backpack", "track_id": "b1"},
            ],
            "relations": [
                {"subject_id": "p1", "predicate": "holding", "object_id": "b1", "score": 0.7},
                {"subject_id": "p1", "predicate": "carrying", "object_id": "b1", "score": 0.85},
            ],
            "person_analysis": [
                {
                    "person_id": "p1",
                    "interactions": [
                        {"predicate": "holding", "object_id": "b1", "model_score": 0.7}
                    ],
                }
            ],
        },
    ]
    replayed = list(replay.replay(rows, load_reasoning(None).__class__(relation_threshold=0.8)))
    assert [i["predicate"] for i in replayed[1]["person_analysis"][0]["interactions"]] == [
        "carrying"
    ]

    cfg = tmp_path / "r.toml"
    cfg.write_text("relation_threshold = 0.8\n")
    assert replay.main([str(tmp_path / "missing.jsonl"), "--reasoning-config", str(cfg), "--output", str(tmp_path / "out.jsonl")]) == 2
    assert "camera-replay:" in capsys.readouterr().err


def test_evaluate_validates_track_mapping_and_relations_types():
    rows, truth = make_demo()
    with pytest.raises(ValueError, match="track_mapping"):
        evaluate(rows, dict(truth, track_mapping=["bad"]))
    with pytest.raises(ValueError, match="relations"):
        evaluate(rows, dict(truth, annotated_frame_indices=[0], relations=["bad"]))


def test_app_restart_persistence_cleanup_and_head_zero_byte(tmp_path, monkeypatch):
    key = "a" * 32
    folder = tmp_path / key
    folder.mkdir()
    (folder / "job.json").write_text(
        json.dumps({"id": key, "status": "processing", "mode": "scene", "limit": 5, "created": "2026-01-01"})
    )
    (folder / "input.mp4").write_bytes(b"")
    jobs = Jobs(tmp_path, Path.cwd())
    assert jobs.items[key]["status"] == "interrupted"
    assert json.loads((folder / "job.json").read_text())["status"] == "interrupted"

    with make_app(tmp_path, Path.cwd(), 0) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            # Verify CSP header on HTML and HEAD + 0-byte media
            with urlopen(base + "/") as resp:
                assert "Content-Security-Policy" in resp.headers
            with urlopen(Request(base + f"/jobs/{key}/media", method="HEAD")) as resp:
                assert resp.status == 200
                assert resp.headers["Content-Length"] == "0"
            with urlopen(base + f"/jobs/{key}/media") as resp:
                assert resp.read() == b""
            with pytest.raises(HTTPError) as err:
                urlopen(Request(base + f"/jobs/{key}/media", headers={"Range": "bytes=0-0"}))
            assert err.value.code == 416

            # Failed upload mid-stream cleans up orphan directory
            handler = server.RequestHandlerClass.__new__(server.RequestHandlerClass)
            handler.server = server
            handler.path = "/api/jobs"
            handler.connection = Mock()
            handler.rfile = BytesIO(b"ab")  # shorter than Content-Length 10
            handler.headers = Message()
            host = f"127.0.0.1:{server.server_port}"
            for k, v in {
                "Host": host,
                "Origin": f"http://{host}",
                "Content-Length": "10",
                "X-File-Extension": ".mp4",
            }.items():
                handler.headers[k] = v
            handler.respond = Mock()
            before_dirs = {p.name for p in tmp_path.iterdir()}
            handler.do_POST()
            after_dirs = {p.name for p in tmp_path.iterdir()}
            assert before_dirs == after_dirs
            assert server.jobs.active is None

            # Failed retry cleans up orphan directory
            key2 = "b" * 32
            folder2 = tmp_path / key2
            folder2.mkdir()
            (folder2 / "input.mp4").write_bytes(b"video")
            server.jobs.items[key2] = {
                "id": key2,
                "status": "failed",
                "mode": "scene",
                "limit": 5,
                "created": "2026-01-01",
            }
            before_dirs = {p.name for p in tmp_path.iterdir()}
            monkeypatch.setattr("camera_system.app.shutil.copyfile", Mock(side_effect=OSError("full")))
            with pytest.raises(OSError):
                server.jobs.retry(key2)
            assert {p.name for p in tmp_path.iterdir()} == before_dirs
            assert server.jobs.active is None
        finally:
            server.shutdown()
            thread.join(timeout=5)


def test_workspace_js_dotless_and_review_html_preserves_video():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed for browser logic checks")
    script = files("camera_system").joinpath("templates/workspace.js").read_text(encoding="utf-8")
    fn = "function uploadError" + script.split("function uploadError", 1)[1].split("function message", 1)[0]
    subprocess.run(
        [
            node,
            "-e",
            fn
            + """
const assert = require('node:assert/strict');
assert.match(uploadError({name:'mp4', size:10}, 'scene'), /Choose an/);
""",
        ],
        check=True,
        timeout=10,
    )

    link_fn = "function linkEventFrames" + HTML.split("function linkEventFrames", 1)[1].split(
        "function load", 1
    )[0]
    subprocess.run(
        [
            node,
            "-e",
            link_fn
            + """
const assert = require('node:assert/strict');
const rows = [
  {type:'session'},
  {type:'frame', frame_index:0, timestamp_seconds:0, regions:[{track_id:'bag'}], events:[]},
  {type:'frame', frame_index:1, timestamp_seconds:5, regions:[{track_id:'bag'}], events:[{type:'possible_unattended_bag', object_track_id:'bag', started_seconds:0, timestamp_seconds:5}]}
];
const linked = linkEventFrames(rows);
assert.deepEqual(linked[2].events[0].review_frame_indices, [0, 1]);
""",
        ],
        check=True,
        timeout=10,
    )
    assert "if(cleanupWorkspacePlayback)cleanupWorkspacePlayback();" in HTML

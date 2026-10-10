"""Regressions for the six 2026-10-09 audit findings."""

import json
from email.message import Message
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock

import pytest

from camera_system.app import Jobs, command, make_app
from camera_system.cli import main
from camera_system.contracts import Region, Relation
from camera_system.reasoning import Reasoner, ReasoningSettings
from camera_system.review import evaluate, make_server


@pytest.mark.parametrize("fps", [25, 30000 / 1001, 30])
def test_website_sampling_confirms_fresh_evidence(fps):
    args = command(Path("clip.mp4"), Path("result.jsonl"), "scene", 30)
    stride = int(args[args.index("--stride") + 1])
    every = int(args[args.index("--relation-every") + 1])
    engine = Reasoner(ReasoningSettings())
    regions = [Region("p", (0, 0, 10, 10), "person", "p"),
               Region("b", (20, 20, 30, 30), "backpack", "b")]
    confirmed = []
    for i in range(30):
        relations = [Relation("p", "holding", "b", 0.9)] if i % every == 0 else []
        current, _ = engine.update(i * stride / fps, regions, relations, (100, 100))
        if not relations:
            assert not current
        confirmed.extend(current)
    assert confirmed


def test_image_existing_output_is_preserved_before_inference(tmp_path, monkeypatch):
    target = tmp_path / "important.json"
    target.write_bytes(b"original content")
    predictor = Mock()
    monkeypatch.setattr("camera_system.cli.RelateAnythingPredictor", predictor)
    assert main(["--image", "examples/horse.jpg", "--regions", "examples/horse.regions.json",
                 "--output", str(target)]) == 2
    assert target.read_bytes() == b"original content"
    predictor.assert_not_called()


@pytest.mark.parametrize("startup_failure", [False, True])
def test_upload_slot_survives_response_disconnect(tmp_path, monkeypatch, startup_failure):
    thread = Mock()
    if startup_failure:
        thread.start.side_effect = OSError("cannot start worker")
    monkeypatch.setattr("camera_system.app.Thread", Mock(return_value=thread))
    with make_app(tmp_path, Path.cwd(), 0) as server:
        handler = server.RequestHandlerClass.__new__(server.RequestHandlerClass)
        handler.server = server
        handler.path = "/api/jobs"
        handler.connection = Mock()
        handler.rfile = BytesIO(b"fake video")
        handler.headers = Message()
        host = f"127.0.0.1:{server.server_port}"
        for key, value in {"Host": host, "Origin": f"http://{host}",
                           "Content-Length": "10", "X-File-Extension": ".mp4"}.items():
            handler.headers[key] = value
        handler.respond = Mock(side_effect=None if startup_failure else BrokenPipeError())
        handler.do_POST()
        item = next(iter(server.jobs.items.values()))
        if startup_failure:
            assert server.jobs.active is None
            assert item["status"] == "failed"
            assert handler.respond.call_args.args[0] == 400
        else:
            assert server.jobs.active == item["id"] and item["status"] == "processing"
            handler.respond = Mock()
            handler.do_POST()
            assert handler.respond.call_args.args[0] == 409
            assert server.jobs.cancel(item["id"])


def test_worker_completion_only_releases_its_own_slot(tmp_path, monkeypatch):
    jobs = Jobs(tmp_path, Path.cwd())
    item = {"id": "a" * 32, "mode": "scene", "limit": 1, "status": "processing"}
    (tmp_path / item["id"]).mkdir()
    jobs.active = "another-job"
    monkeypatch.setattr("camera_system.app.subprocess.Popen", Mock(side_effect=OSError("failed")))
    jobs.run(item, Path("missing.mp4"))
    assert item["status"] == "failed" and jobs.active == "another-job"


def test_evaluation_cannot_dilute_partial_run_false_alarms():
    event = {"type": "test", "object_track_id": "p", "timestamp_seconds": 10}
    rows = [{"type": "frame", "timestamp_seconds": 0},
            {"type": "frame", "timestamp_seconds": 10, "events": [event]},
            {"type": "summary", "status": "completed", "end_reason": "max_frames"}]
    for duration in (10, 3600):
        with pytest.raises(ValueError):
            evaluate(rows, {"duration_seconds": duration, "events": []})
    truth = {"duration_seconds": 10, "events": [], "evaluation_interval_seconds": [0, 10]}
    assert evaluate(rows, truth)["false_alarms_per_hour"] == 360
    with pytest.raises(ValueError):
        evaluate(rows, dict(truth, duration_seconds=3600))
    rows[-1]["end_reason"] = "video_eof_or_decode_failure"
    assert evaluate(rows, {"duration_seconds": 10, "events": []})["false_alarms_per_hour"] == 360


def test_evaluation_nonzero_start_and_out_of_scope_annotations():
    rows = [{"type": "frame", "timestamp_seconds": t} for t in (20, 25, 30)]
    truth = {"duration_seconds": 10, "evaluation_interval_seconds": [20, 30], "events": []}
    result = evaluate(rows, truth)
    assert result["duration_seconds"] == 10 and result["evaluation_interval_seconds"] == [20, 30]
    with pytest.raises(ValueError, match="timestamp"):
        evaluate(rows, dict(truth, events=[{"timestamp_seconds": 15}]))
    for interval in ([0, 30], [20, 31], [True, 30], [20, float("nan")]):
        with pytest.raises(ValueError):
            evaluate(rows, dict(truth, evaluation_interval_seconds=interval))


@pytest.mark.parametrize("path", ["/", "/api/session"])
@pytest.mark.parametrize("host", ["untrusted.example", None, "localhost", "127.0.0.1:1"])
def test_standalone_review_rejects_untrusted_hosts(path, host):
    with make_server([{ "type": "session", "note": "private"}], 0) as server:
        handler = server.RequestHandlerClass.__new__(server.RequestHandlerClass)
        handler.server = server
        handler.path = path
        handler.headers = {"Host": host}
        handler.send_error = Mock()
        handler.wfile = BytesIO()
        handler.do_GET()
        handler.send_error.assert_called_once_with(403)
        assert not handler.wfile.getvalue()


def test_completed_progress_is_cached_across_restart(tmp_path, monkeypatch):
    folder = tmp_path / ("a" * 32)
    folder.mkdir()
    (folder / "job.json").write_text(json.dumps({"id": folder.name, "status": "completed", "created": "now"}))
    result = folder / "result.jsonl"
    result.write_text('{"type":"frame","timestamp_seconds":0}\n{"type":"summary","end_reason":"max_frames"}')
    jobs = Jobs(tmp_path, Path.cwd())
    assert jobs.snapshot()[0]["frames_processed"] == 1
    assert "progress" in json.loads((folder / "job.json").read_text())
    restarted = Jobs(tmp_path, Path.cwd())
    original = Path.open

    def guarded(path, *args, **kwargs):
        assert path != result, "completed result must not be opened again"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    assert jobs.snapshot()[0]["frames_processed"] == 1
    assert restarted.snapshot()[0]["frames_processed"] == 1


def test_active_progress_reads_only_appended_records(tmp_path, monkeypatch):
    folder = tmp_path / ("b" * 32)
    folder.mkdir()
    (folder / "job.json").write_text(json.dumps({"id": folder.name, "status": "processing", "created": "now"}))
    result = folder / "result.jsonl"
    first = b'{"type":"frame","timestamp_seconds":0}\n'
    result.write_bytes(first + b'{"type":')
    jobs = Jobs(tmp_path, Path.cwd())
    jobs.items[folder.name]["status"] = "processing"
    assert jobs.snapshot()[0]["frames_processed"] == 1
    with result.open("ab") as stream:
        stream.write(b'"frame","timestamp_seconds":1}\n')
    original = Path.open
    opened = []

    def tracked(path, *args, **kwargs):
        stream = original(path, *args, **kwargs)
        if path == result:
            wrapper = Mock(wraps=stream)
            opened.append(wrapper)
            return StreamSpy(stream, wrapper)
        return stream

    class StreamSpy:
        def __init__(self, stream, wrapper):
            self.stream, self.wrapper = stream, wrapper
        def __enter__(self):
            return self.wrapper
        def __exit__(self, *args):
            self.stream.close()

    monkeypatch.setattr(Path, "open", tracked)
    assert jobs.snapshot()[0]["frames_processed"] == 2
    opened[0].seek.assert_called_once_with(len(first))
    assert jobs.snapshot()[0]["frames_processed"] == 2
    assert len(opened) == 1

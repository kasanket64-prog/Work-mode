import json
from pathlib import Path
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from camera_system.app import Jobs, command, make_app
from camera_system.demo import make_demo


def test_upload_bounds_origin_and_history(tmp_path, monkeypatch):
    monkeypatch.setattr("camera_system.app.Jobs.run", lambda *args: None)
    with make_app(tmp_path, Path.cwd(), 0) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            headers = {"Origin": base, "X-File-Extension": ".mp4", "X-Analysis-Mode": "scene"}
            with pytest.raises(HTTPError) as error:
                urlopen(
                    Request(
                        base + "/api/jobs",
                        data=b"fake",
                        headers={**headers, "Origin": "https://foreign.test"},
                    )
                )
            assert error.value.code == 403
            with pytest.raises(HTTPError) as error:
                urlopen(
                    Request(
                        base + "/api/jobs",
                        data=b"fake",
                        headers={**headers, "X-File-Extension": "../file"},
                    )
                )
            assert error.value.code == 400
            with urlopen(Request(base + "/api/jobs", data=b"fake", headers=headers)) as response:
                item = json.load(response)
                assert response.status == 202
            assert (tmp_path / item["id"] / "input.mp4").read_bytes() == b"fake"
            with urlopen(base + "/api/jobs") as response:
                assert json.load(response)[0]["status"] == "processing"
            with pytest.raises(HTTPError) as error:
                urlopen(Request(base + "/api/jobs", data=b"fake", headers=headers))
            assert error.value.code == 409
            with urlopen(
                Request(base + f"/api/jobs/{item['id']}/cancel", data=b"{}", headers=headers)
            ) as response:
                assert response.status == 202
            assert server.jobs.items[item["id"]]["status"] == "cancelling"
            assert not server.jobs.cancel("0" * 32)
        finally:
            server.shutdown()
            thread.join()


def test_commands_are_fixed_and_separate_modes():
    scene = command(Path("video.mp4"), Path("result"), "scene", 3)
    assert "camera_system.video_cli" in scene and "--analyze-people" in scene
    assert "camera_system.cli" in command(Path("image.jpg"), Path("result"), "scene", 3)
    assert "camera_system.plates" in command(Path("video.mp4"), Path("result"), "plates", 3)


def test_retry_preserves_original_and_rejects_duplicate_or_missing_upload(tmp_path, monkeypatch):
    monkeypatch.setattr("camera_system.app.Jobs.run", lambda *args: None)
    key = "c" * 32
    folder = tmp_path / key
    folder.mkdir()
    previous = {"id": key, "status": "cancelled", "created": "now", "mode": "scene", "limit": 2}
    (folder / "job.json").write_text(json.dumps(previous))
    (folder / "input.mp4").write_bytes(b"saved video")
    (folder / "result.jsonl").write_text("old partial result")
    jobs = Jobs(tmp_path, Path.cwd())
    retry = jobs.retry(key)
    jobs.worker.join()
    assert retry["id"] != key and retry["retry_of"] == key
    assert retry["limit"] == 2 and retry["mode"] == "scene"
    assert (tmp_path / retry["id"] / "input.mp4").read_bytes() == b"saved video"
    assert (folder / "result.jsonl").read_text() == "old partial result"
    assert json.loads((folder / "job.json").read_text()) == previous
    with pytest.raises(ValueError, match="no active"):
        jobs.retry(key)
    jobs.active = None
    (folder / "input.mp4").unlink()
    with pytest.raises(ValueError, match="missing"):
        jobs.retry(key)
    assert jobs.active is None


def test_retry_copy_failure_releases_slot(tmp_path, monkeypatch):
    key = "d" * 32
    folder = tmp_path / key
    folder.mkdir()
    (folder / "input.mp4").write_bytes(b"video")
    jobs = Jobs(tmp_path, Path.cwd())
    jobs.items[key] = {"id": key, "status": "failed", "mode": "scene", "limit": 2}

    def fail(*args):
        raise OSError("disk full")

    monkeypatch.setattr("camera_system.app.shutil.copyfile", fail)
    with pytest.raises(OSError, match="disk full"):
        jobs.retry(key)
    assert jobs.active is None and jobs.items[key]["status"] == "failed"


def test_progress_coverage_handles_partial_lines_and_pretty_image(tmp_path):
    key = "b" * 32
    folder = tmp_path / key
    folder.mkdir()
    (folder / "job.json").write_text(
        json.dumps({"id": key, "status": "completed", "created": "now"})
    )
    output = folder / "result.jsonl"
    output.write_text('{"type":"frame","timestamp_seconds":1.5}\n{"type":')
    jobs = Jobs(tmp_path, Path.cwd())
    item = jobs.snapshot()[0]
    assert item["frames_processed"] == 1
    assert item["last_timestamp_seconds"] == 1.5
    assert item["coverage"] == "Coverage not recorded"
    with output.open("w") as stream:
        stream.write('{"type":"summary","end_reason":"max_frames"}\n')
    assert "full video coverage is not established" in jobs.snapshot()[0]["coverage"]
    output.write_text('{"type":"summary","end_reason":"video_eof_or_decode_failure"}\n')
    assert "decoding stopped" in jobs.snapshot()[0]["coverage"]
    (folder / "input.jpg").write_bytes(b"test")
    output.write_text(json.dumps({"regions": [], "status": "predicted"}, indent=2))
    item = jobs.snapshot()[0]
    assert item["frames_processed"] == 1 and item["coverage"] == "Single image"


def test_completed_job_evaluation_and_invalid_annotations(tmp_path):
    rows, truth = make_demo()
    key = "a" * 32
    folder = tmp_path / key
    folder.mkdir()
    (folder / "processing.log").write_text("<script>diagnostic</script>", encoding="utf-8")
    (folder / "result.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
    )
    (folder / "job.json").write_text(
        json.dumps({"id": key, "status": "completed", "created": "2026-10-07"}), encoding="utf-8"
    )
    with make_app(tmp_path, Path.cwd(), 0) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            with urlopen(base + f"/jobs/{key}/log") as response:
                assert response.headers.get_content_type() == "text/plain"
                assert response.headers["X-Content-Type-Options"] == "nosniff"
                assert response.read() == b"<script>diagnostic</script>"

            def request(payload):
                return Request(
                    base + f"/api/jobs/{key}/evaluate",
                    data=json.dumps(payload).encode(),
                    headers={"Origin": base, "Content-Type": "application/json"},
                )

            with urlopen(request(truth)) as response:
                assert json.load(response)["events"]["true_positive"] == 1
            for payload in (
                {}, dict(truth, duration_seconds=1), dict(truth, events=[None]),
                dict(truth, relations=[]),
                dict(truth, annotation_status="draft"),
                dict(truth, relations=[], annotated_frame_indices=[]),
            ):
                with pytest.raises(HTTPError) as error:
                    urlopen(request(payload))
                assert error.value.code == 400
        finally:
            server.shutdown()
            thread.join()

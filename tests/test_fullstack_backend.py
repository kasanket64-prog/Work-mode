"""Tests for Tier 2 fullstack backend: SQLite store, warm worker, SSE, REST APIs, and security."""

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Thread
from unittest.mock import Mock
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from PIL import Image

from camera_system.app import Jobs, make_app, validate_magic_bytes
from camera_system.cli import main as cli_main
from camera_system.config import Settings
from camera_system.demo import make_demo
from camera_system.detection import DetectorSettings
from camera_system.store import WorkspaceStore
from camera_system.worker import ModelCache, WarmWorkerManager


def test_sqlite_store_persistence_and_job_json_sync(tmp_path):
    key = "a" * 32
    folder = tmp_path / key
    folder.mkdir()
    item = {
        "id": key,
        "status": "completed",
        "mode": "scene",
        "limit": 30,
        "stride": "15",
        "reasoning_profile": "default",
        "vehicle_crops": False,
        "device": "cpu",
        "created": "2026-10-10T00:00:00+00:00",
        "bytes": 123,
    }
    (folder / "job.json").write_text(json.dumps(item), encoding="utf-8")

    jobs = Jobs(tmp_path, Path.cwd())
    try:
        assert jobs.store.list_jobs()[0]["id"] == key
        zone = jobs.store.upsert_zone(
            {"id": "entry-door", "box": [0.1, 0.1, 0.5, 0.9], "labels": ["person"]}
        )
        assert zone["id"] == "entry-door"
        jobs.store.set_verdicts(key, {"event-1": {"verdict": "correct", "note": "verified"}})
        jobs.store.save_annotation(key, {"annotation_status": "draft", "events": None})
        jobs.store.add_evaluation(key, {"events": {"true_positive": 1}})
    finally:
        jobs.close()

    reopened = WorkspaceStore(tmp_path / "fieldnote.db")
    try:
        assert reopened.list_jobs()[0]["id"] == key
        assert reopened.list_zones()[0]["id"] == "entry-door"
        assert reopened.get_verdicts(key)["event-1"]["verdict"] == "correct"
        assert reopened.get_annotation(key)["annotation_status"] == "draft"
        assert len(reopened.list_evaluations(key)) == 1
    finally:
        reopened.close()


def test_warm_worker_model_cache_reuses_loaded_models():
    cache = ModelCache()
    det_factory = Mock(side_effect=lambda *a, **kw: Mock())
    pred_factory = Mock(side_effect=lambda *a, **kw: Mock())
    pose_factory = Mock(side_effect=lambda *a, **kw: Mock())
    alpr_factory = Mock(side_effect=lambda: Mock())

    d1 = cache.get_detector(DetectorSettings(), "cpu", factory=det_factory)
    d2 = cache.get_detector(DetectorSettings(), "cpu", factory=det_factory)
    assert d1 is d2 and cache.load_counts["detector"] == 1
    cache.get_detector(DetectorSettings(image_size=1280), "cpu", factory=det_factory)
    assert cache.load_counts["detector"] == 2

    p1 = cache.get_predictor(Settings(), factory=pred_factory)
    p2 = cache.get_predictor(Settings(), factory=pred_factory)
    assert p1 is p2 and cache.load_counts["predictor"] == 1

    pose1 = cache.get_pose("cpu", factory=pose_factory)
    pose2 = cache.get_pose("cpu", factory=pose_factory)
    assert pose1 is pose2 and cache.load_counts["pose"] == 1

    a1 = cache.get_alpr(factory=alpr_factory)
    a2 = cache.get_alpr(factory=alpr_factory)
    assert a1 is a2 and cache.load_counts["alpr"] == 1


def test_warm_worker_manager_stub_reuse_cancel_and_crash_recovery(tmp_path):
    stub_script = tmp_path / "stub_worker.py"
    stub_script.write_text(
        """
import json, os, sys, time
print(json.dumps({"event": "ready"}), flush=True)
for line in sys.stdin:
    cmd = json.loads(line)
    if cmd.get("cmd") == "shutdown":
        break
    if cmd.get("cmd") == "run":
        mode = cmd.get("mode")
        if mode == "crash":
            sys.exit(1)
        if mode == "hang":
            time.sleep(10)
        print(json.dumps({"event": "progress", "frames_processed": 1, "pid": os.getpid()}), flush=True)
        print(json.dumps({"event": "done", "status": "completed", "exit_code": 0, "pid": os.getpid()}), flush=True)
""",
        encoding="utf-8",
    )
    mgr = WarmWorkerManager(tmp_path, command=[sys.executable, str(stub_script)])
    try:
        log = tmp_path / "worker.log"
        r1 = mgr.run_job({"mode": "ok"}, log, is_cancelled=lambda: False)
        r2 = mgr.run_job({"mode": "ok"}, log, is_cancelled=lambda: False)
        assert r1["pid"] == r2["pid"]  # Same warm process reused across jobs

        with pytest.raises(RuntimeError, match="unexpectedly"):
            mgr.run_job({"mode": "crash"}, log, is_cancelled=lambda: False)

        # Auto-recovers with fresh worker process
        r3 = mgr.run_job({"mode": "ok"}, log, is_cancelled=lambda: False)
        assert r3["exit_code"] == 0 and r3["pid"] != r1["pid"]

        # Cooperative cancel terminates hanging job cleanly
        calls = {"n": 0}

        def cancel_soon():
            calls["n"] += 1
            return calls["n"] >= 2

        cancelled = mgr.run_job({"mode": "hang"}, log, is_cancelled=cancel_soon)
        assert cancelled["status"] == "cancelled"
    finally:
        mgr.close()


def test_sidecar_frames_and_cli_preview_dir(tmp_path, monkeypatch):
    image = tmp_path / "img.png"
    Image.new("RGB", (80, 60), "green").save(image)
    regions = tmp_path / "regions.json"
    regions.write_text(
        json.dumps(
            [
                {"id": "a", "box": [0, 0, 20, 20], "label": "person"},
                {"id": "b", "box": [30, 30, 50, 50], "label": "horse"},
            ]
        )
    )
    predictor = Mock()
    predictor.return_value.device = "cpu"
    predictor.return_value.predict.return_value = []
    predictor.return_value.rejected_relations = []
    monkeypatch.setattr("camera_system.cli.RelateAnythingPredictor", predictor)

    preview_dir = tmp_path / "frames"
    out = tmp_path / "res.json"
    assert (
        cli_main(
            [
                "--image",
                str(image),
                "--regions",
                str(regions),
                "--include-preview",
                "--preview-dir",
                str(preview_dir),
                "--output",
                str(out),
            ]
        )
        == 0
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["preview_path"] == "image-0.jpg"
    assert "preview_data_url" not in payload
    assert (preview_dir / "image-0.jpg").is_file()


def test_sse_system_status_frames_and_delete_job(tmp_path):
    key_done = "a" * 32
    folder_done = tmp_path / key_done
    (folder_done / "frames").mkdir(parents=True)
    (folder_done / "frames" / "frame-0.jpg").write_bytes(b"\xff\xd8\xff\xe0jpeg")
    (folder_done / "job.json").write_text(
        json.dumps({"id": key_done, "status": "completed", "mode": "scene", "limit": 10, "created": "2026-10-10"})
    )

    with make_app(tmp_path, Path.cwd(), 0) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            with urlopen(base + "/api/system/status") as resp:
                status = json.load(resp)
                assert status["job_count"] == 1
                assert status["warm_worker"] is False
                assert "capabilities" in status and "workspace_bytes" in status

            with urlopen(base + f"/jobs/{key_done}/frames/frame-0.jpg") as resp:
                assert resp.headers["Content-Type"] == "image/jpeg"
                assert resp.read().startswith(b"\xff\xd8\xff")

            with pytest.raises(HTTPError) as err:
                urlopen(base + f"/jobs/{key_done}/frames/../job.json")
            assert err.value.code == 404

            # SSE stream emits initial jobs snapshot cleanly
            with urlopen(base + "/api/events", timeout=5) as sse:
                assert sse.headers["Content-Type"].startswith("text/event-stream")
                lines = []
                for _ in range(3):
                    line = sse.readline().decode("utf-8")
                    lines.append(line)
                    if line == "\n":
                        break
                assert lines[0].startswith("event: jobs")
                assert key_done in lines[1]

            # Active job cannot be deleted
            server.jobs.active = "f" * 32
            server.jobs.items["f" * 32] = {
                "id": "f" * 32,
                "status": "processing",
                "mode": "scene",
                "limit": 10,
                "created": "2026-10-11",
            }
            with pytest.raises(HTTPError) as err:
                urlopen(Request(base + f"/api/jobs/{'f' * 32}", method="DELETE", headers={"Origin": base}))
            assert err.value.code == 409
            server.jobs.active = None

            # Completed job deletes cleanly from disk and store
            with urlopen(Request(base + f"/api/jobs/{key_done}", method="DELETE", headers={"Origin": base})) as resp:
                deleted = json.load(resp)
                assert deleted["deleted"] == key_done and deleted["freed_bytes"] > 0
            assert not folder_done.exists()
            with urlopen(base + "/api/jobs") as resp:
                ids = [j["id"] for j in json.load(resp)]
                assert key_done not in ids
        finally:
            server.shutdown()
            thread.join(timeout=5)


def test_verdicts_zones_replay_annotations_evaluations_flow(tmp_path):
    rows, truth = make_demo()
    key = "c" * 32
    folder = tmp_path / key
    folder.mkdir()
    (folder / "result.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    (folder / "job.json").write_text(
        json.dumps({"id": key, "status": "completed", "mode": "scene", "limit": 36, "created": "2026-10-10"})
    )

    with make_app(tmp_path, Path.cwd(), 0) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            # Persist verdict and verify review HTML includes it
            with urlopen(
                Request(
                    base + f"/api/jobs/{key}/verdicts",
                    data=json.dumps(
                        {"verdicts": {"event-1": {"verdict": "correct", "note": "clear"}}}
                    ).encode(),
                    method="PUT",
                    headers={"Origin": base, "Content-Type": "application/json"},
                )
            ) as resp:
                assert json.load(resp)["verdicts"]["event-1"]["verdict"] == "correct"

            with urlopen(base + f"/jobs/{key}/review") as resp:
                html = resp.read().decode("utf-8")
                assert '"review_verdict": "correct"' in html

            # Create zone, validate invalid zone rejection, then replay
            with pytest.raises(HTTPError) as err:
                urlopen(
                    Request(
                        base + "/api/zones",
                        data=json.dumps({"id": "bad", "box": [0, 0, 2, 1], "labels": ["person"]}).encode(),
                        headers={"Origin": base, "Content-Type": "application/json"},
                    )
                )
            assert err.value.code == 400

            with urlopen(
                Request(
                    base + "/api/zones",
                    data=json.dumps({"id": "bag-zone", "box": [0.1, 0.1, 0.9, 0.9], "labels": ["backpack"]}).encode(),
                    headers={"Origin": base, "Content-Type": "application/json"},
                )
            ) as resp:
                assert len(json.load(resp)["zones"]) == 1

            with urlopen(
                Request(
                    base + f"/api/jobs/{key}/replay",
                    data=json.dumps({"dwell_seconds": 0.0, "use_saved_zones": True}).encode(),
                    headers={"Origin": base, "Content-Type": "application/json"},
                )
            ) as resp:
                rep = json.load(resp)
                assert rep["status"] == "replayed" and rep["zones_count"] == 1
            assert (folder / "original-result.jsonl").is_file()

            # Annotations draft -> save -> evaluate -> evaluations list
            with urlopen(base + f"/api/jobs/{key}/annotations") as resp:
                draft = json.load(resp)
                assert draft["annotation_status"] == "draft"

            reviewed = dict(draft, **truth, annotation_status="reviewed")
            with urlopen(
                Request(
                    base + f"/api/jobs/{key}/annotations",
                    data=json.dumps(reviewed).encode(),
                    method="PUT",
                    headers={"Origin": base, "Content-Type": "application/json"},
                )
            ) as resp:
                assert json.load(resp)["annotation_status"] == "reviewed"

            with urlopen(
                Request(
                    base + f"/api/jobs/{key}/evaluate",
                    data=json.dumps(reviewed).encode(),
                    headers={"Origin": base, "Content-Type": "application/json"},
                )
            ) as resp:
                assert json.load(resp)["events"]["true_positive"] == 1

            with urlopen(base + f"/api/jobs/{key}/evaluations") as resp:
                assert len(json.load(resp)["evaluations"]) == 1

            with urlopen(
                Request(base + "/api/zones/bag-zone", method="DELETE", headers={"Origin": base})
            ) as resp:
                assert json.load(resp)["deleted"] == "bag-zone"
        finally:
            server.shutdown()
            thread.join(timeout=5)


def test_strict_uploads_magic_bytes_auth_and_retention(tmp_path):
    good_png = tmp_path / "good.png"
    good_png.write_bytes(b"\x89PNG\r\n\x1a\n0000")
    validate_magic_bytes(good_png, ".png")
    bad_png = tmp_path / "bad.png"
    bad_png.write_bytes(b"not-a-png")
    with pytest.raises(ValueError, match="media format"):
        validate_magic_bytes(bad_png, ".png")

    with pytest.raises(ValueError, match="auth token"):
        make_app(tmp_path, Path.cwd(), 0, host="0.0.0.0")

    # Seed old job for retention pruning
    old_id = "e" * 32
    old_dir = tmp_path / old_id
    old_dir.mkdir()
    old_date = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
    (old_dir / "job.json").write_text(
        json.dumps({"id": old_id, "status": "completed", "mode": "scene", "limit": 5, "created": old_date})
    )

    with make_app(
        tmp_path,
        Path.cwd(),
        0,
        auth_token="secret-tok",
        strict_uploads=True,
        retention_days=5,
    ) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            # Pruned on startup
            assert not old_dir.exists()

            # Unauthenticated request rejected
            with pytest.raises(HTTPError) as err:
                urlopen(base + "/api/jobs")
            assert err.value.code == 403

            with urlopen(Request(base + "/api/jobs", headers={"Authorization": "Bearer secret-tok"})) as resp:
                assert resp.status == 200

            # Strict upload rejects fake PNG bytes
            with pytest.raises(HTTPError) as err:
                urlopen(
                    Request(
                        base + "/api/jobs",
                        data=b"not actually png",
                        headers={
                            "Origin": base,
                            "X-Fieldnote-Token": "secret-tok",
                            "X-File-Extension": ".png",
                            "X-Analysis-Mode": "scene",
                            "X-Frame-Limit": "1",
                        },
                    )
                )
            assert err.value.code == 400
            assert server.jobs.active is None
        finally:
            server.shutdown()
            thread.join(timeout=5)

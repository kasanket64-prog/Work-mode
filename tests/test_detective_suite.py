"""Tests for the Detective & Forensic Investigation Suite (cases, custody, search, reports)."""

import hashlib
import json
import sqlite3
from pathlib import Path
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from PIL import Image

import camera_system.app as app_module
from camera_system.demo import make_demo
from camera_system.report import compute_wall_clock
from camera_system.store import WorkspaceStore


def test_wall_clock_and_hash_chained_audit_tamper_detection(tmp_path):
    assert (
        compute_wall_clock("2026-11-22T14:00:00Z", 12.5, clock_offset_seconds=60.0)
        == "2026-11-22T14:01:12+00:00"
    )
    assert compute_wall_clock("not-a-date", 10.0) is None

    db_path = tmp_path / "fieldnote.db"
    store = WorkspaceStore(db_path)
    try:
        c = store.create_case({"case_number": "CASE-101", "title": "Alley Entry"})
        store.append_audit("case_created", "case", c["id"], {"case_number": "CASE-101"})
        store.append_audit("media_ingested", "job", "a" * 32, {"sha256": "deadbeef"})
        valid, count = store.verify_audit_chain()
        assert valid is True and count == 2
    finally:
        store.close()

    # Simulate direct DB tampering and verify cryptographic tamper detection catches it.
    conn = sqlite3.connect(str(db_path))
    conn.execute("UPDATE audit_log SET action = 'tampered_action' WHERE id = 2")
    conn.commit()
    conn.close()

    reopened = WorkspaceStore(db_path)
    try:
        valid_after, count_after = reopened.verify_audit_chain()
        assert valid_after is False and count_after == 2
    finally:
        reopened.close()


def test_detective_end_to_end_cases_search_bookmarks_entities_and_report(tmp_path, monkeypatch):
    demo_rows, _ = make_demo()
    enriched_rows = [dict(r) for r in demo_rows]
    for r in enriched_rows:
        if r.get("type") == "frame" and r["frame_index"] == 3:
            r["person_analysis"] = [
                {"person_id": "track-2", "appearance": {"value": "dark blue jacket"}}
            ]
            r["plates"] = [{"text": "ABC74K", "ocr_score": 0.91, "status": "ok"}]
            r["preview_path"] = "frame-3.jpg"

    mp4_bytes = b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2mp41" + b"\x01\x02\x03" * 32
    expected_sha256 = hashlib.sha256(mp4_bytes).hexdigest()

    class FakeProc:
        def __init__(self, cmd):
            out_path = Path(cmd[cmd.index("--output") + 1])
            preview_dir = (
                Path(cmd[cmd.index("--preview-dir") + 1]) if "--preview-dir" in cmd else None
            )
            if preview_dir is not None:
                preview_dir.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (32, 32), (20, 40, 80)).save(preview_dir / "frame-3.jpg", "JPEG")
            out_path.write_text(
                "\n".join(json.dumps(r) for r in enriched_rows) + "\n", encoding="utf-8"
            )
            self.pid = 77777
            self.returncode = 0

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(app_module.subprocess, "Popen", lambda cmd, **_kw: FakeProc(cmd))

    with app_module.make_app(tmp_path, Path.cwd(), 0, strict_uploads=True) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            # 1. Create a case with XSS-prone strings to verify HTML escaping in report later.
            with urlopen(
                Request(
                    base + "/api/cases",
                    data=json.dumps(
                        {
                            "id": "case-alley-01",
                            "case_number": "CASE-2026-099",
                            "title": "Loading Dock <script>alert(1)</script>",
                            "investigator": "Det. K. Rao",
                            "notes": "Night shift entry review",
                        }
                    ).encode(),
                    headers={"Origin": base, "Content-Type": "application/json"},
                )
            ) as resp:
                assert resp.status == 201
                assert json.load(resp)["case"]["id"] == "case-alley-01"

            # 2. Ingest footage linked to the case with camera label & DVR clock skew.
            upload_req = Request(
                base + "/api/jobs",
                data=mp4_bytes,
                headers={
                    "Origin": base,
                    "X-File-Extension": ".mp4",
                    "X-Analysis-Mode": "scene",
                    "X-Frame-Limit": "36",
                    "X-Case-Id": "case-alley-01",
                    "X-Camera-Label": "Cam-North-Dock",
                    "X-Original-Filename": "dock_north.mp4",
                    "X-Recorded-Start": "2026-11-22T18:00:00Z",
                    "X-Clock-Offset": "30.0",
                },
            )
            with urlopen(upload_req) as resp:
                assert resp.status == 202
                created = json.load(resp)
            key = created["id"]
            assert created["source_sha256"] == expected_sha256
            assert created["case_id"] == "case-alley-01"
            assert created["camera_label"] == "Cam-North-Dock"
            assert created["clock_offset_seconds"] == 30.0

            # Wait briefly for fake worker thread completion.
            for _ in range(50):
                with urlopen(base + f"/api/jobs/{key}") as resp:
                    state = json.load(resp)
                if state["status"] == "completed":
                    break
                import time

                time.sleep(0.05)
            assert state["status"] == "completed"

            # 3. Calibrate DVR clock offset via PATCH metadata.
            with urlopen(
                Request(
                    base + f"/api/jobs/{key}/metadata",
                    data=json.dumps({"clock_offset_seconds": 45.0}).encode(),
                    method="PATCH",
                    headers={"Origin": base, "Content-Type": "application/json"},
                )
            ) as resp:
                patched = json.load(resp)
                assert patched["clock_offset_seconds"] == 45.0

            # 4. Pin evidence bookmark on frame 3 -> verify wall clock calculation (+45s +3s = 18:00:48).
            with urlopen(
                Request(
                    base + f"/api/jobs/{key}/bookmarks",
                    data=json.dumps(
                        {
                            "frame_index": 3,
                            "timestamp_seconds": 3.0,
                            "tag": "vehicle",
                            "caption": "Plate ABC74K <observed> near dock",
                        }
                    ).encode(),
                    headers={"Origin": base, "Content-Type": "application/json"},
                )
            ) as resp:
                assert resp.status == 201
                bm = json.load(resp)["bookmark"]
                assert bm["wall_clock"] == "2026-11-22T18:00:48+00:00"
                assert bm["case_id"] == "case-alley-01"

            # 5. Link recurring track under an investigator entity dossier.
            with urlopen(
                Request(
                    base + "/api/cases/case-alley-01/entities",
                    data=json.dumps(
                        {
                            "label": "Subject A (dark blue jacket)",
                            "kind": "person",
                            "linked_tracks": [{"job_id": key, "track_id": "track-2"}],
                            "description": "Entered from alley gate",
                        }
                    ).encode(),
                    headers={"Origin": base, "Content-Type": "application/json"},
                )
            ) as resp:
                assert resp.status == 201
                ent = json.load(resp)["entity"]

            # 6. Search across case by plate, color, and alert event type.
            with urlopen(base + "/api/search?case_id=case-alley-01&plate=74K") as resp:
                hits_plate = json.load(resp)["results"]
                assert len(hits_plate) == 1
                assert hits_plate[0]["preview_url"] == f"/jobs/{key}/frames/frame-3.jpg"
                assert hits_plate[0]["review_url"] == f"/jobs/{key}/review#frame=3"
                assert hits_plate[0]["wall_clock"] == "2026-11-22T18:00:48+00:00"

            with urlopen(base + "/api/search?color=blue") as resp:
                hits_color = json.load(resp)["results"]
                assert any("dark blue jacket" in h["summary"] for h in hits_color)

            with urlopen(base + "/api/search?event_type=unattended") as resp:
                hits_event = json.load(resp)["results"]
                assert any("possible_unattended_bag" in h["summary"] for h in hits_event)

            # 7. Export Printable Case Evidence Report and verify XSS escaping & CSP.
            with urlopen(base + "/cases/case-alley-01/report") as resp:
                assert "Content-Security-Policy" in resp.headers
                report_html = resp.read().decode("utf-8")
                assert "<script>alert(1)</script>" not in report_html
                assert "&lt;script&gt;alert(1)&lt;/script&gt;" in report_html
                assert expected_sha256 in report_html
                assert "Subject A (dark blue jacket)" in report_html
                assert "2026-11-22T18:00:48+00:00" in report_html

            # 8. Audit trail verified intact across all custody actions.
            with urlopen(base + "/api/audit") as resp:
                audit = json.load(resp)
                assert audit["chain_valid"] is True
                actions = [entry["action"] for entry in audit["entries"]]
                assert "case_created" in actions
                assert "media_ingested" in actions
                assert "bookmark_added" in actions
                assert "entity_updated" in actions
                assert "report_exported" in actions

            # 9. Clean deletions and case unassign on delete.
            with urlopen(
                Request(base + f"/api/bookmarks/{bm['id']}", method="DELETE", headers={"Origin": base})
            ) as resp:
                assert json.load(resp)["deleted"] == bm["id"]
            with urlopen(
                Request(base + f"/api/entities/{ent['id']}", method="DELETE", headers={"Origin": base})
            ) as resp:
                assert json.load(resp)["deleted"] == ent["id"]
            with urlopen(
                Request(base + "/api/cases/case-alley-01", method="DELETE", headers={"Origin": base})
            ) as resp:
                assert json.load(resp)["deleted"] == "case-alley-01"
            with urlopen(base + f"/api/jobs/{key}") as resp:
                assert json.load(resp)["case_id"] is None
            with pytest.raises(HTTPError) as err:
                urlopen(base + "/api/cases/case-alley-01")
            assert err.value.code == 404
        finally:
            server.shutdown()
            thread.join(timeout=5)

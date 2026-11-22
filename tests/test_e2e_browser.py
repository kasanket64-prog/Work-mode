"""Full-stack browser JS checks and end-to-end HTTP user journey tests."""

import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from threading import Thread
from urllib.request import Request, urlopen

import pytest
from PIL import Image

import camera_system.app as app_module
from camera_system.demo import make_demo

TEMPLATES = Path("src/camera_system/templates")


def _extract_review_script() -> str:
    html = (TEMPLATES / "review.html").read_text(encoding="utf-8")
    match = re.search(r"<script>\r?\n?(.*?)</script>", html, re.DOTALL)
    assert match is not None, "Expected inline <script> in review.html"
    return match.group(1)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_frontend_javascript_syntax_and_logic(tmp_path):
    ws_js = TEMPLATES / "workspace.js"
    review_js_path = tmp_path / "review-inline.js"
    review_script = _extract_review_script()
    review_js_path.write_text(review_script, encoding="utf-8")

    subprocess.run(["node", "--check", str(ws_js)], check=True)
    subprocess.run(["node", "--check", str(review_js_path)], check=True)

    ws_text = ws_js.read_text(encoding="utf-8")
    helpers_ws = (
        ws_text[ws_text.index("function uploadError") : ws_text.index("function message")]
        + "\n"
        + ws_text[ws_text.index("function matchesJob") : ws_text.index("function applyFilters")]
        + """
        if (!matchesJob({id:'abc', status:'completed', mode:'scene', created:'2026-10-10T00:00:00Z'}, '', 'completed')) process.exit(1);
        if (matchesJob({id:'abc', status:'failed', mode:'scene', created:'2026-10-10T00:00:00Z'}, '', 'completed')) process.exit(2);
        if (uploadError({name:'clip.exe', size:10}, 'scene') === '') process.exit(3);
        if (uploadError({name:'clip.mp4', size:10}, 'scene') !== '') process.exit(4);
        """
    )
    subprocess.run(["node", "-e", helpers_ws], check=True)

    link_fn = review_script[
        review_script.index("function linkEventFrames") : review_script.index("function load")
    ]
    draft_fn = review_script[
        review_script.index("function annotationDraft") : review_script.index("function downloadJSON")
    ]
    helpers_rev = (
        link_fn
        + "\n"
        + draft_fn
        + """
        const rows = [
          {type:'session', relation_endpoint_kind:'track_id'},
          {type:'frame', frame_index:0, timestamp_seconds:0, regions:[{track_id:'track-1'}], events:[]},
          {type:'frame', frame_index:1, timestamp_seconds:3, regions:[{track_id:'track-1'}], events:[{type:'possible_unattended_bag', object_track_id:'track-1', started_seconds:1, timestamp_seconds:3}]}
        ];
        const linked = linkEventFrames(rows);
        if (!linked[2].events[0].review_frame_indices.includes(1)) process.exit(1);
        const draft = annotationDraft(rows);
        if (draft.evaluation_interval_seconds[1] !== 3 || draft.duration_seconds !== null || draft.events !== null) process.exit(2);
        """
    )
    subprocess.run(["node", "-e", helpers_rev], check=True)


def test_frontend_templates_contain_tier3_studios():
    ws_html = (TEMPLATES / "workspace.html").read_text(encoding="utf-8")
    ws_js = (TEMPLATES / "workspace.js").read_text(encoding="utf-8")
    rev_html = (TEMPLATES / "review.html").read_text(encoding="utf-8")
    assert 'id="system-status"' in ws_html
    assert 'id="stride"' in ws_html and 'id="profile"' in ws_html
    assert 'id="device"' in ws_html and 'id="vehicle-crops"' in ws_html
    assert "EventSource('/api/events')" in ws_js
    assert "Draw zone on canvas" in rev_html
    assert "'annotationStudio'" in rev_html
    assert "/api/zones" in rev_html
    assert "/replay" in rev_html
    assert "/annotations" in rev_html
    assert "/evaluate" in rev_html
    assert "/verdicts" in rev_html


def test_full_http_e2e_user_journey(tmp_path, monkeypatch):
    demo_rows, truth = make_demo()
    mp4_bytes = b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2mp41" + b"\x00" * 64

    class FakeProc:
        def __init__(self, cmd):
            out_path = Path(cmd[cmd.index("--output") + 1])
            preview_dir = (
                Path(cmd[cmd.index("--preview-dir") + 1]) if "--preview-dir" in cmd else None
            )
            if preview_dir is not None:
                preview_dir.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (32, 32), (10, 80, 120)).save(preview_dir / "frame-0.jpg", "JPEG")
            rows = [dict(r) for r in demo_rows]
            for r in rows:
                if r.get("type") == "frame" and r["frame_index"] == 0 and preview_dir is not None:
                    r["preview_path"] = "frame-0.jpg"
            out_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            self.pid = 424242
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
            with urlopen(base + "/") as resp:
                assert b"Fieldnote" in resp.read()
            with urlopen(base + "/assets/workspace.js") as resp:
                assert b"EventSource" in resp.read()
            with urlopen(base + "/api/system/status") as resp:
                status = json.load(resp)
                assert status["job_count"] == 0

            with urlopen(base + "/api/events", timeout=5) as sse:
                line1 = sse.readline().decode("utf-8")
                assert line1.startswith("event: jobs")

            upload_req = Request(
                base + "/api/jobs",
                data=mp4_bytes,
                headers={
                    "Origin": base,
                    "X-File-Extension": ".mp4",
                    "X-Analysis-Mode": "scene",
                    "X-Frame-Limit": "36",
                    "X-Frame-Stride": "15",
                    "X-Reasoning-Profile": "behavior",
                    "X-Device": "cpu",
                    "X-Vehicle-Crops": "0",
                },
            )
            with urlopen(upload_req) as resp:
                assert resp.status == 202
                created = json.load(resp)
            key = created["id"]

            deadline = time.monotonic() + 5.0
            job_state = None
            while time.monotonic() < deadline:
                with urlopen(base + f"/api/jobs/{key}") as resp:
                    job_state = json.load(resp)
                if job_state["status"] == "completed":
                    break
                time.sleep(0.05)
            assert job_state is not None and job_state["status"] == "completed"

            with urlopen(base + f"/jobs/{key}/frames/frame-0.jpg") as resp:
                assert resp.read().startswith(b"\xff\xd8\xff")

            with urlopen(base + f"/jobs/{key}/review") as resp:
                review_html = resp.read().decode("utf-8")
                assert "possible_unattended_bag" in review_html

            verdict_id = next(
                e["event_id"]
                for r in demo_rows
                if r.get("type") == "frame"
                for e in r.get("events", [])
            )
            with urlopen(
                Request(
                    base + f"/api/jobs/{key}/verdicts",
                    data=json.dumps(
                        {"verdicts": {verdict_id: {"verdict": "correct", "note": "confirmed"}}}
                    ).encode(),
                    method="PUT",
                    headers={"Origin": base, "Content-Type": "application/json"},
                )
            ) as resp:
                assert json.load(resp)["verdicts"][verdict_id]["verdict"] == "correct"

            with urlopen(
                Request(
                    base + "/api/zones",
                    data=json.dumps(
                        {"id": "lobby", "box": [0.1, 0.1, 0.9, 0.9], "labels": ["backpack"]}
                    ).encode(),
                    headers={"Origin": base, "Content-Type": "application/json"},
                )
            ) as resp:
                assert json.load(resp)["zone"]["id"] == "lobby"

            with urlopen(
                Request(
                    base + f"/api/jobs/{key}/replay",
                    data=json.dumps({"dwell_seconds": 0.0, "use_saved_zones": True}).encode(),
                    headers={"Origin": base, "Content-Type": "application/json"},
                )
            ) as resp:
                assert json.load(resp)["status"] == "replayed"

            with urlopen(base + f"/jobs/{key}/review") as resp:
                assert '"review_verdict": "correct"' in resp.read().decode("utf-8")

            with urlopen(base + f"/api/jobs/{key}/annotations") as resp:
                draft = json.load(resp)
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
                assert json.load(resp)["events"]["f1"] == 1.0

            with urlopen(base + f"/api/jobs/{key}/evaluations") as resp:
                assert len(json.load(resp)["evaluations"]) == 1

            with urlopen(
                Request(base + f"/api/jobs/{key}", method="DELETE", headers={"Origin": base})
            ) as resp:
                assert json.load(resp)["deleted"] == key
            with urlopen(
                Request(base + "/api/zones/lobby", method="DELETE", headers={"Origin": base})
            ) as resp:
                assert json.load(resp)["deleted"] == "lobby"
        finally:
            server.shutdown()
            thread.join(timeout=5)

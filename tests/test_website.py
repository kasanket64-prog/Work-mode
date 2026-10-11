"""Public demo assets and private media playback keep the local boundary intact."""

import hashlib
import json
import shutil
import subprocess
from importlib.resources import files
from pathlib import Path
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from camera_system.app import make_app
from camera_system.demo import make_demo


@pytest.fixture
def website(tmp_path):
    with make_app(tmp_path, Path.cwd(), 0) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}", tmp_path
        finally:
            server.shutdown()
            thread.join()


def test_demo_assets_are_packaged_real_result_and_allowlisted(website):
    base, _ = website
    with urlopen(base) as response:
        assert b"More than watching." in response.read()
    with urlopen(base + "/assets/demo-result.json") as response:
        result = json.load(response)
    with urlopen(base + "/assets/demo-horse.jpg") as response:
        assert response.headers["Content-Type"] == "image/jpeg"
        assert hashlib.sha256(response.read()).hexdigest() == result["image_sha256"]
    assert result["region_source"] == "manual"
    assert result["relations"][0]["predicate"] == "riding"
    for name, mime in [("workspace.js", "text/javascript"), ("workspace.css", "text/css")]:
        with urlopen(base + "/assets/" + name) as response:
            assert response.headers["Content-Type"].startswith(mime)
            assert response.headers["X-Content-Type-Options"] == "nosniff"
    for path in ["/assets/../app.py", "/assets/camera_sources.json", "/assets/.env"]:
        with pytest.raises(HTTPError) as error:
            urlopen(base + path)
        assert error.value.code == 404


def test_media_ranges_and_current_review(website):
    base, root = website
    key = "a" * 32
    folder = root / key
    folder.mkdir()
    content = b"0123456789"
    (folder / "input.mp4").write_bytes(content)
    rows, _ = make_demo()
    (folder / "result.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    (folder / "review.html").write_text("old template", encoding="utf-8")
    media = base + f"/jobs/{key}/media"
    with urlopen(media) as response:
        assert response.read() == content
        assert response.headers["Accept-Ranges"] == "bytes"
        assert response.headers["Cache-Control"] == "no-store"
    for value, expected in [("bytes=2-5", b"2345"), ("bytes=7-", b"789"),
                            ("bytes=-3", b"789"), ("bytes=8-99", b"89")]:
        with urlopen(Request(media, headers={"Range": value})) as response:
            assert response.status == 206
            assert response.read() == expected
            assert response.headers["Content-Range"].endswith("/10")
    for value, status in [("bytes=50-", 416), ("bytes=5-2", 416), ("bytes=-0", 416),
                          ("bytes=0-1,3-4", 400), ("bytes=-", 400),
                          ("bytes=" + "9" * 30 + "-", 400)]:
        with pytest.raises(HTTPError) as error:
            urlopen(Request(media, headers={"Range": value}))
        assert error.value.code == status
    with pytest.raises(HTTPError) as error:
        urlopen(Request(media, headers={"Host": "foreign.example"}))
    assert error.value.code == 403
    with urlopen(base + f"/jobs/{key}/review") as response:
        page = response.read().decode()
        assert "A closer look." in page and "Watch the original recording" in page
        assert "__DATA__" not in page
    assert (folder / "review.html").read_text() == "old template"
    (folder / "result.jsonl").write_text("invalid json", encoding="utf-8")
    with pytest.raises(HTTPError) as error:
        urlopen(base + f"/jobs/{key}/review")
    assert error.value.code == 409
    with pytest.raises(HTTPError) as error:
        urlopen(base + "/jobs/" + "b" * 32 + "/media")
    assert error.value.code == 404


def test_upload_validation_in_browser_script():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed for browser logic checks")
    script = files("camera_system").joinpath("templates/workspace.js").read_text(encoding="utf-8")
    function = "function uploadError" + script.split("function uploadError", 1)[1].split(
        "function message", 1
    )[0]
    checks = """
const assert = require('node:assert/strict');
assert.equal(uploadError({name:'clip.MP4',size:512*1024*1024}, 'scene'), '');
assert.equal(uploadError({name:'image.png',size:1}, 'scene'), '');
assert.match(uploadError({name:'image.jpg',size:10}, 'plates'), /needs a video/);
assert.match(uploadError({name:'clip.mp4',size:512*1024*1024+1}, 'scene'), /too large/);
assert.match(uploadError({name:'clip.exe',size:10}, 'scene'), /Choose an/);
assert.match(uploadError({name:'clip.mp4',size:0}, 'scene'), /not empty/);
assert.match(uploadError(null, 'scene'), /Choose a/);
"""
    subprocess.run([node, "-e", function + checks], check=True, timeout=10)

def test_workspace_search_and_status_filters():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed for browser logic checks")
    script = files("camera_system").joinpath("templates/workspace.js").read_text(encoding="utf-8")
    function = "function matchesJob" + script.split("function matchesJob", 1)[1].split("function applyFilters", 1)[0]
    checks = """
const assert = require('node:assert/strict');
const job = {id:'abc123', mode:'scene', status:'completed', created:'2026-10-11T12:00:00Z'};
assert(matchesJob(job, ' SCENE ', 'all'));
assert(matchesJob(job, 'abc123', 'completed'));
assert(matchesJob(job, '2026-10-11', 'all'));
assert(!matchesJob(job, 'plate', 'all'));
assert(!matchesJob(job, '', 'active'));
assert(matchesJob({...job,status:'cancelling'}, '', 'active'));
for (const status of ['failed','cancelled','interrupted']) assert(matchesJob({...job,status}, '', 'attention'));
assert(matchesJob({...job,mode:'plates'}, 'Plate reading', 'all'));
assert(!matchesJob({...job,status:'unknown'}, '', 'attention'));
"""
    subprocess.run([node, "-e", function + checks], check=True, timeout=10)

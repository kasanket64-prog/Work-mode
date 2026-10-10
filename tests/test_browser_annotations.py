"""Keep offline browser drafts compatible with the shared Python evaluator."""

import json
import shutil
import subprocess

import pytest

from camera_system.demo import make_demo
from camera_system.review import HTML, annotation_template

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="Node.js is needed for browser logic checks")


def browser_draft(rows):
    function = "function annotationDraft" + HTML.split("function annotationDraft", 1)[1].split(
        "function downloadJSON", 1
    )[0]
    script = function + """
const rows=JSON.parse(require('node:fs').readFileSync(0,'utf8'));
try { console.log(JSON.stringify({draft:annotationDraft(rows)})); }
catch(error) { console.log(JSON.stringify({error:error.message})); }
"""
    run = subprocess.run(
        [NODE, "-e", script], input=json.dumps(rows), text=True, encoding="utf-8",
        capture_output=True, check=True, timeout=10,
    )
    return json.loads(run.stdout)


def test_browser_draft_matches_cli_without_predictions_or_verdicts():
    rows, _ = make_demo()
    event = next(e for row in rows for e in row.get("events", []))
    event["review_verdict"] = "correct"
    original = json.dumps(rows)
    draft = browser_draft(rows)["draft"]
    assert draft == annotation_template(rows)
    assert draft["events"] is None and draft["duration_seconds"] is None
    assert "review_verdict" not in json.dumps(draft)
    assert json.dumps(rows) == original


@pytest.mark.parametrize("invalid", [None, True, -1, "1"])
def test_browser_rejects_invalid_timestamps(invalid):
    rows, _ = make_demo()
    rows[1]["timestamp_seconds"] = invalid
    assert "timestamps" in browser_draft(rows)["error"]


def test_browser_rejects_images_and_empty_video():
    rows, _ = make_demo()
    assert "tracked video" in browser_draft([{"type": "session", "source_kind": "image"}])["error"]
    assert "at least one" in browser_draft([rows[0]])["error"]


def test_browser_draft_tracks_newly_loaded_session():
    rows, _ = make_demo()
    other = [rows[0], dict(rows[1], frame_index=120, timestamp_seconds=4,
                          regions=[{"track_id": "new"}, {"track_id": "new"}])]
    assert browser_draft(other)["draft"] == annotation_template(other)

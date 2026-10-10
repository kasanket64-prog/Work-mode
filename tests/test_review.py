import json
from threading import Thread
from urllib.request import urlopen

import pytest

from camera_system.demo import make_demo
from camera_system.review import annotation_template, evaluate, main, make_server, read_records


def test_annotation_draft_cli_and_review_gate(tmp_path):
    rows, truth = make_demo()
    session = tmp_path / "session.jsonl"
    original = "\n".join(json.dumps(row) for row in rows)
    session.write_text(original, encoding="utf-8")
    output = tmp_path / "draft.json"
    args = [str(session), "--annotation-template", "--output", str(output)]
    assert main(args) == 0
    draft = json.loads(output.read_text(encoding="utf-8"))
    assert draft["events"] is None and draft["duration_seconds"] is None
    assert draft["track_mapping"] == {}
    assert draft["reference_frames"][0] == {
        "frame_index": 0, "timestamp_seconds": 0.0,
        "visible_track_ids": ["track-1", "track-2"],
    }
    assert len(draft["reference_frames"]) == 36
    assert "relations" not in draft and "confirmed_relations" not in json.dumps(draft)
    with pytest.raises(ValueError, match="reviewed"):
        evaluate(rows, dict(draft, **truth))
    assert evaluate(rows, dict(draft, **truth, annotation_status="reviewed")) == evaluate(rows, truth)
    assert main(args) == 2
    assert json.loads(output.read_text(encoding="utf-8")) == draft
    assert session.read_text(encoding="utf-8") == original
    assert main([*args, "--serve"]) == 2


def test_annotation_draft_requires_tracked_video_frames():
    rows, _ = make_demo()
    with pytest.raises(ValueError, match="tracked video"):
        annotation_template([{"type": "session", "source_kind": "image"}])
    with pytest.raises(ValueError, match="at least one"):
        annotation_template([rows[0]])
    for invalid in (None, True, -1, float("nan")):
        with pytest.raises(ValueError, match="timestamps"):
            annotation_template([rows[0], dict(rows[1], timestamp_seconds=invalid)])


def test_simulation_and_html_escape(tmp_path):
    rows, truth = make_demo()
    events = [e for r in rows for e in r.get("events", [])]
    assert len(events) == 1 and events[0]["timestamp_seconds"] == 33
    assert evaluate(rows, truth)["events"]["true_positive"] == 1
    rows[0]["note"] = "</script><script>alert('bad')</script>"
    session = tmp_path / "session.jsonl"
    session.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    output = tmp_path / "review.html"
    assert main([str(session), "--output", str(output)]) == 0
    html = output.read_text(encoding="utf-8")
    assert "<script>alert('bad')" not in html
    assert "\\u003c/script>" in html
    assert main([str(session), "--output", str(output)]) == 2


def test_read_only_local_api():
    rows, _ = make_demo()
    with make_server(rows, 0) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with urlopen(
                f"http://127.0.0.1:{server.server_port}/api/session", timeout=5
            ) as response:
                assert json.load(response) == json.loads(json.dumps(rows))
        finally:
            server.shutdown()
            thread.join(timeout=5)


def test_evaluation_rejects_unannotated_frames_and_duration():
    rows, truth = make_demo()
    with pytest.raises(ValueError):
        evaluate(rows, dict(truth, duration_seconds=0))
    with pytest.raises(ValueError):
        evaluate(rows, dict(truth, annotated_frame_indices=[999], relations=[]))


@pytest.mark.parametrize("annotation", [
    {"relations": []},
    {"annotated_frame_indices": [0]},
    {"annotated_frame_indices": [], "relations": []},
    {"annotated_frame_indices": [True], "relations": []},
    {"annotated_frame_indices": [0.0], "relations": []},
    {"annotated_frame_indices": [-1], "relations": []},
])
def test_relation_evaluation_requires_explicit_frame_scope(annotation):
    rows, truth = make_demo()
    with pytest.raises(ValueError, match="annotated_frame_indices"):
        evaluate(rows, dict(truth, **annotation))


def test_evaluation_rejects_empty_session_but_accepts_reviewed_negative_frame():
    rows, truth = make_demo()
    with pytest.raises(ValueError, match="at least one"):
        evaluate([rows[0], rows[-1]], truth)
    result = evaluate(rows, dict(truth, annotated_frame_indices=[35], relations=[]))
    assert result["confirmed_relations"] == {
        "true_positive": 0, "false_positive": 0, "false_negative": 0,
        "precision": None, "recall": None, "f1": None,
    }


def test_review_verdict_survives_export_reload_without_changing_evaluation(tmp_path):
    rows, truth = make_demo()
    before = evaluate(rows, truth)
    event = next(e for row in rows for e in row.get("events", []))
    event["review_verdict"] = "incorrect"
    exported = tmp_path / "camera-session.json"
    exported.write_text(json.dumps(rows), encoding="utf-8")
    loaded = read_records(exported)
    restored = next(e for row in loaded for e in row.get("events", []))
    assert restored["review_verdict"] == "incorrect"
    assert evaluate(loaded, truth) == before

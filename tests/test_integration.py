"""Opt-in real inference test. No fallback to a mock model."""

import os
from pathlib import Path

import pytest

from camera_system.cli import main


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("RUN_MODEL_TESTS") != "1",
    reason="Set RUN_MODEL_TESTS=1 after installing inference dependencies",
)
def test_real_sample(tmp_path):
    import json
    import math

    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "relations.json"
    assert (
        main(
            [
                "--image",
                str(root / "examples/horse.jpg"),
                "--regions",
                str(root / "examples/horse.regions.json"),
                "--config",
                str(root / "configs/phase1.toml"),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    result = json.loads(output.read_text())
    assert result["status"] == "predicted"
    assert result["relations"], "Sample should produce at least one ranked relationship"
    ids = {r["id"] for r in result["regions"]}
    for relation in result["relations"]:
        assert relation["subject_id"] in ids and relation["object_id"] in ids
        assert relation["subject_id"] != relation["object_id"]
        assert relation["predicate"] in result["settings"]["vocabulary"]
        assert math.isfinite(relation["score"]) and 0 <= relation["score"] <= 1


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("RUN_DETECTOR_TESTS") != "1",
    reason="Set RUN_DETECTOR_TESTS=1 after installing detection dependencies",
)
def test_real_automatic_sample(tmp_path):
    import json
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "automatic.json"
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "camera_system.cli",
            "--image",
            "examples/horse.jpg",
            "--detect",
            "--output",
            str(output),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
    )
    assert run.returncode == 0, run.stderr
    result = json.loads(run.stdout)
    assert result == json.loads(output.read_text(encoding="utf-8"))
    assert result["region_source"] == "detector"
    assert {r["label"] for r in result["regions"]} >= {"person", "horse"}
    assert len(result["detector"]["weights_sha256"]) == 64
    ids = {r["id"] for r in result["regions"]}
    assert result["relations"]
    for relation in result["relations"]:
        assert relation["subject_id"] in ids and relation["object_id"] in ids
        assert 0 <= relation["score"] <= 1


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("RUN_VIDEO_TESTS") != "1",
    reason="Set RUN_VIDEO_TESTS=1 for real video decoding and both models",
)
def test_real_video(tmp_path, monkeypatch):
    import json

    import cv2

    from camera_system.video_cli import main as video_main

    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root)
    image = cv2.imread(str(root / "examples/horse.jpg"))
    video = tmp_path / "smoke.avi"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"MJPG"), 5, (image.shape[1], image.shape[0])
    )
    assert writer.isOpened()
    try:
        for _ in range(6):
            writer.write(image)
    finally:
        writer.release()
    output = tmp_path / "video.jsonl"
    person_flags = (
        ["--analyze-people", "--include-preview"]
        if os.environ.get("RUN_PERSON_TESTS") == "1"
        else []
    )
    assert (
        video_main(
            ["--video", str(video), "--stride", "2", "--max-frames", "10", "--output", str(output)]
            + person_flags
        )
        == 0
    )
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    frames = [row for row in rows if row["type"] == "frame"]
    assert [row["frame_index"] for row in frames] == [0, 2, 4]
    assert [row["timestamp_seconds"] for row in frames] == pytest.approx([0, 0.4, 0.8])
    assert all(row["relations"] for row in frames)
    first_tracks = {r["label"]: r["track_id"] for r in frames[0]["regions"]}
    for frame in frames:
        assert {r["label"]: r["track_id"] for r in frame["regions"]} == first_tracks
        ids = {r["track_id"] for r in frame["regions"]}
        assert all(r["subject_id"] in ids and r["object_id"] in ids for r in frame["relations"])
        assert not frame["events"]
    assert frames[-1]["confirmed_relations"]
    if person_flags:
        assert all(f["person_analysis"] for f in frames)
        assert all(f["preview_data_url"].startswith("data:image/jpeg;base64,") for f in frames)
        assert frames[0]["person_analysis"][0]["motion"]["status"] == "unknown"
        assert frames[-1]["person_analysis"][0]["motion"]["status"] == "estimated"
    assert rows[-1]["frames_read"] == 6
    assert rows[-1]["frames_skipped_by_stride"] == 3
    assert rows[-1]["end_reason"] == "video_eof_or_decode_failure"


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("RUN_PERSON_TESTS") != "1",
    reason="Set RUN_PERSON_TESTS=1 for actual pose inference",
)
def test_real_person_analysis():
    from PIL import Image

    from camera_system.contracts import Region
    from camera_system.people import PersonAnalyzer, PoseEstimator

    root = Path(__file__).resolve().parents[1]
    with Image.open(root / "examples/horse.jpg") as source:
        image = source.convert("RGB")
    estimator = PoseEstimator("cpu")
    analyzer = PersonAnalyzer(estimator)
    observations = analyzer.analyze(image, [Region("person-1", (470, 130, 650, 630), "person")], [])
    assert len(observations) == 1
    assert len(observations[0]["keypoints"]) == 17
    assert observations[0]["mood"]["status"] == "not_inferred"
    assert observations[0]["motion"]["status"] == "unknown"
    assert len(estimator.metadata["weights_sha256"]) == 64

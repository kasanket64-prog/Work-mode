"""Verify rate comparison aligns frames and measures known synthetic fragmentation."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from camera_system.contracts import Region


@pytest.mark.parametrize("start_frame", [0, 5])
def test_dense_comparison_preserves_shared_detections(tmp_path, monkeypatch, start_frame):
    script = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_tracking_rate.py"
    spec = importlib.util.spec_from_file_location("tracking_rate", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    video = tmp_path / "synthetic.mp4"
    video.touch()
    output = tmp_path / "result"
    image = Image.new("RGB", (256, 100))

    class Source:
        fps = 25

        def __init__(self, path):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def __iter__(self):
            for i in range(67 + start_frame):
                yield SimpleNamespace(index=i, timestamp_seconds=i / 25, image=image)

    class Detector:
        weights_sha256 = "synthetic"

        def __init__(self, *args):
            self.index = 0

        def detect(self, image):
            x = self.index * 3
            self.index += 1
            return [Region("det-0", (x, 0, x + 10, 10), "car", detection_score=0.9)]

    monkeypatch.setattr(module, "FrameSource", Source)
    monkeypatch.setattr(module, "YoloDetector", Detector)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--video",
            str(video),
            "--output",
            str(output),
            "--start-frame",
            str(start_frame),
        ],
    )
    module.main()
    report = json.loads((output / "comparison.json").read_text())
    assert report["decoded_frames"] == 67
    frames = [
        json.loads(line)
        for line in (output / "dense.jsonl").read_text().splitlines()
        if json.loads(line)["type"] == "frame"
    ]
    assert [r["frame_index"] for r in frames] == list(range(start_frame, start_frame + 67, 6))
    assert report["modes"]["dense"]["shared_frames"] == 12
    assert report["modes"]["dense"]["shared_frame_id_reuse"] == 1
    assert report["modes"]["sparse"]["shared_frame_id_reuse"] == 0

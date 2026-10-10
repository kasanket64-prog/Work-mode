import json
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from camera_system.cli import main
from camera_system.contracts import Region, parse_regions
from camera_system.detection import DetectorSettings, decode_detections


def test_decode_clips_sorts_caps_and_keeps_confidence():
    regions = decode_detections(
        [[1, 2, 20, 30], [-5, 0, 120, 110], [0, 0, 0, 4], [0, 0, np.nan, 5]],
        [0.4, 0.9, 0.8, 0.99],
        [0, 1, 0, 0],
        {0: "person", 1: "horse"},
        (100, 100),
        DetectorSettings(max_detections=1),
    )
    assert regions == [Region("det-0", (0.0, 0.0, 100.0, 100.0), "horse", detection_score=0.9)]


def test_empty_and_low_confidence():
    config = DetectorSettings()
    assert decode_detections(np.empty((0, 4)), [], [], {}, (100, 100), config) == []
    assert decode_detections([[0, 0, 10, 10]], [0.1], [0], {0: "person"}, (100, 100), config) == []


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(confidence=-1),
        dict(iou=float("nan")),
        dict(max_detections=0),
        dict(image_size=True),
        dict(model=""),
    ],
)
def test_bad_config(kwargs):
    with pytest.raises(ValueError):
        DetectorSettings(**kwargs)


def test_bad_shapes():
    with pytest.raises(ValueError):
        decode_detections([[0, 0, 10, 10]], [], [], {}, (100, 100), DetectorSettings())


@pytest.mark.parametrize("count", [0, 1, 2])
def test_automatic_cli(tmp_path, monkeypatch, capsys, count):
    image = tmp_path / "image.png"
    Image.new("RGB", (100, 100)).save(image)
    config = tmp_path / "relation.toml"
    config.write_text("max_regions = 2\n")
    detector_config = tmp_path / "detector.toml"
    detector_config.write_text("max_detections = 10\n")
    detector = Mock()
    detector.return_value.device = "cpu"
    detector.return_value.weights_sha256 = "test-hash"
    detector.return_value.detect.return_value = [
        Region(f"det-{i}", (0.0, 0.0, 50.0, 50.0), "person", detection_score=0.8)
        for i in range(count)
    ]
    predictor = Mock()
    predictor.return_value.device = "cpu"
    predictor.return_value.predict.return_value = []
    predictor.return_value.rejected_relations = []
    monkeypatch.setattr("camera_system.cli.YoloDetector", detector)
    monkeypatch.setattr("camera_system.cli.RelateAnythingPredictor", predictor)
    args = [
        "--image",
        str(image),
        "--detect",
        "--config",
        str(config),
        "--detector-config",
        str(detector_config),
    ]
    assert main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["region_source"] == "detector"
    assert len(result["regions"]) == count
    assert detector.call_args.args[0].max_detections == 2
    if count < 2:
        predictor.assert_not_called()
        assert result["relation_skip_reason"] == "fewer_than_two_regions"
    else:
        predictor.assert_called_once()
        assert len(parse_regions(result["regions"], (100, 100))) == 2
    detector.reset_mock()
    assert main(args + ["--validate-only"]) == 2
    detector.assert_not_called()

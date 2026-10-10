from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from camera_system.contracts import Region, Relation
from camera_system.people import PersonAnalyzer, pose_description, preview_image, upper_body_color
from camera_system.review import read_records


def test_color_and_small_region():
    image = Image.new("RGB", (100, 200), "blue")
    assert upper_body_color(image, (0, 0, 100, 200))["value"] == "blue"
    assert upper_body_color(image, (0, 0, 10, 20))["status"] == "unknown"
    assert preview_image(image).startswith("data:image/jpeg;base64,")


def test_pose_requires_visible_landmarks():
    points = np.zeros((17, 2))
    for i, xy in {
        5: (30, 20),
        6: (60, 20),
        11: (30, 70),
        12: (60, 70),
        13: (30, 100),
        14: (60, 100),
        15: (30, 140),
        16: (60, 140),
    }.items():
        points[i] = xy
    assert pose_description(points, np.ones(17))["value"] == "Upright body with straight legs"
    scores = np.ones(17)
    scores[15] = 0.1
    assert pose_description(points, scores)["status"] == "unknown"


def test_person_analysis_motion_and_interactions():
    image = Image.new("RGB", (200, 200), "blue")
    model = Mock()
    model.predict.return_value = []
    analyzer = PersonAnalyzer(model)
    regions = [
        Region("p", (10, 10, 50, 100), "person", "track-1"),
        Region("b", (20, 20, 50, 40), "bag", "track-2"),
    ]
    relations = [Relation("track-1", "holding", "track-2", 0.8)]
    first = analyzer.analyze(image, regions, relations, 0)[0]
    assert first["motion"]["status"] == "unknown"
    assert first["interactions"][0]["predicate"] == "holding"
    moved = [Region("p", (30, 10, 70, 100), "person", "track-1")]
    second = analyzer.analyze(image, moved, [], 1)[0]
    assert second["motion"]["value"] == "Moving right in the image"
    assert second["mood"]["status"] == "not_inferred"
    assert analyzer.analyze(image, moved, [], 10)[0]["motion"]["status"] == "unknown"
    with pytest.raises(ValueError):
        analyzer.analyze(image, moved, [], 9)


def test_no_people_no_pose_call_and_no_image_motion():
    model = Mock()
    analyzer = PersonAnalyzer(model)
    image = Image.new("RGB", (100, 100))
    assert analyzer.analyze(image, [], []) == []
    model.predict.assert_not_called()
    model.predict.return_value = []
    observation = analyzer.analyze(image, [Region("p", (0, 0, 100, 100), "person")], [])[0]
    assert observation["motion"]["status"] == "unknown"


def test_review_accepts_image_json(tmp_path):
    import json

    path = tmp_path / "image.json"
    path.write_text(json.dumps({"status": "predicted", "regions": [], "image_size": [100, 100]}))
    rows = read_records(path)
    assert rows[0]["source_kind"] == "image"
    assert rows[1]["type"] == "frame"


def test_motion_window_at_video_rate_and_discontinuities():
    model = Mock()
    model.predict.return_value = []
    analyzer = PersonAnalyzer(model)
    image = Image.new("RGB", (200, 200))
    for i in range(31):
        regions = [Region("p", (10 + i, 10, 50 + i, 100), "person", "track-1")]
        observation = analyzer.analyze(image, regions, [], i / 30)[0]
    assert observation["motion"]["value"] == "Moving right in the image"
    assert observation["motion"]["span_seconds"] == 1
    assert observation["motion"]["support_count"] == 31
    assert observation["motion"]["image_diagonals_per_second"] == pytest.approx(0.1061)
    analyzer.analyze(image, [], [], 1.1)
    assert analyzer.analyze(image, regions, [], 1.2)[0]["motion"]["status"] == "unknown"
    assert (
        analyzer.analyze(Image.new("RGB", (400, 400)), regions, [], 1.5)[0]["motion"]["status"]
        == "unknown"
    )

import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from camera_system.cli import main
from camera_system.config import Settings
from camera_system.contracts import parse_regions
from camera_system.relations import RelateAnythingPredictor


@pytest.fixture
def entries():
    return [
        dict(id="p", box=[1, 2, 10, 20], label="person"),
        dict(id="b", box=[20, 20, 30, 30], label="bag"),
    ]


@pytest.mark.parametrize(
    "box",
    [
        [0, 0, 0, 1],
        [-1, 0, 2, 3],
        [0, 0, 101, 3],
        [0, 0, float("nan"), 3],
        [True, 0, 2, 3],
        [1, 2, 3],
    ],
)
def test_invalid_boxes(entries, box):
    entries[0]["box"] = box
    with pytest.raises(ValueError):
        parse_regions(entries, (100, 100))


def test_duplicate_ids(entries):
    entries[1]["id"] = "p"
    with pytest.raises(ValueError, match="unique"):
        parse_regions(entries, (100, 100))


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(topk=0),
        dict(max_regions=1),
        dict(min_score=float("nan")),
        dict(vocabulary=[]),
        dict(vocabulary=["x", "x"]),
        dict(device="invalid"),
        dict(topk=True),
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        Settings(**kwargs)


def test_adapter_preserves_ids_and_filters(entries):
    adapter = object.__new__(RelateAnythingPredictor)
    adapter.settings = Settings(min_score=0.5)
    adapter.model = Mock()
    adapter.model.predict.return_value = [
        SimpleNamespace(subject_idx=1, object_idx=0, predicate="beside", score=0.8),
        SimpleNamespace(subject_idx=0, object_idx=1, predicate="holding", score=0.2),
    ]
    relations = adapter.predict(Image.new("RGB", (100, 100)), parse_regions(entries, (100, 100)))
    assert len(relations) == 1
    assert (relations[0].subject_id, relations[0].object_id) == ("b", "p")
    assert adapter.model.predict.call_args.args[1].dtype == np.float32
    assert adapter.model.predict.call_args.kwargs["box_labels"] == ["person", "bag"]


def test_adapter_rejects_truncation(entries):
    adapter = object.__new__(RelateAnythingPredictor)
    adapter.settings = Settings(max_regions=2)
    regions = parse_regions(entries, (100, 100))
    with pytest.raises(ValueError, match="at most"):
        adapter.predict(Image.new("RGB", (100, 100)), regions + regions)


def test_cli_validation_and_prediction(tmp_path, entries, monkeypatch, capsys):
    image = tmp_path / "image.png"
    Image.new("RGB", (100, 100)).save(image)
    boxes = tmp_path / "regions.json"
    boxes.write_text(json.dumps(entries))
    config = tmp_path / "config.toml"
    config.write_text('device = "cpu"\n')
    args = ["--image", str(image), "--regions", str(boxes), "--config", str(config)]
    fake = Mock()
    monkeypatch.setattr("camera_system.cli.RelateAnythingPredictor", fake)
    assert main(args + ["--validate-only"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "validated" and "relations" not in result
    fake.assert_not_called()
    fake.return_value.device = "cpu"
    fake.return_value.predict.return_value = []
    fake.return_value.rejected_relations = []
    output = tmp_path / "nested" / "result.json"
    assert main(args + ["--output", str(output)]) == 0
    result = json.loads(output.read_text())
    assert result["status"] == "predicted" and result["relations"] == []
    assert main(args + ["--output", str(boxes)]) == 2
    assert json.loads(boxes.read_text()) == entries
    entries[0]["box"] = [0, 0, 500, 500]
    boxes.write_text(json.dumps(entries))
    fake.reset_mock()
    assert main(args) == 2
    fake.assert_not_called()


def test_cli_unknown_config_fails_before_model(tmp_path, monkeypatch, capsys):
    config = tmp_path / "bad.toml"
    config.write_text("typo = 3\n")
    fake = Mock()
    monkeypatch.setattr("camera_system.cli.RelateAnythingPredictor", fake)
    assert (
        main(["--image", "missing.jpg", "--regions", "missing.json", "--config", str(config)]) == 2
    )
    assert "unknown configuration" in capsys.readouterr().err
    fake.assert_not_called()


def test_model_loader_uses_pinned_revision(monkeypatch):
    import sys

    fake_model = Mock()
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "relsgg", SimpleNamespace(RelateAnything=fake_model))
    settings = Settings(device="auto")
    predictor = RelateAnythingPredictor(settings)
    assert predictor.device == "cpu"
    fake_model.from_pretrained.assert_called_once_with(
        settings.model_id,
        revision=settings.model_revision,
        predicates=list(settings.vocabulary),
        device="cpu",
    )
    with pytest.raises(RuntimeError, match="CUDA requested"):
        RelateAnythingPredictor(Settings(device="cuda"))

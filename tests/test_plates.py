from types import SimpleNamespace as Obj

from PIL import Image

from camera_system.plates import candidate, scan


def test_plate_candidate_requires_review_and_keeps_crop():
    detection = Obj(bounding_box=Obj(x1=-1, y1=2, x2=30, y2=15), confidence=0.9)
    result = Obj(detection=detection, ocr=Obj(text="ab-123", confidence=[0.99, 0.2]))
    image = Image.new("RGB", (40, 20))
    plate = candidate(result, image)
    assert plate["text"] == "AB123"
    assert plate["status"] == "unreadable_or_uncertain"
    assert plate["box"] == [0, 2, 30, 15]
    assert plate["crop_data_url"].startswith("data:image/jpeg;base64,")
    result.ocr.confidence = 0.99
    assert candidate(result, image)["status"] == "review_required"
    result.ocr = None
    assert candidate(result, image)["text"] == ""


def test_vehicle_crops_remap_and_deduplicate():
    from unittest.mock import Mock

    model = Mock()
    model.predict.return_value = [
        Obj(
            detection=Obj(bounding_box=Obj(x1=2, y1=3, x2=12, y2=8), confidence=0.9),
            ocr=Obj(text="AB123", confidence=0.95),
        )
    ]
    car = Obj(label="car", box=(20, 10, 80, 50))
    person = Obj(label="person", box=(0, 0, 10, 10))
    result = scan(model, Image.new("RGB", (100, 60)), [car, car, person])
    assert len(result) == 1
    assert result[0]["box"] == [22, 13, 32, 18]
    assert model.predict.call_count == 2
    assert model.predict.call_args.args[0].shape == (40, 60, 3)
    assert scan(model, Image.new("RGB", (100, 60)), []) == []

"""Optional YOLO detector; original-image coordinates and separate confidence."""

import hashlib
import tomllib
from dataclasses import dataclass, fields
from math import isfinite
from pathlib import Path

import numpy as np
from PIL import Image

from .contracts import Region


@dataclass(frozen=True)
class DetectorSettings:
    model: str = "yolo11n.pt"
    confidence: float = 0.25
    iou: float = 0.7
    image_size: int = 640
    max_detections: int = 20

    def __post_init__(self):
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("detector model must be a nonempty path or model name")
        for name in ("confidence", "iou"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"detector {name} must be in [0, 1]")
        for name in ("image_size", "max_detections"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"detector {name} must be a positive integer")


def load_detector_settings(path: Path) -> DetectorSettings:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    if set(data) - {f.name for f in fields(DetectorSettings)}:
        raise ValueError("unknown detector configuration fields")
    return DetectorSettings(**data)


def decode_detections(boxes, scores, classes, names, size, settings):
    """Clip original-pixel boxes, drop invalid rows, sort and cap before assigning IDs."""
    boxes = np.asarray(boxes, dtype=float)
    scores = np.asarray(scores, dtype=float)
    classes = np.asarray(classes, dtype=float)
    if boxes.shape != (len(scores), 4) or classes.shape != scores.shape or scores.ndim != 1:
        raise ValueError("detector returned inconsistent box/score/class shapes")
    width, height = size
    candidates = []
    for box, score, cls in zip(boxes, scores, classes, strict=True):
        if not np.isfinite(box).all() or not isfinite(score) or not isfinite(cls):
            continue
        if not settings.confidence <= score <= 1 or cls < 0 or cls != int(cls):
            continue
        clipped = np.clip(box, [0, 0, 0, 0], [width, height, width, height])
        if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
            continue
        candidates.append((float(score), tuple(clipped.tolist()), str(names[int(cls)])))
    candidates.sort(key=lambda row: (-row[0], row[1], row[2]))
    return [
        Region(f"det-{i}", box, label, detection_score=score)
        for i, (score, box, label) in enumerate(candidates[: settings.max_detections])
    ]


class YoloDetector:
    def __init__(self, settings: DetectorSettings, device: str):
        try:
            import torch
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                'Install detector dependencies: pip install -e ".[detection]"'
            ) from exc
        self.settings = settings
        self.device = (
            ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        )
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; use --device cpu")
        self.model = YOLO(settings.model, task="detect")
        path = Path(self.model.ckpt_path)
        self.weights_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    def detect(self, image: Image.Image) -> list[Region]:
        result = self.model.predict(
            source=image,
            device=self.device,
            conf=self.settings.confidence,
            iou=self.settings.iou,
            imgsz=self.settings.image_size,
            max_det=self.settings.max_detections,
            verbose=False,
            save=False,
        )[0]
        boxes = result.boxes
        if boxes is None:
            return []
        return decode_detections(
            boxes.xyxy.cpu().numpy(),
            boxes.conf.cpu().numpy(),
            boxes.cls.cpu().numpy(),
            result.names,
            image.size,
            self.settings,
        )

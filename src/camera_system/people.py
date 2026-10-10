"""Visible person observations with explicit evidence; no identity or emotion inference."""

import base64
import hashlib
from collections import deque
from io import BytesIO
from math import acos, degrees, hypot, isfinite
from pathlib import Path

import numpy as np

from .tracking import iou


def preview_image(image):
    thumbnail = image.copy()
    thumbnail.thumbnail((960, 640))
    stream = BytesIO()
    thumbnail.convert("RGB").save(stream, format="JPEG", quality=75)
    return "data:image/jpeg;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


def upper_body_color(image, box):
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    if width < 24 or height < 48:
        return {"status": "unknown", "reason": "Person region too small for a useful color sample."}
    crop_box = (
        max(0, int(x1 + width * 0.25)),
        max(0, int(y1 + height * 0.22)),
        min(image.width, int(x1 + width * 0.75)),
        min(image.height, int(y1 + height * 0.52)),
    )
    if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
        return {"status": "unknown", "reason": "No visible upper-body sample."}
    hsv = np.asarray(image.crop(crop_box).resize((32, 32)).convert("HSV"), dtype=float) / 255
    hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    colors = np.full(hue.shape, "gray", dtype="U8")
    colors[value < 0.18] = "black"
    colors[(saturation < 0.18) & (value > 0.82)] = "white"
    colorful = (saturation >= 0.22) & (value >= 0.18)
    for label, low, high in [
        ("red", 0, 0.045),
        ("orange", 0.045, 0.11),
        ("yellow", 0.11, 0.19),
        ("green", 0.19, 0.46),
        ("cyan", 0.46, 0.55),
        ("blue", 0.55, 0.72),
        ("purple", 0.72, 0.87),
        ("pink", 0.87, 1.01),
    ]:
        colors[colorful & (hue >= low) & (hue < high)] = label
    colors[colorful & (hue > 0.96)] = "red"
    labels, counts = np.unique(colors, return_counts=True)
    winner = int(np.argmax(counts))
    fraction = float(counts[winner] / colors.size)
    return {
        "status": "observed" if fraction >= 0.55 else "uncertain",
        "value": str(labels[winner]) if fraction >= 0.55 else "mixed colors",
        "sample_fraction": round(fraction, 3),
        "sample_box": list(crop_box),
        "method": "upper-body pixel sample",
        "limitation": "May include skin, occlusion or background; not a clothing classifier.",
    }


def pose_description(points, scores):
    points, scores = np.asarray(points), np.asarray(scores)
    required = [5, 6, 11, 12, 13, 14, 15, 16]
    unknown = {
        "status": "unknown",
        "value": "Posture unclear",
        "reason": "Shoulders, hips, knees and ankles must be visible with keypoint scores ≥ 0.5.",
    }
    if (
        points.shape != (17, 2)
        or scores.shape != (17,)
        or not np.isfinite(points).all()
        or not np.isfinite(scores).all()
        or np.any(scores[required] < 0.5)
    ):
        return unknown

    def angle(a, b, c):
        v, w = points[a] - points[b], points[c] - points[b]
        norm = np.linalg.norm(v) * np.linalg.norm(w)
        return degrees(acos(float(np.clip(np.dot(v, w) / norm, -1, 1)))) if norm > 1 else None

    angles = [angle(11, 13, 15), angle(12, 14, 16)]
    shoulders = (points[5] + points[6]) / 2
    hips = (points[11] + points[12]) / 2
    if None in angles or np.linalg.norm(hips - shoulders) < 5:
        return unknown
    upright = hips[1] - shoulders[1] > abs(hips[0] - shoulders[0])
    if upright and min(angles) > 155:
        label = "Upright body with straight legs"
    elif upright and max(angles) < 140:
        label = "Upright body with bent knees"
    else:
        return {
            "status": "unknown",
            "value": "Posture unclear",
            "reason": "Visible joint geometry does not clearly match the supported posture patterns.",
        }
    return {
        "status": "estimated",
        "value": label,
        "knee_angles_degrees": [round(a, 1) for a in angles],
        "minimum_keypoint_score": round(float(min(scores[required])), 3),
        "method": "2D joint geometry",
        "limitation": "Not a trained activity classifier; bent knees do not prove sitting.",
    }


class PoseEstimator:
    def __init__(self, device="cpu", model="yolo11n-pose.pt"):
        import torch
        from ultralytics import YOLO

        self.device = (
            ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        )
        self.model = YOLO(model, task="pose")
        self.metadata = {
            "model": model,
            "device": self.device,
            "weights_sha256": hashlib.sha256(Path(self.model.ckpt_path).read_bytes()).hexdigest(),
        }

    def predict(self, image):
        result = self.model.predict(
            source=image, device=self.device, verbose=False, save=False, conf=0.35, max_det=30
        )[0]
        if result.keypoints is None or result.boxes is None or result.keypoints.conf is None:
            return []
        return [
            {"box": box.tolist(), "points": points.tolist(), "scores": scores.tolist()}
            for box, points, scores in zip(
                result.boxes.xyxy.cpu().numpy(),
                result.keypoints.xy.cpu().numpy(),
                result.keypoints.conf.cpu().numpy(),
                strict=True,
            )
        ]


class PersonAnalyzer:
    def __init__(self, pose_estimator):
        self.pose_estimator = pose_estimator
        self.previous = {}
        self.last_timestamp = None

    def analyze(self, image, regions, relations, timestamp=None):
        if timestamp is not None:
            if (
                not isfinite(timestamp)
                or timestamp < 0
                or (self.last_timestamp is not None and timestamp <= self.last_timestamp)
            ):
                raise ValueError("person-analysis timestamps must be finite and increasing")
            self.last_timestamp = timestamp
        people = [r for r in regions if r.label == "person"]
        poses = self.pose_estimator.predict(image) if people else []
        pairs = sorted(
            (
                (iou(person.box, pose["box"]), i, j)
                for i, person in enumerate(people)
                for j, pose in enumerate(poses)
            ),
            reverse=True,
        )
        matches, used = {}, set()
        for overlap, i, j in pairs:
            if overlap >= 0.3 and i not in matches and j not in used:
                matches[i] = j
                used.add(j)
        labels = {r.track_id or r.id: r.label or "object" for r in regions}
        observations, next_previous = [], {}
        for i, person in enumerate(people):
            key = person.track_id or person.id
            x1, y1, x2, y2 = person.box
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            motion = {"status": "unknown", "value": "More video evidence needed"}
            samples = (
                self.previous.get(key, deque(maxlen=128)) if person.track_id else deque(maxlen=128)
            )
            if samples and (
                timestamp is None or samples[-1][2] != image.size or timestamp - samples[-1][0] > 2
            ):
                samples.clear()
            if timestamp is not None and person.track_id:
                while samples and timestamp - samples[0][0] > 1:
                    samples.popleft()
                samples.append((timestamp, center, image.size))
                next_previous[key] = samples
            old = samples[0] if samples else None
            if (
                old
                and timestamp is not None
                and old[2] == image.size
                and 0.25 <= timestamp - old[0] <= 2
            ):
                dt = timestamp - old[0]
                dx, dy = center[0] - old[1][0], center[1] - old[1][1]
                speed = hypot(dx, dy) / hypot(*image.size) / dt
                direction = (
                    ("right" if dx > 0 else "left")
                    if abs(dx) >= abs(dy)
                    else ("down" if dy > 0 else "up")
                )
                motion = {
                    "status": "estimated",
                    "value": "Little image displacement"
                    if speed < 0.01
                    else f"Moving {direction} in the image",
                    "image_diagonals_per_second": round(speed, 4),
                    "span_seconds": round(dt, 3),
                    "support_count": len(samples),
                    "limitation": "Camera movement and box jitter affect this estimate; not walking/running recognition.",
                }
            pose = poses[matches[i]] if i in matches else None
            posture = (
                pose_description(pose["points"], pose["scores"])
                if pose
                else {
                    "status": "unknown",
                    "value": "Posture unclear",
                    "reason": "No reliable pose match.",
                }
            )
            interactions = [
                {
                    "predicate": r.predicate,
                    "object_id": r.object_id,
                    "object_label": labels.get(r.object_id, "object"),
                    "model_score": r.score,
                    "status": "model_candidate",
                }
                for r in relations
                if r.subject_id == key and 0.6 <= r.score <= 1
            ]
            observations.append(
                {
                    "person_id": key,
                    "box": list(person.box),
                    "appearance": upper_body_color(image, person.box),
                    "posture": posture,
                    "motion": motion,
                    "interactions": interactions,
                    "keypoints": pose["points"] if pose else [],
                    "keypoint_scores": pose["scores"] if pose else [],
                    "mood": {
                        "status": "not_inferred",
                        "reason": "Visible cues do not establish internal mood or intent.",
                    },
                }
            )
        self.previous = next_previous
        return observations

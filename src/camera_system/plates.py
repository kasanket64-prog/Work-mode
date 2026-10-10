"""Local plate candidates from video; never automatically query a registry."""

import argparse
import json
import re
from itertools import islice
from pathlib import Path

import numpy as np

from .people import preview_image
from .tracking import iou
from .video import FrameSource


def candidate(result, image):
    box = result.detection.bounding_box
    x1, y1 = max(0, int(box.x1)), max(0, int(box.y1))
    x2, y2 = min(image.width, int(box.x2)), min(image.height, int(box.y2))
    if x2 <= x1 or y2 <= y1:
        return None
    ocr = result.ocr
    text = re.sub(r"[^A-Z0-9]", "", ocr.text.upper()) if ocr else ""
    scores = np.asarray(ocr.confidence if ocr else 0, dtype=float)
    score = float(scores.min()) if scores.size else 0.0
    if not text or not np.isfinite(score):
        score = 0.0
    return {
        "text": text,
        "box": [x1, y1, x2, y2],
        "detection_score": float(result.detection.confidence),
        "ocr_score": score,
        "status": "review_required" if text and score >= 0.8 else "unreadable_or_uncertain",
        "crop_data_url": preview_image(image.crop((x1, y1, x2, y2))),
    }


def scan(model, image, vehicles=None):
    """Run once per vehicle at source resolution, mapping boxes back to the frame."""
    windows = (
        [(0, 0, image.width, image.height)]
        if vehicles is None
        else [
            (
                max(0, int(v.box[0])),
                max(0, int(v.box[1])),
                min(image.width, int(np.ceil(v.box[2]))),
                min(image.height, int(np.ceil(v.box[3]))),
            )
            for v in vehicles
            if v.label in {"car", "truck", "bus", "motorcycle"}
        ]
    )
    plates = []
    for x1, y1, x2, y2 in windows:
        if x2 <= x1 or y2 <= y1:
            continue
        crop = image.crop((x1, y1, x2, y2))
        for result in model.predict(np.asarray(crop)[:, :, ::-1].copy()):
            plate = candidate(result, crop)
            if plate:
                a, b, c, d = plate["box"]
                plate["box"] = [a + x1, b + y1, c + x1, d + y1]
                plates.append(plate)
    # ponytail: overlapping vehicle crops can repeat a plate; greedy spatial deduplication.
    kept = []
    for plate in sorted(plates, key=lambda p: p["detection_score"], reverse=True):
        if not any(iou(plate["box"], p["box"]) > 0.5 for p in kept):
            kept.append(plate)
    return kept


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=15)
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument(
        "--vehicle-crops",
        action="store_true",
        help="Experimental full-resolution vehicle crops; requires detection extra",
    )
    args = parser.parse_args(argv)
    if args.stride < 1 or args.max_frames < 1:
        parser.error("stride and max-frames must be positive")
    from fast_alpr import ALPR

    model = ALPR(
        detector_model="yolo-v9-t-384-license-plate-end2end",
        ocr_model="cct-xs-v2-global-model",
    )
    detector = None
    if args.vehicle_crops:
        from .detection import DetectorSettings, YoloDetector

        detector = YoloDetector(DetectorSettings(image_size=1280, max_detections=40), "cpu")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with (
        FrameSource(str(args.video), args.stride) as source,
        args.output.open("x", encoding="utf-8") as stream,
    ):

        def emit(row):
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()

        emit(
            {
                "type": "session",
                "source_kind": "video",
                "analysis": "plate_candidates",
                "source": str(args.video),
                "frame_stride": args.stride,
                "plate_scan_mode": "vehicle_crops" if detector else "full_frame",
            }
        )
        count = 0
        for frame in islice(source, args.max_frames):
            plates = scan(model, frame.image, detector.detect(frame.image) if detector else None)
            emit(
                {
                    "type": "frame",
                    "frame_index": frame.index,
                    "frame_id": f"frame-{frame.index}",
                    "timestamp_seconds": frame.timestamp_seconds,
                    "image_size": list(frame.image.size),
                    "plates": plates,
                    "regions": [
                        {"id": f"plate-{i}", "label": "plate", "box": p["box"]}
                        for i, p in enumerate(plates)
                    ],
                    "preview_data_url": preview_image(frame.image),
                }
            )
            count += 1
        emit({"type": "summary", "status": "completed", "frames_processed": count})
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

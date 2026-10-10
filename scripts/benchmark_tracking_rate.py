"""Compare dense versus sparse tracking on identical shared-frame detections."""

import argparse
import json
from collections import Counter
from dataclasses import asdict
from itertools import islice
from pathlib import Path
from statistics import mean
from time import perf_counter

from camera_system.detection import DetectorSettings, YoloDetector
from camera_system.people import preview_image
from camera_system.review import HTML
from camera_system.tracking import IoUTracker
from camera_system.video import FrameSource


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    args = parser.parse_args()
    if not args.video.is_file():
        parser.error(f"video file does not exist: {args.video}")
    if args.start_frame < 0:
        parser.error("--start-frame must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=False)
    settings = DetectorSettings(image_size=1280, max_detections=60)
    detector = YoloDetector(settings, "cpu")
    # Match missed-frame expiry in source time; both trackers also expire at 2 seconds.
    trackers = {"dense": IoUTracker(max_missed=30), "sparse": IoUTracker(max_missed=5)}
    rows = {
        name: [
            {
                "type": "session",
                "source_kind": "video",
                "source": str(args.video),
                "analysis_scope": "detection_and_tracking_only",
                "tracking_mode": name,
                "detector_settings": asdict(settings),
                "detector_weights_sha256": detector.weights_sha256,
                "tracker_stride": 1 if name == "dense" else 6,
                "max_missed": 30 if name == "dense" else 5,
                "review_stride": 6,
                "start_frame": args.start_frame,
            }
        ]
        for name in trackers
    }
    times = []
    with FrameSource(str(args.video)) as source:
        for data in rows.values():
            data[0]["fps"] = source.fps
        for frame in islice(source, args.start_frame, args.start_frame + 67):
            start = perf_counter()
            regions = detector.detect(frame.image)
            times.append(perf_counter() - start)
            dense = trackers["dense"].update(regions, frame.timestamp_seconds)
            if (frame.index - args.start_frame) % 6:
                continue
            sparse = trackers["sparse"].update(regions, frame.timestamp_seconds)
            preview = preview_image(frame.image)
            for name, tracked in (("dense", dense), ("sparse", sparse)):
                rows[name].append(
                    {
                        "type": "frame",
                        "frame_id": f"frame-{frame.index}",
                        "frame_index": frame.index,
                        "timestamp_seconds": frame.timestamp_seconds,
                        "image_size": list(frame.image.size),
                        "regions": [asdict(r) for r in tracked],
                        "relations": [],
                        "events": [],
                        "preview_data_url": preview,
                    }
                )
            print(f"Compared shared frame {frame.index}", flush=True)
    report = {
        "start_frame": args.start_frame,
        "decoded_frames": len(times),
        "mean_detection_seconds": mean(times) if times else None,
        "note": "Unlabeled ID-continuity diagnostic on shared frames, not identity accuracy. Detection runs six times as often in dense mode.",
        "modes": {},
    }
    for name, data in rows.items():
        tracks, previous = Counter(), set()
        reused = opportunities = 0
        for i, frame in enumerate(data[1:]):
            ids = {r["track_id"] for r in frame["regions"]}
            assert len(ids) == len(frame["regions"])
            if i:
                reused += len(ids & previous)
                opportunities += len(ids)
            tracks.update(ids)
            previous = ids
        report["modes"][name] = {
            "shared_frames": len(data) - 1,
            "track_ids_on_shared_frames": len(tracks),
            "single_observation_tracks": sum(v == 1 for v in tracks.values()),
            "shared_frame_id_reuse": reused / opportunities if opportunities else None,
        }
        data.append(
            {"type": "summary", "status": "completed", "end_reason": "bounded_tracking_diagnostic"}
        )
        (args.output / f"{name}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in data) + "\n", encoding="utf-8"
        )
        (args.output / f"{name}.html").write_text(
            HTML.replace("__DATA__", json.dumps(data).replace("<", "\\u003c")), encoding="utf-8"
        )
    for a, b in zip(rows["dense"][1:-1], rows["sparse"][1:-1], strict=True):
        assert a["frame_index"] == b["frame_index"]
        assert [{k: v for k, v in r.items() if k != "track_id"} for r in a["regions"]] == [
            {k: v for k, v in r.items() if k != "track_id"} for r in b["regions"]
        ]
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

"""Bounded local detection/tracking diagnostics; no ground-truth accuracy claims."""

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
    parser.add_argument("videos", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=12)
    args = parser.parse_args()
    if args.max_frames < 2:
        parser.error("--max-frames must be at least 2")
    args.output.mkdir(parents=True, exist_ok=False)
    profiles = {
        "baseline": DetectorSettings(),
        "detail": DetectorSettings(image_size=1280, max_detections=60),
    }
    detector = YoloDetector(profiles["baseline"], "cpu")
    summaries = []
    for path in args.videos:
        for name, settings in profiles.items():
            tracker = IoUTracker()
            detector.settings = settings
            previous = set()
            tracks = Counter()
            totals = Counter()
            timings, counts = [], []
            reused, opportunities, cap_hits = 0, 0, 0
            rows = [
                {
                    "type": "session",
                    "source_kind": "video",
                    "source": str(path),
                    "analysis_scope": "detection_and_tracking_only",
                    "profile": name,
                    "detector_settings": asdict(settings),
                    "detector_weights_sha256": detector.weights_sha256,
                    "relations_enabled": False,
                    "person_analysis_enabled": False,
                }
            ]
            with FrameSource(str(path)) as source:
                if not 0 < source.fps < 1000:
                    raise ValueError("video FPS must be valid for interval sampling")
                source.stride = max(1, round(source.fps / 4))
                rows[0].update(fps=source.fps, stride=source.stride)
                for i, frame in enumerate(islice(source, args.max_frames)):
                    if i == 0:
                        detector.detect(frame.image)
                    start = perf_counter()
                    regions = tracker.update(detector.detect(frame.image), frame.timestamp_seconds)
                    timings.append(perf_counter() - start)
                    current = {r.track_id for r in regions}
                    if i:
                        reused += len(current & previous)
                        opportunities += len(current)
                    previous = current
                    tracks.update(current)
                    totals.update(r.label for r in regions)
                    counts.append(len(regions))
                    cap_hits += len(regions) == settings.max_detections
                    rows.append(
                        {
                            "type": "frame",
                            "frame_id": f"frame-{frame.index}",
                            "frame_index": frame.index,
                            "timestamp_seconds": frame.timestamp_seconds,
                            "timestamp_source": frame.timestamp_source,
                            "image_size": list(frame.image.size),
                            "regions": [asdict(r) for r in regions],
                            "relations": [],
                            "events": [],
                            "preview_data_url": preview_image(frame.image),
                        }
                    )
                    print(f"{path.stem}/{name}: frame {i + 1}/{args.max_frames}", flush=True)
            summary = {
                "video": path.name,
                "profile": name,
                "frames": len(counts),
                "last_timestamp_seconds": rows[-1].get("timestamp_seconds"),
                "mean_detections": mean(counts) if counts else 0,
                "class_observations": dict(totals),
                "unique_track_ids": len(tracks),
                "one_observation_tracks": sum(n == 1 for n in tracks.values()),
                "previous_frame_id_reuse_fraction": reused / opportunities
                if opportunities
                else None,
                "mean_detection_tracking_seconds": mean(timings) if timings else None,
                "frames_at_detection_limit": cap_hits,
            }
            summaries.append(summary)
            rows.append(
                {
                    "type": "summary",
                    "status": "completed",
                    "stop_reason": "bounded_sample",
                    **summary,
                }
            )
            stem = f"{path.stem}-{name}"
            (args.output / f"{stem}.jsonl").write_text(
                "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
            )
            (args.output / f"{stem}.html").write_text(
                HTML.replace("__DATA__", json.dumps(rows).replace("<", "\\u003c")), encoding="utf-8"
            )
    (args.output / "comparison.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    lines = [
        "# Video detection and tracking diagnostics",
        "",
        "First 12 sampled frames by default, approximately 4 per second. No pose, relation or event inference.",
        "ID reuse is a diagnostic, not ID accuracy: incorrect assignments may also reuse an ID.",
        "Unique IDs are not counts of unique real objects. No annotations were supplied.",
        "",
        "| Video | Profile | Frames | Mean objects/frame | Track IDs | Mean inference seconds |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for s in summaries:
        stem = f"{Path(s['video']).stem}-{s['profile']}"
        lines.append(
            f"| {s['video']} | [{s['profile']}]({stem}.html) | {s['frames']} | {s['mean_detections']:.1f} | {s['unique_track_ids']} | {s['mean_detection_tracking_seconds']:.2f} |"
        )
    (args.output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

"""Replay saved detections through baseline and motion trackers without model inference."""

import argparse
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

from camera_system.contracts import Region
from camera_system.review import read_records
from camera_system.tracking import IoUTracker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sessions", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("use a new output path")
    report = []
    for path in args.sessions:
        frames = [r for r in read_records(path) if r["type"] == "frame"]
        for motion in (False, True):
            tracker = IoUTracker(motion=motion)
            tracks, previous = Counter(), set()
            reused = total = 0
            for i, frame in enumerate(frames):
                regions = [replace(Region(**r), track_id=None) for r in frame["regions"]]
                output = tracker.update(regions, frame["timestamp_seconds"])
                ids = {r.track_id for r in output}
                assert len(ids) == len(output)
                assert all(a.box == b.box for a, b in zip(regions, output, strict=True))
                if i:
                    reused += len(ids & previous)
                    total += len(ids)
                tracks.update(ids)
                previous = ids
            report.append(
                {
                    "session": path.name,
                    "motion": motion,
                    "frames": len(frames),
                    "track_ids": len(tracks),
                    "single_observation_tracks": sum(v == 1 for v in tracks.values()),
                    "previous_frame_id_reuse": reused / total if total else None,
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "note": "Unlabeled diagnostic; ID reuse is not identity accuracy. Predictions, relations and events are not recomputed.",
                "results": report,
            },
            stream,
            indent=2,
        )
    for row in report:
        print(json.dumps(row))


if __name__ == "__main__":
    main()

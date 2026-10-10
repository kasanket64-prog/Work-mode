"""Explicitly simulated tracked detections: exercise reasoning, not model accuracy."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .contracts import Region, Relation
from .reasoning import Reasoner, ReasoningSettings
from .review import HTML, evaluate
from .tracking import IoUTracker


def make_demo():
    tracker = IoUTracker()
    reasoner = Reasoner(ReasoningSettings())
    rows = [
        {
            "type": "session",
            "schema_version": 2,
            "source_kind": "simulated_detections",
            "relation_endpoint_kind": "track_id",
            "note": "Synthetic scenario; no detector/model inference or accuracy claim",
        }
    ]
    for t in range(36):
        regions = [Region("bag", (25, 20, 40, 40), "backpack")]
        if t < 3:
            regions.append(Region("person", (10, 10, 30, 60), "person"))
        tracked = tracker.update(regions, float(t))
        relation = (
            [Relation(tracked[1].track_id, "holding", tracked[0].track_id, 0.9)] if t < 3 else []
        )
        confirmed, events = reasoner.update(float(t), tracked, relation, (100, 100))
        rows.append(
            {
                "type": "frame",
                "frame_id": f"frame-{t}",
                "frame_index": t,
                "timestamp_seconds": float(t),
                "image_size": [100, 100],
                "regions": [asdict(r) for r in tracked],
                "relations": [asdict(r) for r in relation],
                "confirmed_relations": confirmed,
                "events": events,
            }
        )
    rows.append(
        {
            "type": "summary",
            "status": "completed",
            "frames_processed": 36,
            "note": "Simulated data; not a CCTV benchmark",
        }
    )
    truth = {
        "duration_seconds": 35,
        "events": [
            {
                "type": "possible_unattended_bag",
                "object_track_id": "track-1",
                "timestamp_seconds": 33.0,
            }
        ],
    }
    return rows, truth


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory")
    args = parser.parse_args(argv)
    rows, truth = make_demo()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "session.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    (args.output_dir / "truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
    (args.output_dir / "metrics.json").write_text(
        json.dumps(evaluate(rows, truth), indent=2), encoding="utf-8"
    )
    (args.output_dir / "review.html").write_text(
        HTML.replace("__DATA__", json.dumps(rows).replace("<", "\\u003c")), encoding="utf-8"
    )
    print(args.output_dir / "review.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

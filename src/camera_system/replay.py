"""Re-evaluate saved tracked video observations without running models again."""

import argparse
from dataclasses import asdict
from pathlib import Path

from .contracts import Region, Relation
from .reasoning import Reasoner, load_reasoning
from .relations import screen_relations
from .review import read_records
from .video_cli import emit


def replay(rows, settings):
    if rows[0].get("relation_endpoint_kind") != "track_id":
        raise ValueError("Replay needs a video session with track-ID relation endpoints")
    engine = Reasoner(settings)
    for original in rows:
        row = dict(original)
        if row.get("type") == "session":
            row.update(reasoning=asdict(settings), reasoning_replayed=True)
        elif row.get("type") == "frame":
            regions = [Region(**region) for region in row["regions"]]
            endpoints = [Region(r.track_id, r.box, r.label) for r in regions if r.track_id]
            relations, rejected = screen_relations(
                [Relation(**r) for r in row.get("relations", [])], endpoints
            )
            confirmed, events = engine.update(
                row["timestamp_seconds"], regions, relations, row["image_size"]
            )
            row.update(
                relations=[asdict(r) for r in relations],
                rejected_relations=[*row.get("rejected_relations", []), *rejected],
                confirmed_relations=confirmed,
                events=events,
                zones=settings.zones,
            )
        yield row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--reasoning-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    settings = load_reasoning(args.reasoning_config)
    rows = read_records(args.session)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        for row in replay(rows, settings):
            emit(stream, row)


if __name__ == "__main__":
    main()

"""Compare CPU threads using fixed office boxes and one image, excluding model startup."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from statistics import median
from time import perf_counter

import torch
from PIL import Image

from camera_system.config import load_settings
from camera_system.contracts import Region
from camera_system.relations import RelateAnythingPredictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    frame = next(
        json.loads(s)
        for s in args.session.read_text().splitlines()
        if json.loads(s)["type"] == "frame"
    )
    regions = [Region(**r) for r in frame["regions"]]
    with Image.open(args.image) as source:
        image = source.convert("RGB")
    predictor = RelateAnythingPredictor(load_settings(Path("configs/camera.toml")))
    results = []
    reference = None
    for threads in (8, 4, 2, 1):
        torch.set_num_threads(threads)
        predictor.predict(image, regions)
        times = []
        for _ in range(3):
            start = perf_counter()
            predictions = predictor.predict(image, regions)
            times.append(perf_counter() - start)
        keyed = {(p.subject_id, p.predicate, p.object_id): p.score for p in predictions}
        if reference is None:
            reference = keyed
        same = keyed.keys() == reference.keys()
        row = {
            "threads": threads,
            "seconds": times,
            "median_seconds": median(times),
            "same_candidate_set": same,
            "max_score_delta": max(abs(keyed[k] - reference[k]) for k in keyed) if same else None,
            "predictions": [asdict(p) for p in predictions],
        }
        results.append(row)
        print({k: v for k, v in row.items() if k != "predictions"}, flush=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2)


if __name__ == "__main__":
    main()

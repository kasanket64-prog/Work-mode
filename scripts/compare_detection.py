"""Compare detection settings on local images, without claiming measured accuracy."""

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from PIL import Image

from camera_system.detection import DetectorSettings, YoloDetector
from camera_system.people import preview_image
from camera_system.review import HTML


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    profiles = {
        "baseline": DetectorSettings(),
        "crowd": DetectorSettings(max_detections=60),
        "detail": DetectorSettings(image_size=1280, max_detections=60),
    }
    detector = YoloDetector(profiles["baseline"], "cpu")
    results = []
    for index, path in enumerate(args.images, 1):
        with Image.open(path) as source:
            image = source.convert("RGB")
        for name, settings in profiles.items():
            detector.settings = settings
            # Warm up each shape; report the following measured inference only.
            detector.detect(image)
            start = perf_counter()
            regions = detector.detect(image)
            seconds = perf_counter() - start
            result = {
                "image": path.name,
                "profile": name,
                "settings": asdict(settings),
                "weights_sha256": detector.weights_sha256,
                "seconds": seconds,
                "counts": dict(Counter(r.label for r in regions)),
                "at_detection_limit": len(regions) == settings.max_detections,
                "status": "predicted",
                "image_size": list(image.size),
                "frame_id": f"sample-{index}-{name}",
                "regions": [asdict(r) for r in regions],
                "relations": [],
                "preview_data_url": preview_image(image),
            }
            stem = f"sample-{index}-{name}"
            (args.output / f"{stem}.json").write_text(json.dumps(result), encoding="utf-8")
            rows = [
                {"type": "session", "source_kind": "image"},
                dict(result, type="frame"),
                {"type": "summary", "status": "completed"},
            ]
            (args.output / f"{stem}.html").write_text(
                HTML.replace("__DATA__", json.dumps(rows).replace("<", "\\u003c")),
                encoding="utf-8",
            )
            results.append(
                {k: v for k, v in result.items() if k not in {"preview_data_url", "regions"}}
            )
            print(f"{stem}: {result['counts']} ({seconds:.2f}s)", flush=True)
    (args.output / "comparison.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    lines = [
        "# Local detection comparison",
        "",
        "Unlabeled samples: counts are predictions, not accuracy. Timings are one warm CPU run per setting.",
        "Relation prediction and pose were not run. Original watermarks remain in previews.",
        "",
        "| Sample | Setting | Predicted classes/counts | Seconds | Limit reached |",
        "| --- | --- | --- | --- | --- |",
    ]
    for i, r in enumerate(results):
        stem = f"sample-{i // 3 + 1}-{r['profile']}"
        lines.append(
            f"| {i // 3 + 1} | [{r['profile']}]({stem}.html) | {r['counts']} | {r['seconds']:.2f} | {r['at_detection_limit']} |"
        )
    (args.output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

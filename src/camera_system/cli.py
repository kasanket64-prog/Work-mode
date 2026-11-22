"""Single image smoke runner; stdout is machine-readable JSON."""

import argparse
import hashlib
import json
import sys
import time
from contextlib import redirect_stdout
from dataclasses import asdict, replace
from pathlib import Path

from PIL import Image

from .config import UPSTREAM_REVISION, load_settings
from .contracts import parse_regions
from .detection import YoloDetector, load_detector_settings
from .people import PersonAnalyzer, PoseEstimator, preview_image
from .relations import RelateAnythingPredictor

ATTRIBUTION = "Based on RelateAnything by Maëlic Neau — https://github.com/Maelic/RelateAnything"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=ATTRIBUTION)
    parser.add_argument("--image", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--regions", type=Path)
    source.add_argument("--detect", action="store_true", help="Find object regions automatically")
    parser.add_argument("--detector-config", type=Path, default=Path("configs/detector.toml"))
    parser.add_argument("--config", type=Path, default=Path("configs/phase1.toml"))
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--analyze-people", action="store_true")
    parser.add_argument(
        "--include-preview", action="store_true", help="Embed a local image thumbnail in results"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate inputs without loading a model or making predictions",
    )
    args = parser.parse_args(argv)
    try:
        settings = load_settings(args.config)
        if args.device:
            settings = replace(settings, device=args.device)
        if args.detect and args.validate_only:
            raise ValueError("--validate-only requires manual --regions; --detect runs a model")
        if args.analyze_people and args.validate_only:
            raise ValueError("person analysis requires inference, not --validate-only")
        inputs = [args.image, args.config, args.detector_config if args.detect else args.regions]
        if args.output and args.output.resolve() in {p.resolve() for p in inputs}:
            raise ValueError("output must not overwrite an input file")
        if args.output and args.output.exists():
            raise ValueError("output already exists; choose a new path")
        with Image.open(args.image) as source:
            image = source.convert("RGB")
        detector_info = None
        if args.detect:
            detector_settings = load_detector_settings(args.detector_config)
            detector_settings = replace(
                detector_settings,
                max_detections=min(detector_settings.max_detections, settings.max_regions),
            )
            if args.output and args.output.resolve() == Path(detector_settings.model).resolve():
                raise ValueError("output must not overwrite detector weights")
            start = time.perf_counter()
            with redirect_stdout(sys.stderr):
                detector = YoloDetector(detector_settings, settings.device)
                regions = detector.detect(image)
            detector_info = {
                "settings": asdict(detector_settings),
                "device": detector.device,
                "weights_sha256": detector.weights_sha256,
                "load_and_detection_seconds": time.perf_counter() - start,
            }
        else:
            regions = parse_regions(
                json.loads(args.regions.read_text(encoding="utf-8-sig")), image.size
            )
        if len(regions) > settings.max_regions:
            raise ValueError(f"input exceeds max_regions={settings.max_regions}")
        result = {
            "schema_version": 1,
            "attribution": ATTRIBUTION,
            "status": "validated" if args.validate_only else "predicted",
            "image": str(args.image),
            "image_sha256": hashlib.sha256(args.image.read_bytes()).hexdigest(),
            "image_size": list(image.size),
            "frame_id": "image-0",
            "timestamp_seconds": None,
            "regions": [asdict(r) for r in regions],
            "settings": asdict(settings),
            "upstream_revision": UPSTREAM_REVISION,
        }
        if args.include_preview:
            result["preview_data_url"] = preview_image(image)
        if detector_info is not None:
            result["detector"] = detector_info
            result["region_source"] = "detector"
        else:
            result["region_source"] = "manual"
        if not args.validate_only and len(regions) < 2:
            result["relations"] = []
            result["relation_skip_reason"] = "fewer_than_two_regions"
        elif not args.validate_only:
            print(ATTRIBUTION, file=sys.stderr)
            start = time.perf_counter()
            with redirect_stdout(sys.stderr):
                predictor = RelateAnythingPredictor(settings)
            result["load_seconds"] = time.perf_counter() - start
            start = time.perf_counter()
            with redirect_stdout(sys.stderr):
                result["relations"] = [asdict(r) for r in predictor.predict(image, regions)]
            result["rejected_relations"] = getattr(predictor, "rejected_relations", [])
            result["prediction_seconds"] = time.perf_counter() - start
            result["resolved_device"] = predictor.device
        if args.analyze_people:
            from .contracts import Relation

            with redirect_stdout(sys.stderr):
                pose = PoseEstimator(settings.device)
                result["person_analysis"] = PersonAnalyzer(pose).analyze(
                    image, regions, [Relation(**r) for r in result.get("relations", [])]
                )
                result["pose_model"] = pose.metadata
        payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(payload)
        print(payload, end="")
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"camera-relations: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

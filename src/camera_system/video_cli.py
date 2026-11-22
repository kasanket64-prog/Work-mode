"""Video/camera detection and relationships; flushed JSONL records, reusable models."""

import argparse
import json
import sys
import time
from contextlib import ExitStack, redirect_stdout
from dataclasses import asdict, replace
from math import isfinite
from pathlib import Path

import numpy as np

from .behavior import BehaviorAnalyzer
from .cli import ATTRIBUTION
from .config import UPSTREAM_REVISION, load_settings
from .contracts import Relation
from .detection import YoloDetector, load_detector_settings
from .people import PersonAnalyzer, PoseEstimator, preview_image, save_preview_image
from .reasoning import Reasoner, load_reasoning
from .relations import RelateAnythingPredictor
from .tracking import IoUTracker
from .video import FrameSource


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def emit(stream, record):
    stream.write(json.dumps(record, allow_nan=False) + "\n")
    stream.flush()


def resolve_stride(fps, stride, relation_every, reasoning_settings, auto_stride=False):
    if not isfinite(fps) or fps <= 0:
        return stride
    interval = stride * relation_every / fps
    too_sparse = (
        interval > reasoning_settings.max_gap_seconds
        or interval * max(0, reasoning_settings.min_hits - 1) > reasoning_settings.window_seconds
    )
    if not too_sparse:
        return stride
    if auto_stride:
        max_interval = reasoning_settings.max_gap_seconds
        if reasoning_settings.min_hits > 1:
            max_interval = min(
                max_interval,
                reasoning_settings.window_seconds / (reasoning_settings.min_hits - 1),
            )
        effective = int(fps * max_interval // relation_every)
        while effective >= 1:
            candidate_interval = effective * relation_every / fps
            if (
                candidate_interval <= reasoning_settings.max_gap_seconds
                and candidate_interval * max(0, reasoning_settings.min_hits - 1)
                <= reasoning_settings.window_seconds
            ):
                return effective
            effective -= 1
    raise ValueError(
        "Relationship sampling is too sparse for the reasoning settings; "
        "reduce --stride or --relation-every, or review the evidence window/gap."
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=f"Video relationships. {ATTRIBUTION}")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path)
    source.add_argument("--camera", type=int, help="Local camera index, typically 0")
    source.add_argument("--stream", help="Authorized HTTP(S)/RTSP video URL; not a camera webpage")
    parser.add_argument(
        "--output", type=Path, required=True, help="New JSONL file (never overwritten)"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/phase1.toml"))
    parser.add_argument("--detector-config", type=Path, default=Path("configs/detector.toml"))
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"])
    parser.add_argument(
        "--stride", type=positive, default=1, help="Process every Nth decoded frame"
    )
    parser.add_argument(
        "--auto-stride",
        action="store_true",
        help="Lower stride automatically for low-FPS videos to stay within reasoning gaps",
    )
    parser.add_argument("--max-frames", type=positive, default=100, help="Maximum processed frames")
    parser.add_argument(
        "--relation-every",
        type=positive,
        default=1,
        help="Run fresh relationship inference every N processed frames (default: every frame)",
    )
    parser.add_argument("--track-iou", type=float, default=0.3)
    parser.add_argument("--track-max-missed", type=int, default=5)
    parser.add_argument("--track-max-age", type=float, default=2.0)
    parser.add_argument(
        "--track-motion", action="store_true", help="Experimental constant-velocity box matching"
    )
    parser.add_argument("--reasoning-config", type=Path)
    parser.add_argument("--analyze-people", action="store_true")
    parser.add_argument("--pose-model", default="yolo11n-pose.pt")
    parser.add_argument(
        "--half", action="store_true", help="Use half precision (FP16) on CUDA"
    )
    parser.add_argument(
        "--static-diff-threshold",
        type=float,
        default=0.0,
        help="Optional grayscale pixel-diff threshold [0..255] to skip relation checks on static frames",
    )
    parser.add_argument(
        "--include-preview", action="store_true", help="Embed a thumbnail for each processed frame"
    )
    parser.add_argument(
        "--preview-dir",
        type=Path,
        help="Save JPEG thumbnails under this directory instead of embedding base64 data URLs",
    )
    args = parser.parse_args(argv)
    try:
        if not isfinite(args.static_diff_threshold) or not 0 <= args.static_diff_threshold <= 255:
            raise ValueError("--static-diff-threshold must be in [0, 255]")
        if args.output.exists():
            raise ValueError(f"output already exists; choose a new path: {args.output}")
        settings = load_settings(args.config)
        if args.device:
            settings = replace(settings, device=args.device)
        detector_settings = load_detector_settings(args.detector_config)
        detector_settings = replace(
            detector_settings,
            max_detections=min(detector_settings.max_detections, settings.max_regions),
        )
        tracker = IoUTracker(
            args.track_iou, args.track_max_missed, args.track_max_age, motion=args.track_motion
        )
        reasoning_settings = load_reasoning(args.reasoning_config)
        reasoner = Reasoner(reasoning_settings)
        behavior = BehaviorAnalyzer()
        frames = FrameSource(
            args.stream or (str(args.video) if args.video is not None else args.camera),
            args.stride,
            network=bool(args.stream),
        )
        effective_stride = args.stride
        with ExitStack() as stack:
            if args.video is not None:
                stack.enter_context(frames)
                effective_stride = resolve_stride(
                    frames.fps,
                    args.stride,
                    args.relation_every,
                    reasoning_settings,
                    auto_stride=args.auto_stride,
                )
                frames.stride = effective_stride
            args.output.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation prevents destruction of input files, weights or previous results.
            stream = stack.enter_context(args.output.open("x", encoding="utf-8"))
            processed = 0
            status = "completed"
            reason = None
            error = None
            start = time.perf_counter()
            exit_code = 0
            emit(
                stream,
                {
                    "type": "session",
                    "schema_version": 2,
                    "relation_endpoint_kind": "track_id",
                    "reasoning": asdict(reasoning_settings),
                    "person_analysis_enabled": args.analyze_people,
                    "previews_embedded": bool(args.include_preview and not args.preview_dir),
                    "previews_directory": str(args.preview_dir) if args.preview_dir else None,
                    "attribution": ATTRIBUTION,
                    "source": "network_stream"
                    if args.stream
                    else str(args.video)
                    if args.video is not None
                    else args.camera,
                    "source_kind": "stream"
                    if args.stream
                    else "video"
                    if args.video is not None
                    else "camera",
                    "settings": asdict(settings),
                    "detector_settings": asdict(detector_settings),
                    "upstream_revision": UPSTREAM_REVISION,
                    "stride": effective_stride,
                    "requested_stride": args.stride,
                    "auto_stride": args.auto_stride,
                    "max_frames": args.max_frames,
                    "relation_every": args.relation_every,
                    "tracker": {
                        "iou_threshold": args.track_iou,
                        "max_missed": args.track_max_missed,
                        "max_age_seconds": args.track_max_age,
                        "motion_prediction": args.track_motion,
                    },
                    "buffer_policy": "synchronous_no_application_queue",
                },
            )
            try:
                # Load once before opening a camera so startup does not build a capture backlog.
                with redirect_stdout(sys.stderr):
                    detector = (
                        YoloDetector(detector_settings, settings.device, half=args.half)
                        if args.half
                        else YoloDetector(detector_settings, settings.device)
                    )
                    predictor = RelateAnythingPredictor(settings)
                    if args.analyze_people:
                        pose_kwargs = {}
                        if args.pose_model != "yolo11n-pose.pt":
                            pose_kwargs["model"] = args.pose_model
                        if args.half:
                            pose_kwargs["half"] = True
                        pose = PoseEstimator(settings.device, **pose_kwargs)
                        people = PersonAnalyzer(
                            pose, relation_threshold=reasoning_settings.relation_threshold
                        )
                    else:
                        pose = None
                        people = None
                emit(
                    stream,
                    {
                        "type": "models",
                        "load_seconds": time.perf_counter() - start,
                        "detector_weights_sha256": detector.weights_sha256,
                        "detector_device": detector.device,
                        "relation_device": predictor.device,
                        "pose_model": pose.metadata if pose else None,
                    },
                )
                if args.video is None:
                    stack.enter_context(frames)
                previous_thumb = None
                for frame in frames:
                    inference_start = time.perf_counter()
                    stage_seconds = {}
                    with redirect_stdout(sys.stderr):
                        regions = detector.detect(frame.image)
                        stage_seconds["detection"] = time.perf_counter() - inference_start
                        stage_start = time.perf_counter()
                        regions = tracker.update(regions, frame.timestamp_seconds)
                        behaviors = behavior.update(
                            regions, frame.timestamp_seconds, frame.image.size
                        )
                        stage_seconds["tracking"] = time.perf_counter() - stage_start
                        stage_start = time.perf_counter()
                        static_skip = False
                        if args.static_diff_threshold > 0:
                            thumb = np.asarray(
                                frame.image.convert("L").resize((64, 64)), dtype=float
                            )
                            if (
                                previous_thumb is not None
                                and float(np.mean(np.abs(thumb - previous_thumb)))
                                < args.static_diff_threshold
                            ):
                                static_skip = True
                            previous_thumb = thumb
                        relation_skip_reason = (
                            "fewer_than_two_regions"
                            if len(regions) < 2
                            else "scheduled_skip"
                            if processed % args.relation_every
                            else "static_frame_skip"
                            if static_skip
                            else None
                        )
                        relations = (
                            predictor.predict(frame.image, regions)
                            if relation_skip_reason is None
                            else []
                        )
                        track_by_region = {r.id: r.track_id for r in regions}
                        rejected_relations = [
                            {
                                **r,
                                "subject_id": track_by_region.get(
                                    r["subject_id"], r["subject_id"]
                                ),
                                "object_id": track_by_region.get(
                                    r["object_id"], r["object_id"]
                                ),
                            }
                            for r in (
                                getattr(predictor, "rejected_relations", [])
                                if relation_skip_reason is None
                                else []
                            )
                        ]
                        relations = [
                            Relation(
                                track_by_region[r.subject_id],
                                r.predicate,
                                track_by_region[r.object_id],
                                r.score,
                            )
                            for r in relations
                        ]
                        stage_seconds["relations"] = time.perf_counter() - stage_start
                        stage_start = time.perf_counter()
                        confirmed, events = reasoner.update(
                            frame.timestamp_seconds, regions, relations, frame.image.size
                        )
                        stage_seconds["reasoning"] = time.perf_counter() - stage_start
                        stage_start = time.perf_counter()
                        person_observations = (
                            people.analyze(
                                frame.image, regions, relations, frame.timestamp_seconds
                            )
                            if people
                            else []
                        )
                        stage_seconds["person_analysis"] = time.perf_counter() - stage_start
                    stage_start = time.perf_counter()
                    preview = None
                    preview_path = None
                    if args.include_preview:
                        if args.preview_dir:
                            filename = f"frame-{frame.index}.jpg"
                            save_preview_image(frame.image, args.preview_dir / filename)
                            preview_path = filename
                        else:
                            preview = preview_image(frame.image)
                    stage_seconds["preview"] = time.perf_counter() - stage_start
                    record = {
                        "type": "frame",
                        "frame_id": f"frame-{frame.index}",
                        "frame_index": frame.index,
                        "timestamp_seconds": frame.timestamp_seconds,
                        "timestamp_source": frame.timestamp_source,
                        "image_size": list(frame.image.size),
                        "regions": [asdict(r) for r in regions],
                        "relations": [asdict(r) for r in relations],
                        "rejected_relations": rejected_relations,
                        "confirmed_relations": confirmed,
                        "events": events,
                        "zones": reasoning_settings.zones,
                        "person_analysis": person_observations,
                        "behavior_observations": behaviors,
                        "preview_data_url": preview,
                        "stage_seconds": stage_seconds,
                        "relation_skip_reason": relation_skip_reason,
                        "inference_seconds": time.perf_counter() - inference_start,
                    }
                    if preview_path is not None:
                        record["preview_path"] = preview_path
                    emit(stream, record)
                    processed += 1
                    if processed >= args.max_frames:
                        reason = "max_frames"
                        break
                reason = reason or frames.end_reason
            except KeyboardInterrupt:
                status, reason, exit_code = "interrupted", "keyboard_interrupt", 130
            except Exception as exc:
                status, reason, exit_code = "failed", frames.end_reason or "processing_error", 2
                error = str(exc)
                print(f"camera-video: {error}", file=sys.stderr)
            finally:
                summary = {
                    "type": "summary",
                    "status": status,
                    "end_reason": reason,
                    "frames_read": frames.frames_read,
                    "frames_processed": processed,
                    "frames_skipped_by_stride": frames.frames_skipped,
                    "device_dropped_frames": getattr(frames, "dropped_frames", None),
                    "elapsed_seconds": time.perf_counter() - start,
                    "error": error,
                }
                emit(stream, summary)
            print(json.dumps(summary, allow_nan=False))
            return exit_code
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"camera-video: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

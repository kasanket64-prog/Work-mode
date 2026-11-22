"""Persistent warm model worker for reusing loaded inference weights across workspace jobs."""

import json
import os
import queue
import subprocess
import sys
import time
from contextlib import ExitStack, redirect_stdout
from dataclasses import asdict, replace
from pathlib import Path
from threading import Lock, Thread

from PIL import Image

from .behavior import BehaviorAnalyzer
from .cli import ATTRIBUTION
from .config import UPSTREAM_REVISION, load_settings
from .contracts import Relation
from .detection import DetectorSettings, YoloDetector, load_detector_settings
from .people import PersonAnalyzer, PoseEstimator, preview_image, save_preview_image
from .plates import scan as scan_plates
from .reasoning import Reasoner, load_reasoning
from .relations import RelateAnythingPredictor
from .tracking import IoUTracker
from .video import FrameSource
from .video_cli import emit, resolve_stride

IMAGES = {".jpg", ".jpeg", ".png"}


class ModelCache:
    """In-memory cache of loaded models keyed by settings and device."""

    def __init__(self):
        self._detectors: dict[tuple, object] = {}
        self._predictors: dict[tuple, object] = {}
        self._poses: dict[tuple, object] = {}
        self._alpr = None
        self.load_counts = {"detector": 0, "predictor": 0, "pose": 0, "alpr": 0}

    def get_detector(self, settings: DetectorSettings, device: str, half: bool = False, factory=None):
        key = (
            settings.model,
            settings.confidence,
            settings.iou,
            settings.image_size,
            settings.max_detections,
            device,
            bool(half),
        )
        if key not in self._detectors:
            maker = factory or YoloDetector
            self._detectors[key] = maker(settings, device, half=half) if half else maker(settings, device)
            self.load_counts["detector"] += 1
        return self._detectors[key]

    def get_predictor(self, settings, factory=None):
        key = (
            settings.model_id,
            settings.model_revision,
            settings.device,
            settings.topk,
            settings.max_regions,
            settings.min_score,
            tuple(settings.vocabulary),
        )
        if key not in self._predictors:
            maker = factory or RelateAnythingPredictor
            self._predictors[key] = maker(settings)
            self.load_counts["predictor"] += 1
        return self._predictors[key]

    def get_pose(self, device: str, model: str = "yolo11n-pose.pt", half: bool = False, factory=None):
        key = (device, model, bool(half))
        if key not in self._poses:
            maker = factory or PoseEstimator
            kwargs = {}
            if model != "yolo11n-pose.pt":
                kwargs["model"] = model
            if half:
                kwargs["half"] = True
            self._poses[key] = maker(device, **kwargs)
            self.load_counts["pose"] += 1
        return self._poses[key]

    def get_alpr(self, factory=None):
        if self._alpr is None:
            if factory is not None:
                self._alpr = factory()
            else:
                try:
                    from fast_alpr import ALPR
                except ImportError as exc:
                    raise RuntimeError(
                        'Install plate dependencies: pip install -e ".[plates]"'
                    ) from exc
                self._alpr = ALPR(
                    detector_model="yolo-v9-t-384-license-plate-end2end",
                    ocr_model="cct-xs-v2-global-model",
                )
            self.load_counts["alpr"] += 1
        return self._alpr


def execute_job_request(req: dict, cache: ModelCache, emit_event=None) -> dict:
    """Execute one workspace job using cached models in the worker process."""
    emit_event = emit_event or (lambda payload: None)
    mode = req.get("mode", "scene")
    source = Path(req["source"])
    output = Path(req["output"])
    limit = int(req.get("limit", 60))
    stride_raw = req.get("stride", 15)
    auto_stride = stride_raw == "auto" or bool(req.get("auto_stride", False))
    stride = 15 if stride_raw == "auto" else int(stride_raw)
    device = req.get("device")
    vehicle_crops = bool(req.get("vehicle_crops", False))
    reasoning_config = Path(req["reasoning_config"]) if req.get("reasoning_config") else None
    preview_dir = Path(req["preview_dir"]) if req.get("preview_dir") else None
    config_path = Path(req.get("config", "configs/camera.toml"))
    detector_config_path = Path(req.get("detector_config", "configs/detector-office.toml"))

    if output.exists():
        raise ValueError(f"output already exists: {output}")

    if mode == "plates":
        frames = FrameSource(str(source), stride)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            count = 0
            status, reason, error, exit_code = "completed", None, None, 0
            start = time.perf_counter()
            emit(
                stream,
                {
                    "type": "session",
                    "source_kind": "video",
                    "analysis": "plate_candidates",
                    "source": str(source),
                    "frame_stride": stride,
                    "max_frames": limit,
                    "device": device or "cpu",
                    "plate_scan_mode": "vehicle_crops" if vehicle_crops else "full_frame",
                },
            )
            try:
                model = cache.get_alpr()
                detector = (
                    cache.get_detector(
                        DetectorSettings(image_size=1280, max_detections=40), device or "cpu"
                    )
                    if vehicle_crops
                    else None
                )
                with frames:
                    for frame in frames:
                        plates = scan_plates(
                            model,
                            frame.image,
                            detector.detect(frame.image) if detector else None,
                        )
                        preview = None
                        preview_path = None
                        if preview_dir:
                            filename = f"frame-{frame.index}.jpg"
                            save_preview_image(frame.image, preview_dir / filename)
                            preview_path = filename
                        else:
                            preview = preview_image(frame.image)
                        row = {
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
                            "preview_data_url": preview,
                        }
                        if preview_path is not None:
                            row["preview_path"] = preview_path
                        emit(stream, row)
                        count += 1
                        emit_event({"event": "progress", "frames_processed": count})
                        if count >= limit:
                            reason = "max_frames"
                            break
                    reason = reason or frames.end_reason
            except Exception as exc:
                status, reason, exit_code = "failed", frames.end_reason or "processing_error", 2
                error = str(exc)
            finally:
                emit(
                    stream,
                    {
                        "type": "summary",
                        "status": status,
                        "end_reason": reason,
                        "frames_read": frames.frames_read,
                        "frames_processed": count,
                        "frames_skipped_by_stride": frames.frames_skipped,
                        "elapsed_seconds": time.perf_counter() - start,
                        "error": error,
                    },
                )
        return {"event": "done", "status": status, "exit_code": exit_code, "error": error}

    settings = load_settings(config_path)
    if device:
        settings = replace(settings, device=device)
    detector_settings = load_detector_settings(detector_config_path)
    detector_settings = replace(
        detector_settings,
        max_detections=min(detector_settings.max_detections, settings.max_regions),
    )

    if source.suffix.lower() in IMAGES:
        import hashlib

        with Image.open(source) as raw_image:
            image = raw_image.convert("RGB")
        det_start = time.perf_counter()
        detector = cache.get_detector(detector_settings, settings.device)
        regions = detector.detect(image)
        det_elapsed = time.perf_counter() - det_start
        result = {
            "schema_version": 1,
            "attribution": ATTRIBUTION,
            "status": "predicted",
            "image": str(source),
            "image_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "image_size": list(image.size),
            "frame_id": "image-0",
            "timestamp_seconds": None,
            "regions": [asdict(r) for r in regions],
            "settings": asdict(settings),
            "upstream_revision": UPSTREAM_REVISION,
            "region_source": "detector",
            "detector": {
                "settings": asdict(detector_settings),
                "device": detector.device,
                "weights_sha256": detector.weights_sha256,
                "load_and_detection_seconds": det_elapsed,
            },
        }
        if preview_dir:
            save_preview_image(image, preview_dir / "image-0.jpg")
            result["preview_path"] = "image-0.jpg"
        else:
            result["preview_data_url"] = preview_image(image)
        if len(regions) < 2:
            result["relations"] = []
            result["relation_skip_reason"] = "fewer_than_two_regions"
        else:
            load_start = time.perf_counter()
            predictor = cache.get_predictor(settings)
            result["load_seconds"] = time.perf_counter() - load_start
            pred_start = time.perf_counter()
            result["relations"] = [asdict(r) for r in predictor.predict(image, regions)]
            result["rejected_relations"] = getattr(predictor, "rejected_relations", [])
            result["prediction_seconds"] = time.perf_counter() - pred_start
            result["resolved_device"] = predictor.device
        pose = cache.get_pose(settings.device)
        result["person_analysis"] = PersonAnalyzer(pose).analyze(
            image, regions, [Relation(**r) for r in result.get("relations", [])]
        )
        result["pose_model"] = pose.metadata
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
        emit_event({"event": "progress", "frames_processed": 1})
        return {"event": "done", "status": "completed", "exit_code": 0, "error": None}

    reasoning_settings = load_reasoning(reasoning_config)
    reasoner = Reasoner(reasoning_settings)
    behavior = BehaviorAnalyzer()
    tracker = IoUTracker()
    frames = FrameSource(str(source), stride)
    with ExitStack() as stack:
        stack.enter_context(frames)
        effective_stride = resolve_stride(
            frames.fps, stride, 1, reasoning_settings, auto_stride=auto_stride
        )
        frames.stride = effective_stride
        output.parent.mkdir(parents=True, exist_ok=True)
        stream = stack.enter_context(output.open("x", encoding="utf-8"))
        processed = 0
        status, reason, error, exit_code = "completed", None, None, 0
        start = time.perf_counter()
        emit(
            stream,
            {
                "type": "session",
                "schema_version": 2,
                "relation_endpoint_kind": "track_id",
                "reasoning": asdict(reasoning_settings),
                "person_analysis_enabled": True,
                "previews_embedded": not bool(preview_dir),
                "previews_directory": str(preview_dir) if preview_dir else None,
                "attribution": ATTRIBUTION,
                "source": str(source),
                "source_kind": "video",
                "settings": asdict(settings),
                "detector_settings": asdict(detector_settings),
                "upstream_revision": UPSTREAM_REVISION,
                "stride": effective_stride,
                "requested_stride": stride,
                "auto_stride": auto_stride,
                "max_frames": limit,
                "relation_every": 1,
                "tracker": {
                    "iou_threshold": 0.3,
                    "max_missed": 5,
                    "max_age_seconds": 2.0,
                    "motion_prediction": False,
                },
                "buffer_policy": "synchronous_no_application_queue",
            },
        )
        try:
            detector = cache.get_detector(detector_settings, settings.device)
            predictor = cache.get_predictor(settings)
            pose = cache.get_pose(settings.device)
            people = PersonAnalyzer(pose, relation_threshold=reasoning_settings.relation_threshold)
            emit(
                stream,
                {
                    "type": "models",
                    "load_seconds": time.perf_counter() - start,
                    "detector_weights_sha256": detector.weights_sha256,
                    "detector_device": detector.device,
                    "relation_device": predictor.device,
                    "pose_model": pose.metadata,
                    "warm_cache_loads": dict(cache.load_counts),
                },
            )
            for frame in frames:
                inference_start = time.perf_counter()
                stage_seconds = {}
                regions = detector.detect(frame.image)
                stage_seconds["detection"] = time.perf_counter() - inference_start
                stage_start = time.perf_counter()
                regions = tracker.update(regions, frame.timestamp_seconds)
                behaviors = behavior.update(regions, frame.timestamp_seconds, frame.image.size)
                stage_seconds["tracking"] = time.perf_counter() - stage_start
                stage_start = time.perf_counter()
                relation_skip_reason = "fewer_than_two_regions" if len(regions) < 2 else None
                relations = (
                    predictor.predict(frame.image, regions)
                    if relation_skip_reason is None
                    else []
                )
                track_by_region = {r.id: r.track_id for r in regions}
                rejected_relations = [
                    {
                        **r,
                        "subject_id": track_by_region.get(r["subject_id"], r["subject_id"]),
                        "object_id": track_by_region.get(r["object_id"], r["object_id"]),
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
                person_observations = people.analyze(
                    frame.image, regions, relations, frame.timestamp_seconds
                )
                stage_seconds["person_analysis"] = time.perf_counter() - stage_start
                stage_start = time.perf_counter()
                preview = None
                preview_path = None
                if preview_dir:
                    filename = f"frame-{frame.index}.jpg"
                    save_preview_image(frame.image, preview_dir / filename)
                    preview_path = filename
                else:
                    preview = preview_image(frame.image)
                stage_seconds["preview"] = time.perf_counter() - stage_start
                row = {
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
                    row["preview_path"] = preview_path
                emit(stream, row)
                processed += 1
                emit_event({"event": "progress", "frames_processed": processed})
                if processed >= limit:
                    reason = "max_frames"
                    break
            reason = reason or frames.end_reason
        except Exception as exc:
            status, reason, exit_code = "failed", frames.end_reason or "processing_error", 2
            error = str(exc)
        finally:
            emit(
                stream,
                {
                    "type": "summary",
                    "status": status,
                    "end_reason": reason,
                    "frames_read": frames.frames_read,
                    "frames_processed": processed,
                    "frames_skipped_by_stride": frames.frames_skipped,
                    "device_dropped_frames": getattr(frames, "dropped_frames", None),
                    "elapsed_seconds": time.perf_counter() - start,
                    "error": error,
                },
            )
    return {"event": "done", "status": status, "exit_code": exit_code, "error": error}


def serve_stdio() -> int:
    """Run the JSON-lines warm worker loop on stdin/stdout."""
    cache = ModelCache()
    proto_out = sys.stdout

    def send(payload: dict) -> None:
        proto_out.write(json.dumps(payload, allow_nan=False) + "\n")
        proto_out.flush()

    send({"event": "ready"})
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            cmd = json.loads(line)
        except ValueError as exc:
            send({"event": "error", "error": f"invalid request JSON: {exc}"})
            continue
        action = cmd.get("cmd")
        if action == "ping":
            send({"event": "pong"})
        elif action == "shutdown":
            send({"event": "shutdown"})
            break
        elif action == "run":
            log_path = Path(cmd["log_path"]) if cmd.get("log_path") else None
            try:
                if log_path is not None:
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    with (
                        log_path.open("a", encoding="utf-8") as job_log,
                        redirect_stdout(job_log),
                    ):
                        old_err = sys.stderr
                        sys.stderr = job_log
                        try:
                            result = execute_job_request(cmd, cache, emit_event=send)
                        finally:
                            sys.stderr = old_err
                else:
                    with redirect_stdout(sys.stderr):
                        result = execute_job_request(cmd, cache, emit_event=send)
                send(result)
            except Exception as exc:
                send({"event": "done", "status": "failed", "exit_code": 2, "error": str(exc)})
        else:
            send({"event": "error", "error": f"unknown command: {action}"})
    return 0


class WarmWorkerManager:
    """Manage a persistent child worker process with cooperative cancel and crash recovery."""

    def __init__(self, workspace: Path, env: dict | None = None, command=None):
        self.workspace = Path(workspace)
        self.env = env
        self.command = command or [sys.executable, "-m", "camera_system.worker"]
        self._lock = Lock()
        self._proc: subprocess.Popen | None = None
        self._events: queue.Queue | None = None

    def _spawn(self) -> None:
        env = dict(self.env) if self.env is not None else os.environ.copy()
        env.setdefault("HF_HOME", str(self.workspace / ".hf_cache"))
        env.setdefault("YOLO_CONFIG_DIR", str(self.workspace / ".yolo"))
        proc = subprocess.Popen(
            self.command,
            cwd=self.workspace,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            bufsize=1,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        events: queue.Queue = queue.Queue()

        def reader():
            try:
                assert proc.stdout is not None
                for line in proc.stdout:
                    events.put(("line", line))
            except Exception as exc:
                events.put(("err", str(exc)))
            finally:
                events.put(("eof", None))

        Thread(target=reader, daemon=True).start()
        deadline = time.monotonic() + 30
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._kill_proc(proc)
                raise RuntimeError("Warm worker timed out while starting.")
            try:
                kind, value = events.get(timeout=min(1.0, remaining))
            except queue.Empty:
                continue
            if kind == "eof":
                raise RuntimeError("Warm worker exited before becoming ready.")
            if kind == "line":
                try:
                    msg = json.loads(value)
                except ValueError:
                    continue
                if msg.get("event") == "ready":
                    break
        self._proc = proc
        self._events = events

    def _kill_proc(self, proc: subprocess.Popen | None) -> None:
        if proc is None:
            return
        if proc.poll() is None:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            else:
                proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass

    def _reset(self) -> None:
        self._kill_proc(self._proc)
        self._proc = None
        self._events = None

    def run_job(
        self, request: dict, log_path: Path, is_cancelled, timeout: float = 3600, progress_cb=None
    ) -> dict:
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                self._reset()
                self._spawn()
            proc = self._proc
            events = self._events
            assert proc is not None and events is not None and proc.stdin is not None
            payload = {"cmd": "run", **request, "log_path": str(log_path)}
            try:
                proc.stdin.write(json.dumps(payload) + "\n")
                proc.stdin.flush()
            except OSError as exc:
                self._reset()
                raise RuntimeError(f"Warm worker communication failed: {exc}") from exc

            deadline = time.monotonic() + timeout
            while True:
                if is_cancelled():
                    self._reset()
                    return {"status": "cancelled", "exit_code": 130, "error": None}
                if time.monotonic() > deadline:
                    self._reset()
                    raise RuntimeError("Analysis exceeded the one-hour job limit.")
                try:
                    kind, value = events.get(timeout=0.25)
                except queue.Empty:
                    if proc.poll() is not None:
                        self._reset()
                        raise RuntimeError("Warm worker terminated unexpectedly.")
                    continue
                if kind == "eof":
                    self._reset()
                    raise RuntimeError("Warm worker closed unexpectedly during job execution.")
                if kind == "err":
                    self._reset()
                    raise RuntimeError(f"Warm worker read error: {value}")
                try:
                    msg = json.loads(value)
                except ValueError:
                    continue
                if msg.get("event") == "progress" and progress_cb:
                    progress_cb(msg)
                elif msg.get("event") == "done":
                    return msg
                elif msg.get("event") == "error":
                    raise RuntimeError(msg.get("error") or "Warm worker reported an error.")

    def close(self) -> None:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None and self._proc.stdin is not None:
                try:
                    self._proc.stdin.write('{"cmd": "shutdown"}\n')
                    self._proc.stdin.flush()
                    self._proc.wait(timeout=3)
                except (OSError, subprocess.TimeoutExpired):
                    pass
            self._reset()


if __name__ == "__main__":
    raise SystemExit(serve_stdio())

"""Upload, warm-worker orchestration, SQLite persistence, SSE progress, and review workspace."""

import argparse
import hmac
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from threading import Condition, Lock, Thread
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

from .reasoning import ReasoningSettings, load_reasoning
from .replay import replay
from .review import HTML, annotation_template, evaluate, read_records
from .store import WorkspaceStore
from .vehicles import lookup
from .worker import WarmWorkerManager

MAX_UPLOAD = 512 * 1024 * 1024
EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".jpg", ".jpeg", ".png"}
IMAGES = {".jpg", ".jpeg", ".png"}
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
CSP_POLICY = (
    "default-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:; "
    "style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; "
    "connect-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'"
)


def _format_toml_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_format_toml_value(item) for item in value) + "]"
    raise ValueError(f"unsupported TOML value: {value!r}")


def write_reasoning_toml(path: Path, settings: ReasoningSettings) -> None:
    lines = [
        f"window_seconds = {settings.window_seconds}",
        f"min_hits = {settings.min_hits}",
        f"min_duration_seconds = {settings.min_duration_seconds}",
        f"relation_threshold = {settings.relation_threshold}",
        f"max_gap_seconds = {settings.max_gap_seconds}",
        f"unattended_seconds = {settings.unattended_seconds}",
        f"stationary_fraction = {settings.stationary_fraction}",
        f"person_near_fraction = {settings.person_near_fraction}",
        f"zone_margin_fraction = {settings.zone_margin_fraction}",
        f"dwell_seconds = {settings.dwell_seconds}",
        f"pair_distance_change_fraction = {settings.pair_distance_change_fraction}",
    ]
    for zone in settings.zones:
        lines.extend(
            [
                "",
                "[[zones]]",
                f"id = {_format_toml_value(zone['id'])}",
                f"box = {_format_toml_value(zone['box'])}",
                f"labels = {_format_toml_value(zone['labels'])}",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_magic_bytes(path: Path, extension: str) -> None:
    ext = extension.lower()
    with path.open("rb") as stream:
        header = stream.read(16)
    if ext in {".jpg", ".jpeg"}:
        valid = header.startswith(b"\xff\xd8\xff")
    elif ext == ".png":
        valid = header.startswith(b"\x89PNG\r\n\x1a\n")
    elif ext in {".mp4", ".mov"}:
        valid = len(header) >= 8 and header[4:8] in {b"ftyp", b"moov", b"mdat", b"wide", b"free"}
    elif ext == ".avi":
        valid = len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"AVI "
    elif ext in {".mkv", ".webm"}:
        valid = header.startswith(b"\x1a\x45\xdf\xa3")
    else:
        valid = False
    if not valid:
        raise ValueError("Uploaded file content does not match its declared media format.")


def command(
    source,
    output,
    mode,
    limit,
    reasoning_config=None,
    *,
    stride="15",
    auto_stride=False,
    device=None,
    vehicle_crops=False,
    preview_dir=None,
):
    if mode == "plates":
        cmd = [
            sys.executable,
            "-m",
            "camera_system.plates",
            str(source),
            "--output",
            str(output),
            "--max-frames",
            str(limit),
        ]
        if str(stride) != "15" and str(stride) != "auto":
            cmd.extend(["--stride", str(stride)])
        if device:
            cmd.extend(["--device", str(device)])
        if vehicle_crops:
            cmd.append("--vehicle-crops")
        return cmd
    common = [
        "--output",
        str(output),
        "--include-preview",
        *(["--preview-dir", str(preview_dir)] if preview_dir else []),
        "--analyze-people",
        "--config",
        "configs/camera.toml",
        "--detector-config",
        "configs/detector-office.toml",
        *(["--device", str(device)] if device else []),
    ]
    if source.suffix.lower() in IMAGES:
        return [
            sys.executable,
            "-m",
            "camera_system.cli",
            "--image",
            str(source),
            "--detect",
            *common,
        ]
    use_auto = auto_stride or str(stride) == "auto"
    numeric_stride = "15" if str(stride) == "auto" else str(stride)
    return [
        sys.executable,
        "-m",
        "camera_system.video_cli",
        "--video",
        str(source),
        "--max-frames",
        str(limit),
        "--stride",
        numeric_stride,
        *(["--auto-stride"] if use_auto else []),
        "--relation-every",
        "1",
        *(["--reasoning-config", str(reasoning_config)] if reasoning_config else []),
        *common,
    ]


class JobEventBus:
    """Thread-safe condition variable for SSE subscribers."""

    def __init__(self):
        self.cond = Condition()
        self.seq = 0
        self.closed = False

    def notify(self) -> None:
        with self.cond:
            self.seq += 1
            self.cond.notify_all()

    def close(self) -> None:
        with self.cond:
            self.closed = True
            self.seq += 1
            self.cond.notify_all()

    def wait(self, last_seq: int, timeout: float = 15.0) -> tuple[int, bool]:
        with self.cond:
            if self.seq == last_seq and not self.closed:
                self.cond.wait(timeout=timeout)
            return self.seq, self.closed


def _dir_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                continue
    return total


class Jobs:
    def __init__(
        self,
        root,
        workspace,
        reasoning_config=None,
        *,
        warm_worker: bool = False,
        retention_days: float | None = None,
        max_disk_mb: float | None = None,
    ):
        self.root, self.workspace = root.resolve(), workspace.resolve()
        self.reasoning_config = reasoning_config
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = Lock()
        self.active = None
        self.worker = None
        self.items = {}
        self.bus = JobEventBus()
        self.store = WorkspaceStore(self.root / "fieldnote.db")
        self.warm_worker = WarmWorkerManager(self.workspace) if warm_worker else None
        self.retention_days = retention_days
        self.max_disk_mb = max_disk_mb
        for path in self.root.glob("*/job.json"):
            try:
                item = json.loads(path.read_text(encoding="utf-8-sig"))
                if item["status"] in {"processing", "uploading", "cancelling"}:
                    item["status"] = "interrupted"
                    self.save(item)
                else:
                    self.store.upsert_job(item)
                self.items[item["id"]] = item
            except (OSError, ValueError, KeyError):
                continue
        self.prune_storage()

    def close(self) -> None:
        self.bus.close()
        if self.warm_worker is not None:
            self.warm_worker.close()
        self.store.close()

    def save(self, item):
        folder = self.root / item["id"]
        if folder.is_dir():
            temporary = folder / "job.tmp"
            temporary.write_text(json.dumps(item), encoding="utf-8")
            temporary.replace(folder / "job.json")
        self.store.upsert_job(item)

    def snapshot(self):
        with self.lock:
            for item in self.items.values():
                self.update_progress(item)
            items = sorted(
                (dict(item) for item in self.items.values()),
                key=lambda item: item["created"],
                reverse=True,
            )
            eval_counts = {
                key: len(self.store.list_evaluations(key)) for key in self.items
            }
        for item in items:
            progress = item.pop("progress", {})
            item.update(
                progress.get(
                    "values", {"frames_processed": 0, "coverage": "Coverage not recorded"}
                )
            )
            item["has_log"] = (self.root / item["id"] / "processing.log").is_file()
            item["evaluation_count"] = eval_counts.get(item["id"], 0)
        return items

    def update_progress(self, item):
        """Called under the job lock; results are append-only while workers run."""
        folder = self.root / item["id"]
        output = folder / "result.jsonl"
        if not output.is_file():
            return
        stat = output.stat()
        image = any((folder / ("input" + ext)).is_file() for ext in IMAGES)
        signature = [stat.st_ino, stat.st_mtime_ns, stat.st_size, image]
        previous = item.get("progress", {})
        if previous.get("signature") == signature:
            return
        old = previous.get("signature")
        append = (
            item["status"] in {"processing", "cancelling"}
            and old
            and not image
            and not old[3]
            and old[0] == stat.st_ino
            and old[2] < stat.st_size
        )
        values = (
            dict(previous["values"])
            if append
            else {"frames_processed": 0, "coverage": "Coverage not recorded"}
        )
        offset = previous["offset"] if append else 0
        if image:
            try:
                record = json.loads(output.read_text(encoding="utf-8-sig"))
                if isinstance(record, dict) and isinstance(record.get("regions"), list):
                    values.update(frames_processed=1, coverage="Single image")
            except json.JSONDecodeError:
                pass
        else:
            with output.open("rb") as stream:
                stream.seek(offset)
                while line := stream.readline():
                    # Revisit a partially written record on the next append.
                    if not line.endswith(b"\n") and item["status"] in {"processing", "cancelling"}:
                        break
                    offset = stream.tell()
                    try:
                        record = json.loads(line)
                    except (ValueError, UnicodeError):
                        if not line.endswith(b"\n"):
                            offset -= len(line)
                        continue
                    if not isinstance(record, dict):
                        continue
                    if record.get("type") == "frame":
                        values["frames_processed"] += 1
                        values["last_timestamp_seconds"] = record.get("timestamp_seconds")
                    elif record.get("type") == "summary":
                        values["coverage"] = {
                            "max_frames": "Stopped at sampled-frame limit; full video coverage is not established",
                            "video_eof_or_decode_failure": "Video ended or decoding stopped; only sampled frames were analyzed",
                        }.get(record.get("end_reason"), "Coverage not recorded")
        item["progress"] = {"signature": signature, "offset": offset, "values": values}
        self.save(item)

    def delete(self, key: str) -> int:
        with self.lock:
            item = self.items.get(key)
            if item is None:
                raise KeyError(key)
            if key == self.active or item["status"] in {"processing", "cancelling"}:
                raise ValueError("Cannot delete an active analysis.")
            self.items.pop(key)
            self.store.delete_job(key)
        folder = self.root / key
        freed = _dir_bytes(folder)
        shutil.rmtree(folder, ignore_errors=True)
        self.bus.notify()
        return freed

    def prune_storage(self) -> int:
        if self.retention_days is None and self.max_disk_mb is None:
            return 0
        removed = 0
        now = datetime.now(timezone.utc)
        with self.lock:
            candidates = sorted(
                (
                    item
                    for item in self.items.values()
                    if item["id"] != self.active
                    and item.get("status") not in {"processing", "cancelling"}
                ),
                key=lambda item: str(item.get("created", "")),
            )
        if self.retention_days is not None:
            cutoff = now - timedelta(days=self.retention_days)
            for item in list(candidates):
                try:
                    created = datetime.fromisoformat(str(item.get("created", "")))
                    if created.tzinfo is None:
                        created = created.replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
                if created < cutoff:
                    try:
                        self.delete(item["id"])
                        removed += 1
                        candidates.remove(item)
                    except (KeyError, ValueError):
                        pass
        if self.max_disk_mb is not None:
            limit_bytes = int(self.max_disk_mb * 1024 * 1024)
            while candidates and _dir_bytes(self.root) > limit_bytes:
                oldest = candidates.pop(0)
                try:
                    self.delete(oldest["id"])
                    removed += 1
                except (KeyError, ValueError):
                    break
        return removed

    def cancel(self, key):
        with self.lock:
            item = self.items.get(key)
            if key != self.active or item is None:
                return False
            item["status"] = "cancelling"
            self.save(item)
        self.bus.notify()
        return True

    def retry(self, key):
        with self.lock:
            previous = self.items.get(key)
            if (
                self.active
                or not previous
                or previous["status"] not in {"failed", "cancelled", "interrupted"}
            ):
                raise ValueError("Retry requires an unsuccessful job and no active analysis.")
            sources = [self.root / key / ("input" + ext) for ext in EXTENSIONS]
            sources = [source for source in sources if source.is_file()]
            if len(sources) != 1:
                raise ValueError("Saved upload is missing or ambiguous. Upload the file again.")
            source = sources[0]
            if not 0 < source.stat().st_size <= MAX_UPLOAD:
                raise ValueError("Saved upload must be between 1 byte and 512 MB.")
            item = {
                "id": uuid4().hex,
                "status": "processing",
                "mode": previous["mode"],
                "limit": previous["limit"],
                "stride": previous.get("stride", "15"),
                "reasoning_profile": previous.get("reasoning_profile", "default"),
                "vehicle_crops": bool(previous.get("vehicle_crops", False)),
                "device": previous.get("device", "cpu"),
                "created": datetime.now(timezone.utc).isoformat(),
                "bytes": source.stat().st_size,
                "retry_of": key,
            }
            self.active = item["id"]
        folder = self.root / item["id"]
        try:
            folder.mkdir()
            copied = folder / source.name
            shutil.copyfile(source, copied)
            with self.lock:
                self.save(item)
                self.items[item["id"]] = item
            self.bus.notify()
            self.worker = Thread(target=self.run, args=(item, copied), daemon=True)
            self.worker.start()
            return dict(item)
        except Exception:
            with self.lock:
                if self.active == item["id"]:
                    self.active = None
                if item["id"] in self.items:
                    item.update(
                        status="failed",
                        error="Could not start retry. Check available disk space and retry.",
                    )
                    self.save(item)
                else:
                    shutil.rmtree(folder, ignore_errors=True)
            self.bus.notify()
            raise

    def _resolve_reasoning_config(self, item: dict, folder: Path) -> Path | None:
        profile = item.get("reasoning_profile", "default")
        if profile == "behavior":
            candidate = self.workspace / "configs" / "behavior.toml"
            return candidate if candidate.is_file() else self.reasoning_config
        if profile == "zones":
            zones = self.store.list_zones()
            if zones:
                zone_cfg = folder / "reasoning-zones.toml"
                write_reasoning_toml(zone_cfg, ReasoningSettings(zones=zones))
                return zone_cfg
            return self.reasoning_config
        return self.reasoning_config

    def run(self, item, source):
        folder = self.root / item["id"]
        output = folder / "result.jsonl"
        log_path = folder / "processing.log"
        preview_dir = folder / "frames" if item["mode"] == "scene" else None
        reasoning_cfg = self._resolve_reasoning_config(item, folder)
        stride_val = item.get("stride", "15")
        device_val = item.get("device") if item.get("device") in {"cpu", "cuda", "auto"} else None
        vehicle_crops = bool(item.get("vehicle_crops", False))
        try:
            if self.warm_worker is not None:

                def is_cancelled():
                    with self.lock:
                        return item["status"] == "cancelling"

                req = {
                    "job_id": item["id"],
                    "mode": item["mode"],
                    "source": str(source),
                    "output": str(output),
                    "limit": item["limit"],
                    "stride": stride_val,
                    "auto_stride": stride_val == "auto",
                    "device": device_val,
                    "vehicle_crops": vehicle_crops,
                    "reasoning_config": str(reasoning_cfg) if reasoning_cfg else None,
                    "preview_dir": str(preview_dir) if preview_dir else None,
                }
                result = self.warm_worker.run_job(
                    req,
                    log_path,
                    is_cancelled=is_cancelled,
                    progress_cb=lambda _: self.bus.notify(),
                )
                if result.get("status") == "cancelled":
                    with self.lock:
                        item["status"] = "cancelled"
                    return
                if result.get("exit_code"):
                    raise RuntimeError(
                        result.get("error")
                        or "Analysis failed. See processing.log in this job's folder."
                    )
            else:
                environment = os.environ.copy()
                environment.setdefault("HF_HOME", str(self.workspace / ".hf_cache"))
                environment.setdefault("YOLO_CONFIG_DIR", str(self.workspace / ".yolo"))
                cmd = command(
                    source,
                    output,
                    item["mode"],
                    item["limit"],
                    reasoning_cfg,
                    stride=stride_val,
                    auto_stride=(stride_val == "auto"),
                    device=device_val,
                    vehicle_crops=vehicle_crops,
                    preview_dir=preview_dir,
                )
                with log_path.open("w", encoding="utf-8") as log:
                    process = subprocess.Popen(
                        cmd,
                        cwd=self.workspace,
                        env=environment,
                        stdout=log,
                        stderr=log,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    )
                    deadline = time.monotonic() + 3600
                    while process.poll() is None:
                        with self.lock:
                            cancelling = item["status"] == "cancelling"
                        if cancelling or time.monotonic() > deadline:
                            if os.name == "nt":
                                subprocess.run(
                                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                    stdout=log,
                                    stderr=log,
                                    check=False,
                                )
                            else:
                                process.kill()
                            process.wait()
                            with self.lock:
                                was_cancelling = item["status"] == "cancelling"
                            if was_cancelling:
                                with self.lock:
                                    item["status"] = "cancelled"
                                return
                            raise RuntimeError("Analysis exceeded the one-hour job limit.")
                        try:
                            process.wait(timeout=1)
                        except subprocess.TimeoutExpired:
                            pass
                        self.bus.notify()
                    code = process.returncode
                if code:
                    raise RuntimeError("Analysis failed. See processing.log in this job's folder.")
            rows = read_records(output)
            page = HTML.replace(
                "__DATA__", json.dumps(rows, allow_nan=False).replace("<", "\\u003c")
            )
            (folder / "review.html").write_text(page, encoding="utf-8")
            with self.lock:
                item["status"] = "completed"
        except Exception as exc:
            with self.lock:
                item.update(status="failed", error=str(exc))
        finally:
            with self.lock:
                if self.active == item["id"]:
                    self.active = None
                self.save(item)
            self.prune_storage()
            self.bus.notify()


def _system_status(jobs: Jobs, workspace: Path, auth_required: bool) -> dict:
    has_torch = importlib.util.find_spec("torch") is not None
    gpu_detected = shutil.which("nvidia-smi") is not None
    cuda_active = False
    if "torch" in sys.modules:
        try:
            cuda_active = bool(sys.modules["torch"].cuda.is_available())
        except Exception:
            cuda_active = False
    hf_cache = Path(os.environ.get("HF_HOME") or (workspace / ".hf_cache"))
    yolo_dir = Path(os.environ.get("YOLO_CONFIG_DIR") or (workspace / ".yolo"))
    detect_cached = (workspace / "yolo11n.pt").is_file() or (yolo_dir / "yolo11n.pt").is_file()
    pose_cached = (
        (workspace / "yolo11n-pose.pt").is_file() or (yolo_dir / "yolo11n-pose.pt").is_file()
    )
    with jobs.lock:
        active_job = jobs.active
        job_count = len(jobs.items)
    return {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "capabilities": {
            "inference": importlib.util.find_spec("relsgg") is not None,
            "detection": importlib.util.find_spec("ultralytics") is not None,
            "video": importlib.util.find_spec("cv2") is not None,
            "plates": importlib.util.find_spec("fast_alpr") is not None,
            "torch": has_torch,
            "cuda": cuda_active or (has_torch and gpu_detected),
        },
        "cache": {
            "hf_cache_exists": hf_cache.is_dir(),
            "yolo_detect_weights": detect_cached,
            "yolo_pose_weights": pose_cached,
        },
        "warm_worker": jobs.warm_worker is not None,
        "active_job": active_job,
        "job_count": job_count,
        "workspace_bytes": _dir_bytes(jobs.root),
        "retention_days": jobs.retention_days,
        "max_disk_mb": jobs.max_disk_mb,
        "auth_required": auth_required,
    }


def _merge_verdicts(rows: list[dict], verdicts: dict[str, dict]) -> list[dict]:
    if not verdicts:
        return rows
    merged = []
    for row in rows:
        if row.get("type") != "frame" or not row.get("events"):
            merged.append(row)
            continue
        updated_events = []
        for event in row["events"]:
            copy_event = dict(event)
            eid = copy_event.get("event_id")
            if eid and eid in verdicts:
                saved = verdicts[eid]
                if saved.get("verdict"):
                    copy_event["review_verdict"] = saved["verdict"]
                if saved.get("note"):
                    copy_event["review_note"] = saved["note"]
            updated_events.append(copy_event)
        merged.append({**row, "events": updated_events})
    return merged


def make_app(
    root,
    workspace,
    port=8780,
    reasoning_config=None,
    *,
    host: str = "127.0.0.1",
    auth_token: str | None = None,
    warm_worker: bool = False,
    strict_uploads: bool = False,
    retention_days: float | None = None,
    max_disk_mb: float | None = None,
):
    if host not in LOOPBACK_HOSTS and not auth_token:
        raise ValueError("Non-loopback binding requires an auth token (--auth-token-env).")
    jobs = Jobs(
        root,
        workspace,
        reasoning_config,
        warm_worker=warm_worker,
        retention_days=retention_days,
        max_disk_mb=max_disk_mb,
    )
    home = files("camera_system").joinpath("templates/workspace.html").read_bytes()
    loopback_only = host in LOOPBACK_HOSTS

    class Handler(BaseHTTPRequestHandler):
        def _extract_token(self) -> str:
            auth = self.headers.get("Authorization", "")
            if auth.lower().startswith("bearer "):
                return auth[7:].strip()
            token = self.headers.get("X-Fieldnote-Token")
            if token:
                return token.strip()
            cookie_hdr = self.headers.get("Cookie", "")
            if cookie_hdr:
                cookie = SimpleCookie()
                try:
                    cookie.load(cookie_hdr)
                except Exception:
                    pass
                if "fieldnote_token" in cookie:
                    return cookie["fieldnote_token"].value
            query = parse_qs(urlsplit(self.path).query)
            if "token" in query and query["token"]:
                return query["token"][0]
            return ""

        def local(self, write=False):
            hdr_host = self.headers.get("Host")
            if not hdr_host:
                return False
            if loopback_only:
                allowed = {
                    f"127.0.0.1:{self.server.server_port}",
                    f"localhost:{self.server.server_port}",
                }
                if hdr_host not in allowed:
                    return False
                if write and self.headers.get("Origin") != f"http://{hdr_host}":
                    return False
            else:
                if write:
                    origin = self.headers.get("Origin")
                    if origin and origin not in {f"http://{hdr_host}", f"https://{hdr_host}"}:
                        return False
            if auth_token:
                supplied = self._extract_token()
                if not supplied or not hmac.compare_digest(supplied, auth_token):
                    return False
            return True

        def respond(self, status, data, content_type="application/json"):
            body = data if isinstance(data, bytes) else json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if content_type.startswith("text/html"):
                self.send_header("Content-Security-Policy", CSP_POLICY)
            self.end_headers()
            self.wfile.write(body)

        def _read_json_body(self, max_bytes: int = 262144):
            length = int(self.headers.get("Content-Length", "0"))
            if self.headers.get("Transfer-Encoding") or not 0 < length <= max_bytes:
                raise ValueError(f"Expected JSON body between 1 byte and {max_bytes} bytes.")
            if self.headers.get_content_type() != "application/json":
                raise ValueError("Expected application/json content type.")
            return json.loads(self.rfile.read(length))

        def serve_media(self, key, head_only=False):
            sources = [jobs.root / key / ("input" + ext) for ext in EXTENSIONS]
            sources = [source for source in sources if source.is_file()]
            if len(sources) != 1:
                return self.respond(404, {"error": "Saved media is unavailable"})
            source = sources[0]
            size = source.stat().st_size
            requested = self.headers.get("Range")
            if size == 0:
                if requested:
                    self.send_response(416)
                    self.send_header("Content-Range", "bytes */0")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                start, end = 0, -1
            else:
                start, end = 0, size - 1
                if requested:
                    match = re.fullmatch(r"bytes=(\d{0,20})-(\d{0,20})", requested)
                    if not match or not any(match.groups()):
                        return self.respond(400, {"error": "Expected a single byte range"})
                    first, last = match.groups()
                    if first:
                        start = int(first)
                        end = min(int(last), end) if last else end
                    else:
                        start = max(0, size - int(last))
                if start > end or start >= size:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
            mime = {
                ".mp4": "video/mp4",
                ".mov": "video/quicktime",
                ".webm": "video/webm",
                ".avi": "video/x-msvideo",
                ".mkv": "video/x-matroska",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
            }
            content_length = 0 if size == 0 else end - start + 1
            self.send_response(206 if requested else 200)
            self.send_header("Content-Type", mime[source.suffix])
            self.send_header("Content-Length", str(content_length))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if requested:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if head_only or content_length == 0:
                return
            try:
                with source.open("rb") as stream:
                    stream.seek(start)
                    remaining = content_length
                    while remaining:
                        chunk = stream.read(min(1024 * 1024, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _serve_sse(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Connection", "close")
            self.end_headers()
            seq = -1
            try:
                while True:
                    payload = json.dumps(jobs.snapshot(), allow_nan=False)
                    msg = f"event: jobs\ndata: {payload}\n\n".encode("utf-8")
                    self.wfile.write(msg)
                    self.wfile.flush()
                    if seq == -1:
                        seq = jobs.bus.seq
                    new_seq, closed = jobs.bus.wait(seq, timeout=15.0)
                    if closed:
                        break
                    if new_seq == seq:
                        self.wfile.write(b": heartbeat\n\n")
                        self.wfile.flush()
                    else:
                        seq = new_seq
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            self.close_connection = True

        def do_HEAD(self):
            if not self.local():
                self.send_response(403)
                self.end_headers()
                return
            path = urlsplit(self.path).path
            media = re.fullmatch(r"/jobs/([a-f0-9]{32})/media", path)
            if media:
                return self.serve_media(media[1], head_only=True)
            self.send_response(404)
            self.end_headers()

        def do_GET(self):
            if not self.local():
                return self.respond(403, {"error": "Local access only"})
            path = urlsplit(self.path).path
            if path == "/":
                return self.respond(200, home, "text/html; charset=utf-8")
            assets = {
                "/assets/workspace.css": ("workspace.css", "text/css; charset=utf-8"),
                "/assets/workspace.js": ("workspace.js", "text/javascript; charset=utf-8"),
                "/assets/demo-horse.jpg": ("demo-horse.jpg", "image/jpeg"),
                "/assets/demo-result.json": ("demo-result.json", "application/json"),
            }
            if path in assets:
                name, content_type = assets[path]
                asset = files("camera_system").joinpath("templates", name).read_bytes()
                return self.respond(200, asset, content_type)
            if path == "/api/jobs":
                return self.respond(200, jobs.snapshot())
            if path == "/api/events":
                return self._serve_sse()
            if path == "/api/system/status":
                return self.respond(200, _system_status(jobs, workspace, bool(auth_token)))
            if path == "/api/zones":
                return self.respond(200, {"zones": jobs.store.list_zones()})
            verdicts_match = re.fullmatch(r"/api/jobs/([a-f0-9]{32})/verdicts", path)
            if verdicts_match:
                return self.respond(200, {"verdicts": jobs.store.get_verdicts(verdicts_match[1])})
            evals_match = re.fullmatch(r"/api/jobs/([a-f0-9]{32})/evaluations", path)
            if evals_match:
                return self.respond(
                    200, {"evaluations": jobs.store.list_evaluations(evals_match[1])}
                )
            ann_match = re.fullmatch(r"/api/jobs/([a-f0-9]{32})/annotations", path)
            if ann_match:
                key = ann_match[1]
                saved = jobs.store.get_annotation(key)
                if saved is not None:
                    return self.respond(200, saved)
                result_file = jobs.root / key / "result.jsonl"
                if not result_file.is_file():
                    return self.respond(404, {"error": "Saved results are unavailable"})
                try:
                    rows = read_records(result_file)
                    return self.respond(200, annotation_template(rows))
                except (OSError, ValueError) as exc:
                    return self.respond(409, {"error": str(exc)})
            frame_match = re.fullmatch(
                r"/jobs/([a-f0-9]{32})/frames/(frame-\d+\.jpg|image-0\.jpg)", path
            )
            if frame_match:
                key, filename = frame_match.groups()
                target = jobs.root / key / "frames" / filename
                if target.is_file():
                    return self.respond(200, target.read_bytes(), "image/jpeg")
                return self.respond(404, {"error": "Frame thumbnail is unavailable"})
            media = re.fullmatch(r"/jobs/([a-f0-9]{32})/media", path)
            if media:
                return self.serve_media(media[1], head_only=False)
            match = re.fullmatch(r"/jobs/([a-f0-9]{32})/(review|result|log)", path)
            if match:
                key, kind = match.groups()
                if kind == "review":
                    result_file = jobs.root / key / "result.jsonl"
                    if result_file.is_file():
                        try:
                            rows = read_records(result_file)
                        except (OSError, ValueError):
                            return self.respond(
                                409, {"error": "Saved results are unavailable or invalid"}
                            )
                        rows = _merge_verdicts(rows, jobs.store.get_verdicts(key))
                        page = HTML.replace(
                            "__DATA__", json.dumps(rows, allow_nan=False).replace("<", "\\u003c")
                        )
                        return self.respond(200, page.encode("utf-8"), "text/html; charset=utf-8")
                name = {
                    "review": "review.html",
                    "result": "result.jsonl",
                    "log": "processing.log",
                }[kind]
                target = jobs.root / key / name
                if target.is_file():
                    return self.respond(
                        200,
                        target.read_bytes(),
                        {
                            "review": "text/html; charset=utf-8",
                            "result": "application/x-ndjson",
                            "log": "text/plain; charset=utf-8",
                        }[kind],
                    )
            self.respond(404, {"error": "Not found"})

        def do_DELETE(self):
            if not self.local(write=True):
                self.close_connection = True
                return self.respond(403, {"error": "Same-origin local access required"})
            path = urlsplit(self.path).path
            job_match = re.fullmatch(r"/api/jobs/([a-f0-9]{32})", path)
            if job_match:
                key = job_match[1]
                try:
                    freed = jobs.delete(key)
                    return self.respond(200, {"deleted": key, "freed_bytes": freed})
                except KeyError:
                    return self.respond(404, {"error": "Analysis not found"})
                except ValueError as exc:
                    return self.respond(409, {"error": str(exc)})
            zone_match = re.fullmatch(r"/api/zones/([A-Za-z0-9_-]{1,40})", path)
            if zone_match:
                if jobs.store.delete_zone(zone_match[1]):
                    return self.respond(200, {"deleted": zone_match[1]})
                return self.respond(404, {"error": "Zone not found"})
            self.respond(404, {"error": "Not found"})

        def do_PUT(self):
            if not self.local(write=True):
                self.close_connection = True
                return self.respond(403, {"error": "Same-origin local access required"})
            try:
                self.connection.settimeout(30)
                path = urlsplit(self.path).path
                verdicts_match = re.fullmatch(r"/api/jobs/([a-f0-9]{32})/verdicts", path)
                if verdicts_match:
                    key = verdicts_match[1]
                    if not (jobs.root / key / "result.jsonl").is_file():
                        return self.respond(404, {"error": "Analysis not found"})
                    payload = self._read_json_body(262144)
                    if not isinstance(payload, dict) or not isinstance(payload.get("verdicts"), dict):
                        raise ValueError("Expected JSON object with verdicts dictionary.")
                    saved = jobs.store.set_verdicts(key, payload["verdicts"])
                    jobs.bus.notify()
                    return self.respond(200, {"verdicts": saved})
                ann_match = re.fullmatch(r"/api/jobs/([a-f0-9]{32})/annotations", path)
                if ann_match:
                    key = ann_match[1]
                    if not (jobs.root / key / "result.jsonl").is_file():
                        return self.respond(404, {"error": "Analysis not found"})
                    payload = self._read_json_body(262144)
                    saved = jobs.store.save_annotation(key, payload)
                    return self.respond(200, saved)
                return self.respond(404, {"error": "Not found"})
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True
            except (ValueError, TypeError) as exc:
                self.close_connection = True
                self.respond(400, {"error": str(exc)})
            except OSError:
                self.close_connection = True
                self.respond(400, {"error": "Local file or connection error while processing request."})

        def do_POST(self):
            if not self.local(write=True):
                self.close_connection = True
                return self.respond(403, {"error": "Same-origin local access required"})
            try:
                self.connection.settimeout(30)
                path = urlsplit(self.path).path
                length = int(self.headers.get("Content-Length", "0"))
                if self.headers.get("Transfer-Encoding") or not 0 < length <= MAX_UPLOAD:
                    raise ValueError("Upload must have a known size between 1 byte and 512 MB.")
                if path == "/api/zones":
                    if length > 32768 or self.headers.get_content_type() != "application/json":
                        raise ValueError("Zone definition must be small JSON")
                    payload = json.loads(self.rfile.read(length))
                    zone = jobs.store.upsert_zone(payload)
                    return self.respond(200, {"zone": zone, "zones": jobs.store.list_zones()})
                replay_match = re.fullmatch(r"/api/jobs/([a-f0-9]{32})/replay", path)
                if replay_match:
                    if length > 32768 or self.headers.get_content_type() != "application/json":
                        raise ValueError("Replay settings must be small JSON")
                    key = replay_match[1]
                    payload = json.loads(self.rfile.read(length))
                    if not isinstance(payload, dict):
                        raise ValueError("Expected JSON settings object")
                    with jobs.lock:
                        item = jobs.items.get(key)
                        ready = item is not None and item["status"] == "completed"
                        mode_ok = item is not None and item.get("mode") == "scene"
                    result_path = jobs.root / key / "result.jsonl"
                    if item is None and result_path.is_file():
                        ready, mode_ok = True, True
                    if not ready or not mode_ok:
                        return self.respond(409, {"error": "Choose a completed scene analysis"})
                    params = dict(payload)
                    use_saved_zones = bool(params.pop("use_saved_zones", True))
                    if use_saved_zones and "zones" not in params:
                        params["zones"] = jobs.store.list_zones()
                    settings = ReasoningSettings(**params)
                    orig_backup = jobs.root / key / "original-result.jsonl"
                    source_jsonl = orig_backup if orig_backup.is_file() else result_path
                    rows = read_records(source_jsonl)
                    if rows[0].get("relation_endpoint_kind") != "track_id":
                        return self.respond(
                            409, {"error": "Replay requires a tracked video session"}
                        )
                    if not orig_backup.exists():
                        shutil.copyfile(result_path, orig_backup)
                    replayed_rows = list(replay(rows, settings))
                    tmp_jsonl = jobs.root / key / "result.tmp"
                    with tmp_jsonl.open("w", encoding="utf-8") as stream:
                        for row in replayed_rows:
                            stream.write(json.dumps(row, allow_nan=False) + "\n")
                    tmp_jsonl.replace(result_path)
                    linked_rows = read_records(result_path)
                    merged_rows = _merge_verdicts(linked_rows, jobs.store.get_verdicts(key))
                    page = HTML.replace(
                        "__DATA__",
                        json.dumps(merged_rows, allow_nan=False).replace("<", "\\u003c"),
                    )
                    (jobs.root / key / "review.html").write_text(page, encoding="utf-8")
                    with jobs.lock:
                        if key in jobs.items:
                            jobs.items[key].pop("progress", None)
                            jobs.save(jobs.items[key])
                    jobs.bus.notify()
                    events_count = sum(
                        len(r.get("events", [])) for r in linked_rows if r.get("type") == "frame"
                    )
                    confirmed_count = sum(
                        len(r.get("confirmed_relations", []))
                        for r in linked_rows
                        if r.get("type") == "frame"
                    )
                    from dataclasses import asdict

                    return self.respond(
                        200,
                        {
                            "status": "replayed",
                            "events_count": events_count,
                            "confirmed_relations_count": confirmed_count,
                            "zones_count": len(settings.zones),
                            "reasoning": asdict(settings),
                        },
                    )
                evaluation = re.fullmatch(r"/api/jobs/([a-f0-9]{32})/evaluate", path)
                if evaluation:
                    if length > 262144 or self.headers.get_content_type() != "application/json":
                        raise ValueError("Annotations must be JSON smaller than 256 KB")
                    truth = json.loads(self.rfile.read(length))
                    with jobs.lock:
                        item = jobs.items.get(evaluation[1])
                        ready = item is not None and item["status"] == "completed"
                    if not ready:
                        return self.respond(409, {"error": "Choose a completed analysis"})
                    try:
                        rows = read_records(jobs.root / evaluation[1] / "result.jsonl")
                        if (
                            not isinstance(truth, dict)
                            or not isinstance(truth.get("events"), list)
                            or len(truth["events"]) > 200
                        ):
                            raise ValueError(
                                "Annotations need an events list with at most 200 entries"
                            )
                        metrics = evaluate(rows, truth)
                    except (ValueError, TypeError, KeyError, AttributeError) as exc:
                        raise ValueError("Invalid annotations: " + str(exc)) from exc
                    jobs.store.add_evaluation(evaluation[1], metrics)
                    jobs.bus.notify()
                    return self.respond(200, metrics)
                if path == "/api/vehicle-lookup":
                    if length > 4096 or self.headers.get_content_type() != "application/json":
                        raise ValueError("Expected small JSON request")
                    payload = json.loads(self.rfile.read(length))
                    if not isinstance(payload, dict) or set(payload) != {"country", "identifier"}:
                        raise ValueError("Expected country and identifier")
                    return self.respond(200, lookup(**payload))
                cancel = re.fullmatch(r"/api/jobs/([a-f0-9]{32})/cancel", path)
                retry = re.fullmatch(r"/api/jobs/([a-f0-9]{32})/retry", path)
                if retry:
                    if length > 32:
                        raise ValueError("Retry body too large")
                    self.rfile.read(length)
                    try:
                        item = jobs.retry(retry[1])
                    except ValueError as exc:
                        return self.respond(409, {"error": str(exc)})
                    return self.respond(202, item)
                if cancel:
                    if length > 32:
                        raise ValueError("Cancellation body too large")
                    self.rfile.read(length)
                    return self.respond(
                        202 if jobs.cancel(cancel[1]) else 409,
                        {"message": "Cancellation requested if this job is active"},
                    )
                if path != "/api/jobs":
                    return self.respond(404, {"error": "Not found"})
                extension = self.headers.get("X-File-Extension", "").lower()
                mode = self.headers.get("X-Analysis-Mode", "scene")
                limit = int(self.headers.get("X-Frame-Limit", "60"))
                stride_hdr = self.headers.get("X-Frame-Stride", "15").strip().lower()
                profile = self.headers.get("X-Reasoning-Profile", "default").strip().lower()
                crops_hdr = self.headers.get("X-Vehicle-Crops", "0").strip().lower()
                device_hdr = self.headers.get("X-Device", "cpu").strip().lower()
                if (
                    extension not in EXTENSIONS
                    or mode not in {"scene", "plates"}
                    or not 1 <= limit <= 600
                ):
                    raise ValueError("Unsupported file, mode or frame limit")
                if stride_hdr != "auto":
                    stride_num = int(stride_hdr)
                    if not 1 <= stride_num <= 120:
                        raise ValueError("Frame stride must be 'auto' or an integer in [1, 120]")
                    stride_hdr = str(stride_num)
                if profile not in {"default", "behavior", "zones"}:
                    raise ValueError("Reasoning profile must be default, behavior, or zones")
                if device_hdr not in {"cpu", "auto", "cuda"}:
                    raise ValueError("Compute device must be cpu, auto, or cuda")
                if crops_hdr not in {"0", "1", "false", "true"}:
                    raise ValueError("X-Vehicle-Crops must be 0 or 1")
                vehicle_crops = crops_hdr in {"1", "true"}
                if mode == "plates" and extension in IMAGES:
                    raise ValueError("Plate mode currently needs a video")
                with jobs.lock:
                    if jobs.active:
                        self.close_connection = True
                        return self.respond(
                            409, {"error": "Another analysis is running; wait for it to finish."}
                        )
                    key = uuid4().hex
                    jobs.active = key
                folder = jobs.root / key
                try:
                    folder.mkdir()
                    source = folder / ("input" + extension)
                    with source.open("xb") as stream:
                        remaining = length
                        while remaining:
                            chunk = self.rfile.read(min(1024 * 1024, remaining))
                            if not chunk:
                                raise ValueError("Upload ended early")
                            stream.write(chunk)
                            remaining -= len(chunk)
                    if strict_uploads or not loopback_only:
                        validate_magic_bytes(source, extension)
                    item = {
                        "id": key,
                        "status": "processing",
                        "mode": mode,
                        "limit": limit,
                        "stride": stride_hdr,
                        "reasoning_profile": profile,
                        "vehicle_crops": vehicle_crops,
                        "device": device_hdr,
                        "created": datetime.now(timezone.utc).isoformat(),
                        "bytes": length,
                    }
                    with jobs.lock:
                        jobs.items[key] = item
                        jobs.save(item)
                    jobs.bus.notify()
                    jobs.worker = Thread(target=jobs.run, args=(item, source), daemon=True)
                    jobs.worker.start()
                except Exception:
                    with jobs.lock:
                        if jobs.active == key:
                            jobs.active = None
                        if key in jobs.items:
                            jobs.items[key].update(status="failed", error="Could not start analysis")
                            jobs.save(jobs.items[key])
                        else:
                            shutil.rmtree(folder, ignore_errors=True)
                    jobs.bus.notify()
                    raise
                self.respond(202, dict(item))
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True
            except (ValueError, TypeError) as exc:
                self.close_connection = True
                self.respond(400, {"error": str(exc)})
            except OSError:
                self.close_connection = True
                self.respond(
                    400, {"error": "Local file or connection error while processing request."}
                )

        def log_message(self, *args):
            pass

    class WorkspaceHTTPServer(ThreadingHTTPServer):
        daemon_threads = True

        def server_close(self):
            jobs.close()
            super().server_close()

    server = WorkspaceHTTPServer((host, port), Handler)
    server.jobs = jobs
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8780)
    parser.add_argument("--data", type=Path, default=Path("outputs/workspace"))
    parser.add_argument(
        "--reasoning-config", type=Path, help="Reviewed event/zone configuration for scene videos"
    )
    parser.add_argument(
        "--warm-worker",
        action="store_true",
        help="Keep a persistent worker process so loaded model weights stay warm across jobs",
    )
    parser.add_argument(
        "--strict-uploads",
        action="store_true",
        help="Verify uploaded file magic bytes match their declared extension",
    )
    parser.add_argument(
        "--auth-token-env",
        help="Environment variable containing the bearer/header token (required for non-loopback --host)",
    )
    parser.add_argument("--retention-days", type=float, help="Delete finished jobs older than N days")
    parser.add_argument("--max-disk-mb", type=float, help="Prune oldest finished jobs above this size")
    args = parser.parse_args()
    if not Path("configs/camera.toml").is_file():
        parser.error("Run this command from the Work-mode repository root")
    auth_token = None
    if args.auth_token_env:
        auth_token = os.environ.get(args.auth_token_env, "").strip()
        if not auth_token:
            parser.error(f"Environment variable {args.auth_token_env} is empty or unset")
    if args.host not in LOOPBACK_HOSTS and not auth_token:
        parser.error("--auth-token-env is required when binding to a non-loopback --host")
    if args.reasoning_config:
        try:
            args.reasoning_config = args.reasoning_config.resolve()
            load_reasoning(args.reasoning_config)
        except (OSError, ValueError, TypeError) as exc:
            parser.error(str(exc))
    with make_app(
        args.data,
        Path.cwd(),
        args.port,
        args.reasoning_config,
        host=args.host,
        auth_token=auth_token,
        warm_worker=args.warm_worker,
        strict_uploads=args.strict_uploads,
        retention_days=args.retention_days,
        max_disk_mb=args.max_disk_mb,
    ) as server:
        display_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
        print(f"Workspace: http://{display_host}:{server.server_port}/", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            if server.jobs.active:
                server.jobs.cancel(server.jobs.active)
                if server.jobs.worker:
                    server.jobs.worker.join(timeout=15)


if __name__ == "__main__":
    main()

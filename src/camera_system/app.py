"""Loopback-only upload and analysis workspace. Run from the repository root."""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from threading import Lock, Thread
from urllib.parse import urlsplit
from uuid import uuid4

from .review import HTML, evaluate, read_records
from .vehicles import lookup

MAX_UPLOAD = 512 * 1024 * 1024
EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".jpg", ".jpeg", ".png"}
IMAGES = {".jpg", ".jpeg", ".png"}


def command(source, output, mode, limit, reasoning_config=None):
    if mode == "plates":
        return [
            sys.executable,
            "-m",
            "camera_system.plates",
            str(source),
            "--output",
            str(output),
            "--max-frames",
            str(limit),
        ]
    common = [
        "--output",
        str(output),
        "--include-preview",
        "--analyze-people",
        "--config",
        "configs/camera.toml",
        "--detector-config",
        "configs/detector-office.toml",
    ]
    if source.suffix in IMAGES:
        return [
            sys.executable,
            "-m",
            "camera_system.cli",
            "--image",
            str(source),
            "--detect",
            *common,
        ]
    return [
        sys.executable,
        "-m",
        "camera_system.video_cli",
        "--video",
        str(source),
        "--max-frames",
        str(limit),
        "--stride",
        "15",
        "--relation-every",
        "1",
        *(["--reasoning-config", str(reasoning_config)] if reasoning_config else []),
        *common,
    ]


class Jobs:
    def __init__(self, root, workspace, reasoning_config=None):
        self.root, self.workspace = root.resolve(), workspace.resolve()
        self.reasoning_config = reasoning_config
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = Lock()
        self.active = None
        self.worker = None
        self.items = {}
        for path in self.root.glob("*/job.json"):
            try:
                item = json.loads(path.read_text(encoding="utf-8"))
                if item["status"] in {"processing", "uploading", "cancelling"}:
                    item["status"] = "interrupted"
                self.items[item["id"]] = item
            except (OSError, ValueError, KeyError):
                continue

    def save(self, item):
        folder = self.root / item["id"]
        temporary = folder / "job.tmp"
        temporary.write_text(json.dumps(item), encoding="utf-8")
        temporary.replace(folder / "job.json")

    def snapshot(self):
        with self.lock:
            for item in self.items.values():
                self.update_progress(item)
            items = sorted(
                (dict(item) for item in self.items.values()),
                key=lambda item: item["created"],
                reverse=True,
            )
        for item in items:
            progress = item.pop("progress", {})
            item.update(progress.get("values", {"frames_processed": 0, "coverage": "Coverage not recorded"}))
            item["has_log"] = (self.root / item["id"] / "processing.log").is_file()
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
            and old and not image and not old[3]
            and old[0] == stat.st_ino and old[2] < stat.st_size
        )
        values = dict(previous["values"]) if append else {
            "frames_processed": 0, "coverage": "Coverage not recorded"
        }
        offset = previous["offset"] if append else 0
        if image:
            try:
                record = json.loads(output.read_text(encoding="utf-8"))
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

    def cancel(self, key):
        with self.lock:
            item = self.items.get(key)
            if key != self.active or item is None:
                return False
            item["status"] = "cancelling"
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
            raise

    def run(self, item, source):
        folder = self.root / item["id"]
        output = folder / "result.jsonl"
        try:
            environment = os.environ.copy()
            environment.setdefault("HF_HOME", str(self.workspace / ".hf_cache"))
            environment.setdefault("YOLO_CONFIG_DIR", str(self.workspace / ".yolo"))
            with (folder / "processing.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(
                    command(source, output, item["mode"], item["limit"], self.reasoning_config),
                    cwd=self.workspace,
                    env=environment,
                    stdout=log,
                    stderr=log,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                deadline = time.monotonic() + 3600
                while process.poll() is None:
                    if item["status"] == "cancelling" or time.monotonic() > deadline:
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
                        if item["status"] == "cancelling":
                            with self.lock:
                                item["status"] = "cancelled"
                            return
                        raise RuntimeError("Analysis exceeded the one-hour job limit.")
                    try:
                        process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        pass
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


def make_app(root, workspace, port=8780, reasoning_config=None):
    jobs = Jobs(root, workspace, reasoning_config)
    home = files("camera_system").joinpath("templates/workspace.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def local(self, write=False):
            host = self.headers.get("Host")
            allowed = {
                f"127.0.0.1:{self.server.server_port}",
                f"localhost:{self.server.server_port}",
            }
            return host in allowed and (not write or self.headers.get("Origin") == f"http://{host}")

        def respond(self, status, data, content_type="application/json"):
            body = data if isinstance(data, bytes) else json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

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
            media = re.fullmatch(r"/jobs/([a-f0-9]{32})/media", path)
            if media:
                sources = [jobs.root / media[1] / ("input" + ext) for ext in EXTENSIONS]
                sources = [source for source in sources if source.is_file()]
                if len(sources) != 1:
                    return self.respond(404, {"error": "Saved media is unavailable"})
                source = sources[0]
                size = source.stat().st_size
                start, end = 0, size - 1
                requested = self.headers.get("Range")
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
                    ".mp4": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm",
                    ".avi": "video/x-msvideo", ".mkv": "video/x-matroska",
                    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
                }
                self.send_response(206 if requested else 200)
                self.send_header("Content-Type", mime[source.suffix])
                self.send_header("Content-Length", str(end - start + 1))
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                if requested:
                    self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.end_headers()
                try:
                    with source.open("rb") as stream:
                        stream.seek(start)
                        remaining = end - start + 1
                        while remaining:
                            chunk = stream.read(min(1024 * 1024, remaining))
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            remaining -= len(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            match = re.fullmatch(r"/jobs/([a-f0-9]{32})/(review|result|log)", path)
            if match:
                key, kind = match.groups()
                name = {"review": "review.html", "result": "result.jsonl", "log": "processing.log"}[
                    kind
                ]
                target = jobs.root / key / name
                if target.is_file():
                    if kind == "review":
                        try:
                            rows = read_records(jobs.root / key / "result.jsonl")
                        except (OSError, ValueError):
                            return self.respond(409, {"error": "Saved results are unavailable or invalid"})
                        page = HTML.replace(
                            "__DATA__", json.dumps(rows, allow_nan=False).replace("<", "\\u003c")
                        )
                        return self.respond(200, page.encode("utf-8"), "text/html; charset=utf-8")
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

        def do_POST(self):
            if not self.local(write=True):
                self.close_connection = True
                return self.respond(403, {"error": "Same-origin local access required"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if self.headers.get("Transfer-Encoding") or not 0 < length <= MAX_UPLOAD:
                    raise ValueError("Upload must have a known size between 1 byte and 512 MB.")
                evaluation = re.fullmatch(r"/api/jobs/([a-f0-9]{32})/evaluate", self.path)
                if evaluation:
                    if length > 262144 or self.headers.get_content_type() != "application/json":
                        raise ValueError("Annotations must be JSON smaller than 256 KB")
                    self.connection.settimeout(30)
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
                    return self.respond(200, metrics)
                if self.path == "/api/vehicle-lookup":
                    if length > 4096 or self.headers.get_content_type() != "application/json":
                        raise ValueError("Expected small JSON request")
                    payload = json.loads(self.rfile.read(length))
                    if not isinstance(payload, dict) or set(payload) != {"country", "identifier"}:
                        raise ValueError("Expected country and identifier")
                    return self.respond(200, lookup(**payload))
                cancel = re.fullmatch(r"/api/jobs/([a-f0-9]{32})/cancel", self.path)
                retry = re.fullmatch(r"/api/jobs/([a-f0-9]{32})/retry", self.path)
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
                if self.path != "/api/jobs":
                    return self.respond(404, {"error": "Not found"})
                extension = self.headers.get("X-File-Extension", "").lower()
                mode = self.headers.get("X-Analysis-Mode", "scene")
                limit = int(self.headers.get("X-Frame-Limit", "60"))
                if (
                    extension not in EXTENSIONS
                    or mode not in {"scene", "plates"}
                    or not 1 <= limit <= 600
                ):
                    raise ValueError("Unsupported file, mode or frame limit")
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
                    self.connection.settimeout(30)
                    with source.open("xb") as stream:
                        remaining = length
                        while remaining:
                            chunk = self.rfile.read(min(1024 * 1024, remaining))
                            if not chunk:
                                raise ValueError("Upload ended early")
                            stream.write(chunk)
                            remaining -= len(chunk)
                    item = {
                        "id": key,
                        "status": "processing",
                        "mode": mode,
                        "limit": limit,
                        "created": datetime.now(timezone.utc).isoformat(),
                        "bytes": length,
                    }
                    with jobs.lock:
                        jobs.items[key] = item
                        jobs.save(item)
                    jobs.worker = Thread(target=jobs.run, args=(item, source), daemon=True)
                    jobs.worker.start()
                except Exception:
                    with jobs.lock:
                        if jobs.active == key:
                            jobs.active = None
                        if key in jobs.items:
                            jobs.items[key].update(status="failed", error="Could not start analysis")
                            jobs.save(jobs.items[key])
                    raise
                self.respond(202, dict(item))
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True
            except (ValueError, TypeError, OSError) as exc:
                self.close_connection = True
                self.respond(400, {"error": str(exc)})

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.jobs = jobs
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8780)
    parser.add_argument("--data", type=Path, default=Path("outputs/workspace"))
    parser.add_argument(
        "--reasoning-config", type=Path, help="Reviewed event/zone configuration for scene videos"
    )
    args = parser.parse_args()
    if not Path("configs/camera.toml").is_file():
        parser.error("Run this command from the Work-mode repository root")
    if args.reasoning_config:
        from .reasoning import load_reasoning

        try:
            args.reasoning_config = args.reasoning_config.resolve()
            load_reasoning(args.reasoning_config)
        except (OSError, ValueError, TypeError) as exc:
            parser.error(str(exc))
    with make_app(args.data, Path.cwd(), args.port, args.reasoning_config) as server:
        print(f"Workspace: http://127.0.0.1:{server.server_port}/", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            if server.jobs.active:
                server.jobs.cancel(server.jobs.active)
                if server.jobs.worker:
                    server.jobs.worker.join(timeout=15)


if __name__ == "__main__":
    main()

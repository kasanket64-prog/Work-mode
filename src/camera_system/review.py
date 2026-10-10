"""Local, offline result review and event/relation evaluation."""

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from importlib.resources import files
from math import isclose, isfinite
from pathlib import Path

from .vehicles import lookup


def read_records(path):
    raw = path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "regions" in data:
            rows = [
                {"type": "session", "source_kind": "image"},
                dict(data, type="frame", frame_index=0),
                {"type": "summary", "status": data.get("status", "unknown"), "frames_processed": 1},
            ]
        else:
            rows = data
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not isinstance(rows, list):
        raise ValueError("expected a session or image result")
    if not rows or rows[0].get("type") != "session":
        raise ValueError("expected a camera-video JSONL session")
    return link_event_frames(rows)


def link_event_frames(rows):
    """Attach review frame indices without altering the saved model observations."""
    frames = [r for r in rows if r.get("type") == "frame"]
    output = []
    for row in rows:
        if row.get("type") != "frame":
            output.append(row)
            continue
        events = []
        for event in row.get("events", []):
            end = event.get("timestamp_seconds")
            start = event.get("started_seconds", event.get("transition_started_seconds", end))
            required = {event.get("object_track_id"), event.get("subject_track_id")} - {None}
            indices = []
            if (
                type(start) in (int, float)
                and type(end) in (int, float)
                and isfinite(start)
                and isfinite(end)
                and 0 <= start <= end
            ):
                for frame in frames:
                    timestamp = frame.get("timestamp_seconds")
                    index = frame.get("frame_index")
                    if (
                        type(timestamp) in (int, float)
                        and start <= timestamp <= end
                        and type(index) is int
                        and index >= 0
                        and required
                        and required <= {r.get("track_id") for r in frame.get("regions", [])}
                    ):
                        indices.append(index)
            events.append(dict(event, review_frame_indices=indices))
        output.append(dict(row, events=events))
    return output


def annotation_template(rows):
    """Export navigation references, never model predictions as truth."""
    if rows[0].get("relation_endpoint_kind") != "track_id":
        raise ValueError("annotation drafts require a tracked video session")
    frames = []
    for row in rows:
        if row.get("type") != "frame":
            continue
        index, timestamp = row.get("frame_index"), row.get("timestamp_seconds")
        if (
            type(index) is not int or index < 0
            or type(timestamp) not in (int, float)
            or not isfinite(timestamp) or timestamp < 0
        ):
            raise ValueError("annotation drafts require valid frame indices and timestamps")
        frames.append({
            "frame_index": index,
            "timestamp_seconds": timestamp,
            "visible_track_ids": sorted({
                r["track_id"] for r in row.get("regions", [])
                if isinstance(r.get("track_id"), str) and r["track_id"]
            }),
        })
    if not frames:
        raise ValueError("annotation drafts require at least one sampled video frame")
    return {
        "annotation_status": "draft",
        "instructions": (
            "Review the original footage independently, including missed events. "
            "Review exactly the evaluation_interval_seconds (first to last sampled timestamp). "
            "Set duration_seconds to that interval length and events to your labels "
            "([] only for verified negatives). Map predicted track IDs to reviewed identities. "
            "Set annotation_status to reviewed only when complete. "
            "reference_frames are navigation aids, not annotations or proof of full coverage."
        ),
        "duration_seconds": None,
        "evaluation_interval_seconds": [frames[0]["timestamp_seconds"], frames[-1]["timestamp_seconds"]],
        "track_mapping": {},
        "events": None,
        "reference_frames": frames,
    }


def metrics(tp, fp, fn):
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
    }


def evaluate(rows, truth, tolerance=2.0):
    if "annotation_status" in truth and truth["annotation_status"] != "reviewed":
        raise ValueError("annotation draft must be reviewed before evaluation")
    duration = truth["duration_seconds"]
    if (
        type(duration) not in (int, float) or not isfinite(duration) or duration <= 0
        or type(tolerance) not in (int, float) or not isfinite(tolerance) or tolerance < 0
    ):
        raise ValueError("duration must be positive; tolerance must be nonnegative")
    frames = [r for r in rows if r.get("type") == "frame"]
    if not frames:
        raise ValueError("evaluation requires at least one sampled video frame")
    if ("relations" in truth) != ("annotated_frame_indices" in truth):
        raise ValueError("relations and annotated_frame_indices must be supplied together")
    for frame in frames:
        timestamp = frame.get("timestamp_seconds")
        if type(timestamp) not in (int, float) or not isfinite(timestamp) or timestamp < 0:
            raise ValueError(
                "annotations must cover all sampled video timestamps; images need video-independent evaluation"
            )
    times = [f["timestamp_seconds"] for f in frames]
    if len(times) < 2 or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("evaluation requires at least two increasing sampled timestamps")
    interval = truth.get("evaluation_interval_seconds", [0, duration])
    if (
        not isinstance(interval, list) or len(interval) != 2
        or any(type(t) not in (int, float) or not isfinite(t) for t in interval)
        or not isclose(interval[0], times[0], rel_tol=0, abs_tol=1e-6)
        or not isclose(interval[1], times[-1], rel_tol=0, abs_tol=1e-6)
        or not isclose(duration, times[-1] - times[0], rel_tol=0, abs_tol=1e-6)
    ):
        raise ValueError("annotation duration and evaluation_interval_seconds must match the first-to-last sampled interval")
    summary = next((r for r in reversed(rows) if r.get("type") == "summary"), {})
    if (
        summary.get("end_reason") == "max_frames"
        or summary.get("status") in {"failed", "interrupted", "cancelled"}
    ) and "evaluation_interval_seconds" not in truth:
        raise ValueError("partial runs require explicit evaluation_interval_seconds annotations")
    predicted = [e for frame in frames for e in frame.get("events", [])]
    expected = truth["events"]
    mapping = truth.get("track_mapping", {})

    def identity(event, predicted=False):
        key = event["object_track_id"]
        subject = event.get("subject_track_id")
        return (
            event["type"],
            mapping.get(key, key) if predicted else key,
            event.get("zone_id"),
            mapping.get(subject, subject) if predicted else subject,
        )

    for event in predicted + expected:
        t = event["timestamp_seconds"]
        if type(t) not in (int, float) or not isfinite(t) or not times[0] <= t <= times[-1]:
            raise ValueError("event timestamp lies outside annotated duration")
    # Maximum cardinality one-to-one matching; duplicate alerts cannot reuse an annotation.
    edges = [
        [
            j
            for j, target in enumerate(expected)
            if identity(event, True) == identity(target)
            and abs(event["timestamp_seconds"] - target["timestamp_seconds"]) <= tolerance
        ]
        for event in predicted
    ]
    matches = {}

    def match(i, visited):
        for j in edges[i]:
            if j not in visited:
                visited.add(j)
                if j not in matches or match(matches[j], visited):
                    matches[j] = i
                    return True
        return False

    tp = sum(match(i, set()) for i in range(len(predicted)))
    result = {
        "events": metrics(tp, len(predicted) - tp, len(expected) - tp),
        "false_alarms_per_hour": (len(predicted) - tp) * 3600 / duration,
        "duration_seconds": duration,
        "evaluation_interval_seconds": [times[0], times[-1]],
        "coverage_basis": "first_to_last_sampled_timestamp",
        "event_tolerance_seconds": tolerance,
    }
    if "annotated_frame_indices" in truth:
        if (
            not isinstance(truth["annotated_frame_indices"], list)
            or not truth["annotated_frame_indices"]
            or any(type(i) is not int or i < 0 for i in truth["annotated_frame_indices"])
        ):
            raise ValueError("annotated_frame_indices must be a nonempty list of nonnegative integers")
        indices = set(truth["annotated_frame_indices"])
        if not indices <= {f["frame_index"] for f in frames}:
            raise ValueError("annotations include frames absent from this run")

        def relation_key(frame, r, remap=False):
            return (
                frame,
                mapping.get(r["subject_id"], r["subject_id"]) if remap else r["subject_id"],
                r["predicate"],
                mapping.get(r["object_id"], r["object_id"]) if remap else r["object_id"],
            )

        observed = {
            relation_key(f["frame_index"], r, True)
            for f in frames
            if f["frame_index"] in indices
            for r in f.get("confirmed_relations", [])
        }
        target = {relation_key(r["frame_index"], r) for r in truth["relations"]}
        if any(r[0] not in indices for r in target):
            raise ValueError("relation label outside annotated_frame_indices")
        result["confirmed_relations"] = metrics(
            len(observed & target), len(observed - target), len(target - observed)
        )
    return result


HTML = files("camera_system").joinpath("templates/review.html").read_text(encoding="utf-8")
HTML = HTML.replace(
    "__CAMERA_SOURCES__",
    files("camera_system")
    .joinpath("templates/camera_sources.json")
    .read_text(encoding="utf-8")
    .replace("<", "\\u003c"),
)


def make_server(rows, port=8765):
    """Serve only this loaded session, never arbitrary filesystem paths."""
    page = HTML.replace(
        "__DATA__", json.dumps(rows, allow_nan=False).replace("<", "\\u003c")
    ).encode("utf-8")
    data = json.dumps(rows, allow_nan=False).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/api/vehicle-lookup":
                self.send_error(404)
                return
            hosts = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            host = self.headers.get("Host")
            try:
                self.connection.settimeout(5)
                length = int(self.headers.get("Content-Length", "0"))
                if self.headers.get("Transfer-Encoding") or not 0 < length <= 4096:
                    raise ValueError("Expected a small JSON request.")
                raw = self.rfile.read(length)
                # Consume the bounded body before rejecting it so Windows can deliver the response.
                if host not in hosts or self.headers.get("Origin") != f"http://{host}":
                    self.send_error(403)
                    return
                if self.headers.get_content_type() != "application/json":
                    raise ValueError("Expected a small JSON request.")
                payload = json.loads(raw)
                if not isinstance(payload, dict) or set(payload) != {"country", "identifier"}:
                    raise ValueError("Expected country and identifier.")
                result = lookup(payload["country"], payload["identifier"])
                status = 200
            except (ValueError, TypeError, OSError):
                result, status = (
                    {
                        "status": "invalid_request",
                        "message": "Check the country and plate/VIN format.",
                    },
                    400,
                )
            body = json.dumps(result).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            hosts = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            if self.headers.get("Host") not in hosts:
                self.send_error(403)
                return
            if self.path not in {"/", "/api/session"}:
                self.send_error(404)
                return
            body = data if self.path == "/api/session" else page
            self.send_response(200)
            self.send_header(
                "Content-Type",
                "application/json; charset=utf-8"
                if self.path == "/api/session"
                else "text/html; charset=utf-8",
            )
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    return HTTPServer(("127.0.0.1", port), Handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Review or evaluate a local camera-video session")
    parser.add_argument("session", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--truth", type=Path, help="Annotations JSON; omit to generate HTML review")
    mode.add_argument("--annotation-template", action="store_true", help="Export an unreviewed draft")
    parser.add_argument("--tolerance", type=float, default=2.0)
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Serve this snapshot on localhost with a read-only JSON API",
    )
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    try:
        if args.serve and (args.truth or args.annotation_template or not 1 <= args.port <= 65535):
            raise ValueError("--serve requires HTML mode and a port in [1, 65535]")
        rows = read_records(args.session)
        if args.annotation_template:
            payload = json.dumps(annotation_template(rows), indent=2, allow_nan=False)
        elif args.truth:
            payload = json.dumps(
                evaluate(rows, json.loads(args.truth.read_text(encoding="utf-8")), args.tolerance),
                indent=2,
                allow_nan=False,
            )
        else:
            payload = HTML.replace(
                "__DATA__", json.dumps(rows, allow_nan=False).replace("<", "\\u003c")
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(payload)
        print(args.output)
        if args.serve:
            with make_server(rows, args.port) as server:
                print(f"Review: http://127.0.0.1:{args.port}/ ; API: /api/session", flush=True)
                try:
                    server.serve_forever()
                except KeyboardInterrupt:
                    pass
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"camera-review: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

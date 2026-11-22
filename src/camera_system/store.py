"""SQLite persistence layer for Fieldnote workspace jobs, verdicts, zones, and evaluations."""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from .reasoning import ReasoningSettings

VALID_VERDICTS = {"", "correct", "incorrect", "uncertain"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class WorkspaceStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self.lock:
            self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    frame_limit INTEGER NOT NULL,
                    stride TEXT NOT NULL DEFAULT '15',
                    reasoning_profile TEXT NOT NULL DEFAULT 'default',
                    vehicle_crops INTEGER NOT NULL DEFAULT 0,
                    device TEXT NOT NULL DEFAULT 'cpu',
                    created TEXT NOT NULL,
                    bytes INTEGER NOT NULL DEFAULT 0,
                    retry_of TEXT,
                    error TEXT,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS verdicts (
                    job_id TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    verdict TEXT NOT NULL,
                    note TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (job_id, event_id)
                );
                CREATE TABLE IF NOT EXISTS zones (
                    id TEXT PRIMARY KEY,
                    box_json TEXT NOT NULL,
                    labels_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS annotations (
                    job_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            self._conn.commit()

    def close(self) -> None:
        with self.lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def upsert_job(self, item: dict) -> None:
        payload = json.dumps(item, allow_nan=False)
        with self.lock:
            self._conn.execute(
                """
                INSERT INTO jobs (
                    id, status, mode, frame_limit, stride, reasoning_profile,
                    vehicle_crops, device, created, bytes, retry_of, error, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    status = excluded.status,
                    mode = excluded.mode,
                    frame_limit = excluded.frame_limit,
                    stride = excluded.stride,
                    reasoning_profile = excluded.reasoning_profile,
                    vehicle_crops = excluded.vehicle_crops,
                    device = excluded.device,
                    created = excluded.created,
                    bytes = excluded.bytes,
                    retry_of = excluded.retry_of,
                    error = excluded.error,
                    payload_json = excluded.payload_json
                """,
                (
                    str(item["id"]),
                    str(item.get("status", "unknown")),
                    str(item.get("mode", "scene")),
                    int(item.get("limit", 30)),
                    str(item.get("stride", "15")),
                    str(item.get("reasoning_profile", "default")),
                    1 if item.get("vehicle_crops") else 0,
                    str(item.get("device", "cpu")),
                    str(item.get("created", "")),
                    int(item.get("bytes", 0)),
                    item.get("retry_of"),
                    item.get("error"),
                    payload,
                ),
            )
            self._conn.commit()

    def delete_job(self, job_id: str) -> None:
        with self.lock:
            self._conn.execute("DELETE FROM verdicts WHERE job_id = ?", (job_id,))
            self._conn.execute("DELETE FROM annotations WHERE job_id = ?", (job_id,))
            self._conn.execute("DELETE FROM evaluations WHERE job_id = ?", (job_id,))
            self._conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            self._conn.commit()

    def list_jobs(self) -> list[dict]:
        with self.lock:
            rows = self._conn.execute(
                "SELECT payload_json FROM jobs ORDER BY created DESC"
            ).fetchall()
        return [json.loads(row["payload_json"]) for row in rows]

    def get_verdicts(self, job_id: str) -> dict[str, dict]:
        with self.lock:
            rows = self._conn.execute(
                "SELECT event_id, verdict, note, updated_at FROM verdicts WHERE job_id = ?",
                (job_id,),
            ).fetchall()
        return {
            row["event_id"]: {
                "verdict": row["verdict"],
                "note": row["note"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        }

    def set_verdicts(self, job_id: str, verdicts: dict) -> dict[str, dict]:
        if not isinstance(verdicts, dict) or len(verdicts) > 1000:
            raise ValueError("verdicts must be a dictionary of at most 1000 event entries")
        now = _now()
        cleaned = []
        for event_id, entry in verdicts.items():
            if not isinstance(event_id, str) or not 1 <= len(event_id) <= 64:
                raise ValueError("event_id must be a short nonempty string")
            if isinstance(entry, str):
                verdict, note = entry, ""
            elif isinstance(entry, dict):
                verdict = entry.get("verdict", "")
                note = entry.get("note", "")
            else:
                raise ValueError("verdict entry must be a string or object")
            if not isinstance(verdict, str) or verdict not in VALID_VERDICTS:
                raise ValueError("verdict must be '', 'correct', 'incorrect', or 'uncertain'")
            if not isinstance(note, str) or len(note) > 500:
                raise ValueError("verdict note must be a string of at most 500 characters")
            cleaned.append((event_id, verdict, note.strip()))

        with self.lock:
            for event_id, verdict, note in cleaned:
                if not verdict and not note:
                    self._conn.execute(
                        "DELETE FROM verdicts WHERE job_id = ? AND event_id = ?",
                        (job_id, event_id),
                    )
                else:
                    self._conn.execute(
                        """
                        INSERT INTO verdicts (job_id, event_id, verdict, note, updated_at)
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT(job_id, event_id) DO UPDATE SET
                            verdict = excluded.verdict,
                            note = excluded.note,
                            updated_at = excluded.updated_at
                        """,
                        (job_id, event_id, verdict, note, now),
                    )
            self._conn.commit()
        return self.get_verdicts(job_id)

    def list_zones(self) -> list[dict]:
        with self.lock:
            rows = self._conn.execute(
                "SELECT id, box_json, labels_json FROM zones ORDER BY id ASC"
            ).fetchall()
        return [
            {
                "id": row["id"],
                "box": json.loads(row["box_json"]),
                "labels": json.loads(row["labels_json"]),
            }
            for row in rows
        ]

    def upsert_zone(self, zone: dict) -> dict:
        if not isinstance(zone, dict):
            raise ValueError("zone must be an object with id, box, and labels")
        normalized = {
            "id": zone.get("id"),
            "box": zone.get("box"),
            "labels": zone.get("labels"),
        }
        ReasoningSettings(zones=[normalized])
        existing = [z for z in self.list_zones() if z["id"] != normalized["id"]]
        if len(existing) >= 20:
            raise ValueError("zones must contain at most 20 rectangles")
        now = _now()
        with self.lock:
            self._conn.execute(
                """
                INSERT INTO zones (id, box_json, labels_json, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    box_json = excluded.box_json,
                    labels_json = excluded.labels_json,
                    updated_at = excluded.updated_at
                """,
                (
                    normalized["id"],
                    json.dumps([float(v) for v in normalized["box"]]),
                    json.dumps(list(normalized["labels"])),
                    now,
                ),
            )
            self._conn.commit()
        return {
            "id": normalized["id"],
            "box": [float(v) for v in normalized["box"]],
            "labels": list(normalized["labels"]),
        }

    def delete_zone(self, zone_id: str) -> bool:
        with self.lock:
            cursor = self._conn.execute("DELETE FROM zones WHERE id = ?", (zone_id,))
            self._conn.commit()
            return cursor.rowcount > 0

    def get_annotation(self, job_id: str) -> dict | None:
        with self.lock:
            row = self._conn.execute(
                "SELECT payload_json FROM annotations WHERE job_id = ?", (job_id,)
            ).fetchone()
        return json.loads(row["payload_json"]) if row else None

    def save_annotation(self, job_id: str, payload: dict) -> dict:
        if not isinstance(payload, dict):
            raise ValueError("annotations payload must be a JSON object")
        encoded = json.dumps(payload, allow_nan=False)
        if len(encoded.encode("utf-8")) > 262144:
            raise ValueError("annotations must be smaller than 256 KB")
        now = _now()
        with self.lock:
            self._conn.execute(
                """
                INSERT INTO annotations (job_id, payload_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    updated_at = excluded.updated_at
                """,
                (job_id, encoded, now),
            )
            self._conn.commit()
        return payload

    def add_evaluation(self, job_id: str, metrics: dict) -> dict:
        encoded = json.dumps(metrics, allow_nan=False)
        now = _now()
        with self.lock:
            cursor = self._conn.execute(
                "INSERT INTO evaluations (job_id, metrics_json, created_at) VALUES (?, ?, ?)",
                (job_id, encoded, now),
            )
            self._conn.commit()
            row_id = cursor.lastrowid
        return {"id": row_id, "job_id": job_id, "created_at": now, "metrics": metrics}

    def list_evaluations(self, job_id: str) -> list[dict]:
        with self.lock:
            rows = self._conn.execute(
                "SELECT id, job_id, metrics_json, created_at FROM evaluations "
                "WHERE job_id = ? ORDER BY id DESC",
                (job_id,),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "job_id": row["job_id"],
                "created_at": row["created_at"],
                "metrics": json.loads(row["metrics_json"]),
            }
            for row in rows
        ]

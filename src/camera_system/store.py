"""SQLite persistence layer for Fieldnote jobs, verdicts, zones, cases, bookmarks, and audit."""

import hashlib
import json
import re
import sqlite3
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from threading import Lock
from uuid import uuid4

from .reasoning import ReasoningSettings

VALID_VERDICTS = {"", "correct", "incorrect", "uncertain"}
VALID_CASE_STATUSES = {"open", "under_review", "closed"}
VALID_BOOKMARK_TAGS = {
    "arrival",
    "handoff",
    "vehicle",
    "departure",
    "suspect_lead",
    "observation",
}
VALID_ENTITY_KINDS = {"person", "vehicle", "object"}
GENESIS_HASH = "0" * 64


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
                CREATE TABLE IF NOT EXISTS cases (
                    id TEXT PRIMARY KEY,
                    case_number TEXT NOT NULL,
                    title TEXT NOT NULL,
                    investigator TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'open',
                    notes TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS bookmarks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    case_id TEXT,
                    frame_index INTEGER NOT NULL,
                    timestamp_seconds REAL NOT NULL,
                    wall_clock TEXT,
                    tag TEXT NOT NULL DEFAULT 'observation',
                    caption TEXT NOT NULL,
                    investigator TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS entities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_id TEXT NOT NULL,
                    label TEXT NOT NULL,
                    kind TEXT NOT NULL DEFAULT 'person',
                    description TEXT NOT NULL DEFAULT '',
                    linked_tracks_json TEXT NOT NULL DEFAULT '[]',
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    action TEXT NOT NULL,
                    target_type TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    details_json TEXT NOT NULL,
                    prev_hash TEXT NOT NULL,
                    entry_hash TEXT NOT NULL
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
            self._conn.execute("DELETE FROM bookmarks WHERE job_id = ?", (job_id,))
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

    @staticmethod
    def _case_row(row: sqlite3.Row) -> dict:
        return {
            "id": row["id"],
            "case_number": row["case_number"],
            "title": row["title"],
            "investigator": row["investigator"],
            "status": row["status"],
            "notes": row["notes"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def list_cases(self) -> list[dict]:
        with self.lock:
            rows = self._conn.execute(
                "SELECT * FROM cases ORDER BY updated_at DESC, created_at DESC"
            ).fetchall()
        return [self._case_row(r) for r in rows]

    def get_case(self, case_id: str) -> dict | None:
        with self.lock:
            row = self._conn.execute("SELECT * FROM cases WHERE id = ?", (case_id,)).fetchone()
        return self._case_row(row) if row else None

    def create_case(self, payload: dict) -> dict:
        if not isinstance(payload, dict):
            raise ValueError("Case payload must be a JSON object.")
        raw_id = payload.get("id")
        if raw_id is not None:
            case_id = str(raw_id).strip()
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", case_id):
                raise ValueError("Case ID must match [A-Za-z0-9_-]{1,40}.")
        else:
            case_id = f"case-{uuid4().hex[:10]}"
        title = str(payload.get("title") or "").strip()
        if not 1 <= len(title) <= 140:
            raise ValueError("Case title must be 1 to 140 characters.")
        case_number = str(payload.get("case_number") or case_id.upper()).strip()
        if not 1 <= len(case_number) <= 60:
            raise ValueError("Case number must be 1 to 60 characters.")
        investigator = str(payload.get("investigator") or "").strip()
        if len(investigator) > 100:
            raise ValueError("Investigator name must be at most 100 characters.")
        status = str(payload.get("status") or "open").strip().lower()
        if status not in VALID_CASE_STATUSES:
            raise ValueError("Case status must be open, under_review, or closed.")
        notes = str(payload.get("notes") or "").strip()
        if len(notes) > 2000:
            raise ValueError("Case notes must be at most 2000 characters.")
        now = _now()
        with self.lock:
            existing = self._conn.execute("SELECT 1 FROM cases WHERE id = ?", (case_id,)).fetchone()
            if existing:
                raise ValueError(f"Case ID {case_id} already exists.")
            self._conn.execute(
                """
                INSERT INTO cases (
                    id, case_number, title, investigator, status, notes, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (case_id, case_number, title, investigator, status, notes, now, now),
            )
            self._conn.commit()
        return {
            "id": case_id,
            "case_number": case_number,
            "title": title,
            "investigator": investigator,
            "status": status,
            "notes": notes,
            "created_at": now,
            "updated_at": now,
        }

    def update_case(self, case_id: str, payload: dict) -> dict:
        if not isinstance(payload, dict):
            raise ValueError("Case update payload must be a JSON object.")
        current = self.get_case(case_id)
        if current is None:
            raise KeyError(case_id)
        case_number = str(payload.get("case_number", current["case_number"])).strip()
        title = str(payload.get("title", current["title"])).strip()
        investigator = str(payload.get("investigator", current["investigator"])).strip()
        status = str(payload.get("status", current["status"])).strip().lower()
        notes = str(payload.get("notes", current["notes"])).strip()
        if not 1 <= len(title) <= 140:
            raise ValueError("Case title must be 1 to 140 characters.")
        if not 1 <= len(case_number) <= 60:
            raise ValueError("Case number must be 1 to 60 characters.")
        if len(investigator) > 100:
            raise ValueError("Investigator name must be at most 100 characters.")
        if status not in VALID_CASE_STATUSES:
            raise ValueError("Case status must be open, under_review, or closed.")
        if len(notes) > 2000:
            raise ValueError("Case notes must be at most 2000 characters.")
        now = _now()
        with self.lock:
            self._conn.execute(
                """
                UPDATE cases
                SET case_number = ?, title = ?, investigator = ?, status = ?,
                    notes = ?, updated_at = ?
                WHERE id = ?
                """,
                (case_number, title, investigator, status, notes, now, case_id),
            )
            self._conn.commit()
        return {
            "id": case_id,
            "case_number": case_number,
            "title": title,
            "investigator": investigator,
            "status": status,
            "notes": notes,
            "created_at": current["created_at"],
            "updated_at": now,
        }

    def delete_case(self, case_id: str) -> bool:
        with self.lock:
            self._conn.execute("DELETE FROM entities WHERE case_id = ?", (case_id,))
            self._conn.execute(
                "UPDATE bookmarks SET case_id = NULL WHERE case_id = ?", (case_id,)
            )
            cursor = self._conn.execute("DELETE FROM cases WHERE id = ?", (case_id,))
            self._conn.commit()
            return cursor.rowcount > 0

    @staticmethod
    def _bookmark_row(row: sqlite3.Row) -> dict:
        return {
            "id": row["id"],
            "job_id": row["job_id"],
            "case_id": row["case_id"],
            "frame_index": row["frame_index"],
            "timestamp_seconds": row["timestamp_seconds"],
            "wall_clock": row["wall_clock"],
            "tag": row["tag"],
            "caption": row["caption"],
            "investigator": row["investigator"],
            "created_at": row["created_at"],
        }

    def list_bookmarks(
        self, job_id: str | None = None, case_id: str | None = None
    ) -> list[dict]:
        with self.lock:
            if job_id is not None:
                rows = self._conn.execute(
                    "SELECT * FROM bookmarks WHERE job_id = ? ORDER BY timestamp_seconds ASC, id ASC",
                    (job_id,),
                ).fetchall()
            elif case_id is not None:
                rows = self._conn.execute(
                    "SELECT * FROM bookmarks WHERE case_id = ? ORDER BY wall_clock ASC, timestamp_seconds ASC, id ASC",
                    (case_id,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM bookmarks ORDER BY created_at DESC, id DESC"
                ).fetchall()
        return [self._bookmark_row(r) for r in rows]

    def add_bookmark(self, job_id: str, payload: dict) -> dict:
        if not isinstance(payload, dict):
            raise ValueError("Bookmark payload must be a JSON object.")
        frame_index = payload.get("frame_index")
        if type(frame_index) is not int or frame_index < 0:
            raise ValueError("Bookmark frame_index must be a nonnegative integer.")
        ts = payload.get("timestamp_seconds")
        if type(ts) not in (int, float) or not isfinite(ts) or ts < 0:
            raise ValueError("Bookmark timestamp_seconds must be finite and nonnegative.")
        tag = str(payload.get("tag") or "observation").strip().lower()
        if tag not in VALID_BOOKMARK_TAGS:
            raise ValueError("Unsupported bookmark tag.")
        caption = str(payload.get("caption") or "").strip()
        if not 1 <= len(caption) <= 500:
            raise ValueError("Bookmark caption must be 1 to 500 characters.")
        case_id = payload.get("case_id")
        if case_id is not None:
            case_id = str(case_id).strip() or None
        wall_clock = payload.get("wall_clock")
        if wall_clock is not None:
            wall_clock = str(wall_clock).strip()[:64] or None
        investigator = str(payload.get("investigator") or "").strip()
        if len(investigator) > 100:
            raise ValueError("Investigator name must be at most 100 characters.")
        now = _now()
        with self.lock:
            cursor = self._conn.execute(
                """
                INSERT INTO bookmarks (
                    job_id, case_id, frame_index, timestamp_seconds, wall_clock,
                    tag, caption, investigator, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    case_id,
                    frame_index,
                    float(ts),
                    wall_clock,
                    tag,
                    caption,
                    investigator,
                    now,
                ),
            )
            self._conn.commit()
            row_id = cursor.lastrowid
        return {
            "id": row_id,
            "job_id": job_id,
            "case_id": case_id,
            "frame_index": frame_index,
            "timestamp_seconds": float(ts),
            "wall_clock": wall_clock,
            "tag": tag,
            "caption": caption,
            "investigator": investigator,
            "created_at": now,
        }

    def delete_bookmark(self, bookmark_id: int) -> bool:
        with self.lock:
            cursor = self._conn.execute("DELETE FROM bookmarks WHERE id = ?", (bookmark_id,))
            self._conn.commit()
            return cursor.rowcount > 0

    def list_entities(self, case_id: str) -> list[dict]:
        with self.lock:
            rows = self._conn.execute(
                "SELECT * FROM entities WHERE case_id = ? ORDER BY id ASC",
                (case_id,),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "case_id": row["case_id"],
                "label": row["label"],
                "kind": row["kind"],
                "description": row["description"],
                "linked_tracks": json.loads(row["linked_tracks_json"]),
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    def upsert_entity(self, case_id: str, payload: dict) -> dict:
        if not isinstance(payload, dict):
            raise ValueError("Entity payload must be a JSON object.")
        if self.get_case(case_id) is None:
            raise KeyError(case_id)
        label = str(payload.get("label") or "").strip()
        if not 1 <= len(label) <= 100:
            raise ValueError("Entity label must be 1 to 100 characters.")
        kind = str(payload.get("kind") or "person").strip().lower()
        if kind not in VALID_ENTITY_KINDS:
            raise ValueError("Entity kind must be person, vehicle, or object.")
        description = str(payload.get("description") or "").strip()
        if len(description) > 1000:
            raise ValueError("Entity description must be at most 1000 characters.")
        raw_links = payload.get("linked_tracks", [])
        if not isinstance(raw_links, list) or len(raw_links) > 100:
            raise ValueError("linked_tracks must be a list of at most 100 references.")
        links = []
        for entry in raw_links:
            if not isinstance(entry, dict):
                raise ValueError("Each linked track must be an object.")
            jid = str(entry.get("job_id") or "").strip()
            tid = str(entry.get("track_id") or "").strip()
            if not jid or not tid:
                raise ValueError("Each linked track requires nonempty job_id and track_id.")
            item = {"job_id": jid, "track_id": tid}
            if entry.get("note"):
                item["note"] = str(entry["note"]).strip()[:200]
            links.append(item)
        links_json = json.dumps(links, allow_nan=False)
        now = _now()
        entity_id = payload.get("id")
        with self.lock:
            if entity_id is not None:
                if type(entity_id) is not int or entity_id < 1:
                    raise ValueError("Entity id must be a positive integer.")
                cursor = self._conn.execute(
                    """
                    UPDATE entities
                    SET label = ?, kind = ?, description = ?, linked_tracks_json = ?, updated_at = ?
                    WHERE id = ? AND case_id = ?
                    """,
                    (label, kind, description, links_json, now, entity_id, case_id),
                )
                if cursor.rowcount == 0:
                    raise KeyError(entity_id)
                self._conn.commit()
                saved_id = entity_id
            else:
                cursor = self._conn.execute(
                    """
                    INSERT INTO entities (
                        case_id, label, kind, description, linked_tracks_json, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (case_id, label, kind, description, links_json, now),
                )
                self._conn.commit()
                saved_id = cursor.lastrowid
        return {
            "id": saved_id,
            "case_id": case_id,
            "label": label,
            "kind": kind,
            "description": description,
            "linked_tracks": links,
            "updated_at": now,
        }

    def delete_entity(self, entity_id: int) -> bool:
        with self.lock:
            cursor = self._conn.execute("DELETE FROM entities WHERE id = ?", (entity_id,))
            self._conn.commit()
            return cursor.rowcount > 0

    @staticmethod
    def _compute_audit_hash(
        prev_hash: str,
        timestamp: str,
        actor: str,
        action: str,
        target_type: str,
        target_id: str,
        details_json: str,
    ) -> str:
        raw = f"{prev_hash}|{timestamp}|{actor}|{action}|{target_type}|{target_id}|{details_json}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def append_audit(
        self,
        action: str,
        target_type: str,
        target_id: str,
        details: dict | None = None,
        actor: str = "local-investigator",
    ) -> dict:
        ts = _now()
        actor_clean = str(actor or "local-investigator").strip()[:100]
        action_clean = str(action).strip()[:80]
        target_type_clean = str(target_type).strip()[:40]
        target_id_clean = str(target_id).strip()[:120]
        details_json = json.dumps(details or {}, sort_keys=True, allow_nan=False)
        with self.lock:
            last = self._conn.execute(
                "SELECT entry_hash FROM audit_log ORDER BY id DESC LIMIT 1"
            ).fetchone()
            prev_hash = last["entry_hash"] if last else GENESIS_HASH
            entry_hash = self._compute_audit_hash(
                prev_hash,
                ts,
                actor_clean,
                action_clean,
                target_type_clean,
                target_id_clean,
                details_json,
            )
            cursor = self._conn.execute(
                """
                INSERT INTO audit_log (
                    timestamp, actor, action, target_type, target_id,
                    details_json, prev_hash, entry_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts,
                    actor_clean,
                    action_clean,
                    target_type_clean,
                    target_id_clean,
                    details_json,
                    prev_hash,
                    entry_hash,
                ),
            )
            self._conn.commit()
            row_id = cursor.lastrowid
        return {
            "id": row_id,
            "timestamp": ts,
            "actor": actor_clean,
            "action": action_clean,
            "target_type": target_type_clean,
            "target_id": target_id_clean,
            "details": json.loads(details_json),
            "prev_hash": prev_hash,
            "entry_hash": entry_hash,
        }

    def list_audit(self, limit: int = 100) -> list[dict]:
        bounded = max(1, min(500, int(limit)))
        with self.lock:
            rows = self._conn.execute(
                "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?",
                (bounded,),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "timestamp": row["timestamp"],
                "actor": row["actor"],
                "action": row["action"],
                "target_type": row["target_type"],
                "target_id": row["target_id"],
                "details": json.loads(row["details_json"]),
                "prev_hash": row["prev_hash"],
                "entry_hash": row["entry_hash"],
            }
            for row in rows
        ]

    def verify_audit_chain(self) -> tuple[bool, int]:
        with self.lock:
            rows = self._conn.execute(
                "SELECT * FROM audit_log ORDER BY id ASC"
            ).fetchall()
        expected_prev = GENESIS_HASH
        for row in rows:
            if row["prev_hash"] != expected_prev:
                return False, len(rows)
            expected = self._compute_audit_hash(
                row["prev_hash"],
                row["timestamp"],
                row["actor"],
                row["action"],
                row["target_type"],
                row["target_id"],
                row["details_json"],
            )
            if row["entry_hash"] != expected:
                return False, len(rows)
            expected_prev = row["entry_hash"]
        return True, len(rows)

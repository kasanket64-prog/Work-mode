"""Cross-clip forensic evidence search, DVR clock calibration, and printable case reports."""

import html
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .review import read_records

SAFE_PREVIEW_RE = re.compile(r"^(frame-\d+|image-0)\.jpg$")


def compute_wall_clock(
    recorded_start_iso: str | None,
    timestamp_seconds: float,
    clock_offset_seconds: float = 0.0,
) -> str | None:
    """Resolve real-world ISO-8601 timestamp from clip start, offset, and frame time."""
    if not recorded_start_iso:
        return None
    raw = str(recorded_start_iso).strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        start = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    try:
        actual = start + timedelta(seconds=float(timestamp_seconds) + float(clock_offset_seconds))
    except (OverflowError, TypeError, ValueError):
        return None
    return actual.isoformat(timespec="seconds")


def search_evidence(
    jobs_root: Path,
    jobs_list: list[dict],
    *,
    query: str = "",
    case_id: str | None = None,
    object_label: str | None = None,
    color: str | None = None,
    plate: str | None = None,
    event_type: str | None = None,
    limit: int = 100,
) -> list[dict]:
    """Search completed analyses across a case or workspace for leads."""
    q = (query or "").strip().lower()
    lbl_filter = (object_label or "").strip().lower()
    color_filter = (color or "").strip().lower()
    plate_filter = (plate or "").strip().upper()
    ev_filter = (event_type or "").strip().lower()
    cap = max(1, min(500, int(limit)))

    results: list[dict] = []
    for job in jobs_list:
        if job.get("status") != "completed":
            continue
        if case_id and job.get("case_id") != case_id:
            continue
        job_id = str(job["id"])
        result_path = jobs_root / job_id / "result.jsonl"
        if not result_path.is_file():
            continue
        try:
            rows = read_records(result_path)
        except (OSError, ValueError):
            continue

        camera = str(job.get("camera_label") or job.get("original_filename") or job_id[:8])
        offset = float(job.get("clock_offset_seconds") or 0.0)
        start_iso = job.get("recorded_start_iso")
        sha256 = job.get("source_sha256")

        for row in rows:
            if row.get("type") != "frame":
                continue
            frame_index = int(row.get("frame_index") or 0)
            ts = float(row.get("timestamp_seconds") or 0.0)
            wall = compute_wall_clock(start_iso, ts, offset)
            p_path = row.get("preview_path")
            preview_url = (
                f"/jobs/{job_id}/frames/{p_path}"
                if isinstance(p_path, str)
                and SAFE_PREVIEW_RE.fullmatch(p_path)
                and (jobs_root / job_id / "frames" / p_path).is_file()
                else None
            )

            events = row.get("events") or []
            regions = row.get("regions") or []
            people = row.get("person_analysis") or []
            plates = row.get("plates") or []
            relations = row.get("confirmed_relations") or row.get("relations") or []

            # Frame-level structured gating so combined filters narrow down to the same moment.
            if lbl_filter and not any(
                lbl_filter in str(r.get("label") or "").lower() for r in regions
            ):
                continue
            if color_filter and not any(
                color_filter in str((p.get("appearance") or {}).get("value") or "").lower()
                for p in people
            ):
                continue
            if plate_filter and not any(
                plate_filter in str(pl.get("text") or "").upper() for pl in plates
            ):
                continue
            if ev_filter and not any(
                ev_filter in str(ev.get("type") or "").lower() for ev in events
            ):
                continue

            structured_only = bool(lbl_filter or color_filter or plate_filter or ev_filter)
            candidates: list[tuple[str, str]] = []
            if ev_filter or not structured_only:
                for ev in events:
                    ev_type = str(ev.get("type") or "")
                    summary = f"Alert: {ev_type} ({ev.get('object_track_id') or ''})"
                    if ev.get("zone_id"):
                        summary += f" @ {ev['zone_id']}"
                    if not ev_filter or ev_filter in ev_type.lower():
                        candidates.append(("event", summary))
            if plate_filter or not structured_only:
                for pl in plates:
                    text = str(pl.get("text") or "").strip()
                    if text and (not plate_filter or plate_filter in text.upper()):
                        candidates.append(("plate", f"Plate OCR: {text}"))
            if color_filter or not structured_only:
                for p in people:
                    app = str((p.get("appearance") or {}).get("value") or "").strip()
                    if app and (not color_filter or color_filter in app.lower()):
                        candidates.append(("appearance", f"Person {p.get('person_id')}: {app}"))
            if q and not structured_only:
                for rel in relations:
                    txt = f"{rel.get('subject_id')} {rel.get('predicate')} {rel.get('object_id')}"
                    if q in txt.lower():
                        candidates.append(("relation", f"Relation: {txt}"))
            if lbl_filter or (q and not structured_only and not candidates):
                for r in regions:
                    lbl = str(r.get("label") or "").strip()
                    tid = str(r.get("track_id") or r.get("id") or "")
                    if lbl and (not lbl_filter or lbl_filter in lbl.lower()):
                        candidates.append(("object", f"Detected {lbl} ({tid})"))

            if not candidates and not structured_only:
                if q and q in camera.lower():
                    candidates.append(("clip", f"Camera clip: {camera}"))

            seen_summaries = set()
            for kind, summary in candidates:
                haystack = f"{camera} {summary}".lower()
                if q and q not in haystack:
                    continue
                if summary in seen_summaries:
                    continue
                seen_summaries.add(summary)
                results.append(
                    {
                        "job_id": job_id,
                        "case_id": job.get("case_id"),
                        "camera_label": camera,
                        "source_sha256": sha256,
                        "frame_index": frame_index,
                        "timestamp_seconds": ts,
                        "wall_clock": wall,
                        "match_kind": kind,
                        "summary": summary,
                        "preview_url": preview_url,
                        "review_url": f"/jobs/{job_id}/review#frame={frame_index}",
                    }
                )
                if len(results) >= cap:
                    return results

    results.sort(key=lambda item: (item["wall_clock"] or "", item["timestamp_seconds"]))
    return results


def render_case_report_html(
    case: dict,
    jobs: list[dict],
    verdicts_by_job: dict[str, dict],
    bookmarks: list[dict],
    entities: list[dict],
    audit_entries: list[dict],
    chain_valid: bool,
    chain_count: int,
) -> str:
    """Render a self-contained, CSP-safe printable Case Evidence Dossier."""
    e = html.escape
    exported = datetime.now(timezone.utc).isoformat(timespec="seconds")
    case_num = e(str(case["case_number"]))
    title = e(str(case["title"]))
    investigator = e(str(case.get("investigator") or "Unassigned"))
    status = e(str(case.get("status") or "open").replace("_", " ").upper())
    notes = e(str(case.get("notes") or "No investigator case notes recorded."))
    chain_badge = (
        f"CHAIN VERIFIED ({chain_count} entries)"
        if chain_valid
        else f"INTEGRITY WARNING ({chain_count} entries)"
    )
    chain_cls = "badge ok" if chain_valid else "badge warn"

    media_rows = []
    verified_rows = []
    job_cameras = {}
    for job in jobs:
        jid = str(job["id"])
        cam = str(job.get("camera_label") or job.get("original_filename") or jid[:8])
        job_cameras[jid] = cam
        sha = e(str(job.get("source_sha256") or "Not recorded"))
        orig = e(str(job.get("original_filename") or f"{jid[:8]} ({job.get('mode')})"))
        start = e(str(job.get("recorded_start_iso") or "Not calibrated"))
        offset = float(job.get("clock_offset_seconds") or 0.0)
        size_mb = f"{(job.get('bytes', 0) / 1048576):.2f} MB"
        frames_n = int(job.get("frames_processed") or 0)
        media_rows.append(
            "<tr>"
            f"<td><strong>{e(cam)}</strong><br><small>{orig}</small></td>"
            f"<td class='mono'>{sha}</td>"
            f"<td>{start}<br><small>Offset: {offset:+.1f}s</small></td>"
            f"<td>{e(str(job.get('status', '')))} · {frames_n} frames · {size_mb}</td>"
            "</tr>"
        )
        for event_id, v in verdicts_by_job.get(jid, {}).items():
            if not v.get("verdict") and not v.get("note"):
                continue
            verified_rows.append(
                {
                    "when": v.get("updated_at") or "",
                    "camera": cam,
                    "source": "Event Verdict",
                    "tag": str(v.get("verdict") or "reviewed").upper(),
                    "detail": f"{event_id}: {v.get('note') or '(verified without note)'}",
                }
            )

    for bm in bookmarks:
        cam = job_cameras.get(str(bm["job_id"]), str(bm["job_id"])[:8])
        when = bm.get("wall_clock") or f"t={bm['timestamp_seconds']:.2f}s (frame {bm['frame_index']})"
        verified_rows.append(
            {
                "when": str(when),
                "camera": cam,
                "source": f"Bookmark · {str(bm['tag']).replace('_', ' ').upper()}",
                "tag": f"frame #{bm['frame_index']}",
                "detail": str(bm["caption"]),
            }
        )
    verified_rows.sort(key=lambda r: r["when"])

    timeline_html = "\n".join(
        "<tr>"
        f"<td class='mono'>{e(r['when'])}</td>"
        f"<td>{e(r['camera'])}</td>"
        f"<td><span class='pill'>{e(r['source'])}</span></td>"
        f"<td>{e(r['tag'])}</td>"
        f"<td>{e(r['detail'])}</td>"
        "</tr>"
        for r in verified_rows
    ) or "<tr><td colspan='5'>No pinned evidence bookmarks or saved verdicts in this case yet.</td></tr>"

    entity_html = "\n".join(
        "<div class='entity'>"
        f"<strong>{e(ent['label'])}</strong> <span class='pill'>{e(ent['kind'])}</span>"
        f"<p>{e(ent.get('description') or 'No description')}</p>"
        f"<p class='mono small'>Linked tracks: {e(', '.join(f'{t['job_id'][:8]}:{t['track_id']}' for t in ent.get('linked_tracks', [])) or 'None')}</p>"
        "</div>"
        for ent in entities
    ) or "<p>No human-linked subject or vehicle dossiers recorded yet.</p>"

    audit_html = "\n".join(
        "<tr>"
        f"<td class='mono'>{e(item['timestamp'])}</td>"
        f"<td>{e(item['actor'])}</td>"
        f"<td><code>{e(item['action'])}</code></td>"
        f"<td class='mono'>{e(item['target_type'])}:{e(item['target_id'])}</td>"
        f"<td class='mono'>{e(item['entry_hash'][:16])}…</td>"
        "</tr>"
        for item in audit_entries[:50]
    ) or "<tr><td colspan='5'>No audit entries recorded.</td></tr>"

    media_html = "\n".join(media_rows) or (
        "<tr><td colspan='4'>No media jobs assigned to this case yet.</td></tr>"
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Evidence Report · {case_num} · {title}</title>
<style>
  :root {{ --ink:#14211b; --muted:#54655d; --line:#d2dbd5; --bg:#ffffff; --panel:#f5f8f5; --accent:#1c5239; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; padding:32px; background:var(--bg); color:var(--ink); font:14px/1.55 "Segoe UI",Arial,sans-serif; max-width:1100px; margin-inline:auto; }}
  header {{ display:flex; justify-content:space-between; align-items:flex-start; gap:20px; border-bottom:2px solid var(--ink); padding-bottom:18px; margin-bottom:22px; }}
  h1 {{ margin:4px 0; font-size:26px; letter-spacing:-0.6px; }}
  h2 {{ font-size:16px; margin:28px 0 10px; border-bottom:1px solid var(--line); padding-bottom:6px; text-transform:uppercase; letter-spacing:0.8px; color:var(--accent); }}
  .meta {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; background:var(--panel); border:1px solid var(--line); padding:14px; border-radius:8px; margin-bottom:18px; }}
  .meta small {{ display:block; font-size:10px; text-transform:uppercase; color:var(--muted); letter-spacing:1px; }}
  .badge {{ display:inline-block; padding:4px 10px; border-radius:999px; font-size:11px; font-weight:700; }}
  .badge.ok {{ background:#e0f2e7; color:#135334; }}
  .badge.warn {{ background:#fde8e4; color:#9a2b1f; }}
  .pill {{ display:inline-block; padding:2px 8px; border-radius:4px; background:var(--panel); border:1px solid var(--line); font-size:11px; font-weight:600; }}
  .notice {{ background:#fff9ea; border:1px solid #e5d2a4; padding:12px 14px; border-radius:6px; font-size:12.5px; margin-bottom:20px; }}
  table {{ width:100%; border-collapse:collapse; font-size:12.5px; margin-top:8px; }}
  th, td {{ text-align:left; vertical-align:top; padding:9px 10px; border-bottom:1px solid var(--line); }}
  th {{ background:var(--panel); font-size:11px; text-transform:uppercase; letter-spacing:0.6px; color:var(--muted); }}
  .mono {{ font-family:ui-monospace,Consolas,monospace; font-size:11.5px; word-break:break-all; }}
  .small {{ font-size:11.5px; color:var(--muted); }}
  .entity {{ border:1px solid var(--line); border-radius:6px; padding:10px 14px; margin-bottom:8px; background:var(--panel); }}
  .entity p {{ margin:4px 0; font-size:12.5px; }}
  .actions {{ display:flex; gap:10px; }}
  button {{ border:1px solid var(--accent); background:var(--accent); color:#fff; padding:9px 14px; border-radius:6px; font-weight:600; cursor:pointer; }}
  a.back {{ border:1px solid var(--line); padding:8px 14px; border-radius:6px; text-decoration:none; color:var(--ink); font-weight:600; }}
  @media print {{ body {{ padding:0; }} .actions {{ display:none; }} }}
</style>
</head>
<body>
  <header>
    <div>
      <small class="mono">FIELDNOTE FORENSIC CASE DOSSIER · {case_num}</small>
      <h1>{title}</h1>
      <span class="{chain_cls}">{e(chain_badge)}</span>
    </div>
    <div class="actions">
      <a class="back" href="/#cases">← Back to Cases</a>
      <button id="printBtn" type="button">Print / Save PDF</button>
    </div>
  </header>

  <section class="meta">
    <div><small>Case Status</small><strong>{status}</strong></div>
    <div><small>Lead Investigator</small><strong>{investigator}</strong></div>
    <div><small>Assigned Media Clips</small><strong>{len(jobs)}</strong></div>
    <div><small>Export Timestamp (UTC)</small><strong class="mono">{e(exported)}</strong></div>
  </section>

  <div class="notice">
    <strong>Forensic Verification Notice:</strong> Source media integrity is established using the SHA-256 digests recorded on ingest. Automated object detections, relation scores, and candidate alerts require independent human inspection and must not be treated as standalone proof of identity, intent, or legal guilt.
  </div>

  <p><strong>Investigator Notes:</strong> {notes}</p>

  <h2>1. Media Chain-of-Custody Inventory</h2>
  <table>
    <thead><tr><th>Camera / Source File</th><th>Ingest SHA-256 Digest</th><th>Clock Calibration</th><th>Processing Coverage</th></tr></thead>
    <tbody>{media_html}</tbody>
  </table>

  <h2>2. Human-Verified Timeline (Pinned Bookmarks &amp; Event Verdicts)</h2>
  <table>
    <thead><tr><th>Time</th><th>Camera</th><th>Source</th><th>Tag / Verdict</th><th>Investigator Summary</th></tr></thead>
    <tbody>{timeline_html}</tbody>
  </table>

  <h2>3. Human-Linked Entity Dossier</h2>
  {entity_html}

  <h2>4. Hash-Chained Audit Log (Most Recent Entries)</h2>
  <table>
    <thead><tr><th>Timestamp</th><th>Actor</th><th>Action</th><th>Target</th><th>Entry Hash</th></tr></thead>
    <tbody>{audit_html}</tbody>
  </table>

  <script>
    'use strict';
    const btn = document.getElementById('printBtn');
    if (btn) btn.addEventListener('click', () => window.print());
  </script>
</body>
</html>
"""

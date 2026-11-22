"""Calibration benchmark, error attribution, and replay threshold sweep CLI."""

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import replace
from math import isfinite
from pathlib import Path

from .reasoning import ReasoningSettings, load_reasoning
from .replay import replay
from .review import evaluate, metrics, read_records

DEFAULT_THRESHOLDS = [0.4, 0.5, 0.6, 0.7, 0.8]


def _match_events(
    predicted: list[dict], expected: list[dict], mapping: dict[str, str], tolerance: float
) -> tuple[dict[int, int], dict[int, int]]:
    def identity(event: dict, is_pred: bool) -> tuple:
        obj = event["object_track_id"]
        subj = event.get("subject_track_id")
        return (
            event["type"],
            mapping.get(obj, obj) if is_pred else obj,
            event.get("zone_id"),
            mapping.get(subj, subj) if is_pred else subj,
        )

    edges = [
        [
            exp_idx
            for exp_idx, target in enumerate(expected)
            if identity(pred, True) == identity(target, False)
            and abs(pred["timestamp_seconds"] - target["timestamp_seconds"]) <= tolerance
        ]
        for pred in predicted
    ]
    exp_to_pred: dict[int, int] = {}

    def dfs(pred_idx: int, visited: set[int]) -> bool:
        for exp_idx in edges[pred_idx]:
            if exp_idx not in visited:
                visited.add(exp_idx)
                if exp_idx not in exp_to_pred or dfs(exp_to_pred[exp_idx], visited):
                    exp_to_pred[exp_idx] = pred_idx
                    return True
        return False

    for pred_idx in range(len(predicted)):
        dfs(pred_idx, set())
    pred_to_exp = {pred_idx: exp_idx for exp_idx, pred_idx in exp_to_pred.items()}
    return pred_to_exp, exp_to_pred


def _attribute_errors(
    rows: list[dict],
    predicted: list[dict],
    expected: list[dict],
    mapping: dict[str, str],
    tolerance: float,
    pred_to_exp: dict[int, int],
    exp_to_pred: dict[int, int],
    relation_threshold: float,
) -> dict:
    frames = [row for row in rows if row.get("type") == "frame"]
    reverse_map: dict[str, set[str]] = defaultdict(set)
    for raw_id, mapped_id in mapping.items():
        reverse_map[mapped_id].add(raw_id)

    def identity(event: dict, is_pred: bool) -> tuple:
        obj = event["object_track_id"]
        subj = event.get("subject_track_id")
        return (
            event["type"],
            mapping.get(obj, obj) if is_pred else obj,
            event.get("zone_id"),
            mapping.get(subj, subj) if is_pred else subj,
        )

    false_negatives = []
    for exp_idx, target in enumerate(expected):
        if exp_idx in exp_to_pred:
            continue
        target_id = identity(target, False)
        t_exp = float(target["timestamp_seconds"])
        same_id_deltas = [
            abs(float(pred["timestamp_seconds"]) - t_exp)
            for pred in predicted
            if identity(pred, True) == target_id
        ]
        candidate_raw_ids = reverse_map.get(target["object_track_id"], set()) | {
            target["object_track_id"]
        }
        window_frames = [
            frame
            for frame in frames
            if abs(float(frame["timestamp_seconds"]) - t_exp) <= max(tolerance, 2.0)
        ]
        tracked_near = any(
            (r.get("track_id") in candidate_raw_ids or r.get("id") in candidate_raw_ids)
            for frame in window_frames
            for r in frame.get("regions", [])
        )
        max_rel = max(
            (
                float(rel["score"])
                for frame in window_frames
                for rel in frame.get("relations", [])
                if rel.get("object_id") in candidate_raw_ids
                or rel.get("subject_id") in candidate_raw_ids
            ),
            default=None,
        )
        if same_id_deltas:
            reason = "predicted_outside_tolerance"
        elif not tracked_near:
            reason = "track_missing_near_timestamp"
        elif max_rel is not None and max_rel < relation_threshold:
            reason = "relation_below_threshold"
        else:
            reason = "reasoning_filtered_or_insufficient_hits"
        false_negatives.append(
            {
                "event": target,
                "reason": reason,
                "nearest_same_identity_seconds": min(same_id_deltas) if same_id_deltas else None,
                "track_visible_near_timestamp": tracked_near,
                "max_relation_score_near_timestamp": max_rel,
            }
        )

    false_positives = []
    for pred_idx, pred in enumerate(predicted):
        if pred_idx in pred_to_exp:
            continue
        pred_id = identity(pred, True)
        t_pred = float(pred["timestamp_seconds"])
        same_id_deltas = [
            abs(float(target["timestamp_seconds"]) - t_pred)
            for target in expected
            if identity(target, False) == pred_id
        ]
        false_positives.append(
            {
                "event": pred,
                "reason": "timing_offset" if same_id_deltas else "unexpected_event",
                "nearest_same_identity_seconds": min(same_id_deltas) if same_id_deltas else None,
            }
        )

    return {
        "false_negatives": false_negatives,
        "false_positives": false_positives,
    }


def _per_type_breakdown(
    predicted: list[dict],
    expected: list[dict],
    pred_to_exp: dict[int, int],
    exp_to_pred: dict[int, int],
) -> dict[str, dict]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for pred_idx, pred in enumerate(predicted):
        ev_type = pred["type"]
        if pred_idx in pred_to_exp:
            counts[ev_type]["tp"] += 1
        else:
            counts[ev_type]["fp"] += 1
    for exp_idx, target in enumerate(expected):
        if exp_idx not in exp_to_pred:
            counts[target["type"]]["fn"] += 1
    return {
        ev_type: metrics(c["tp"], c["fp"], c["fn"]) for ev_type, c in sorted(counts.items())
    }


def run_benchmark(
    rows: list[dict],
    truth: dict,
    settings: ReasoningSettings | None = None,
    tolerance: float = 2.0,
    thresholds: list[float] | None = None,
) -> dict:
    """Evaluate predictions vs. reviewed truth, attribute errors, and sweep thresholds."""
    if not rows:
        raise ValueError("session contains no records")
    base_settings = settings or ReasoningSettings()
    base_eval = evaluate(rows, truth, tolerance=tolerance)
    frames = [r for r in rows if r.get("type") == "frame"]
    predicted = [e for frame in frames for e in frame.get("events", [])]
    expected = truth["events"]
    mapping = truth.get("track_mapping", {})
    pred_to_exp, exp_to_pred = _match_events(predicted, expected, mapping, tolerance)

    threshold_values = list(thresholds) if thresholds is not None else list(DEFAULT_THRESHOLDS)
    for value in threshold_values:
        if type(value) not in (int, float) or not isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"relation threshold must be in (0, 1], got {value!r}")

    sweep: list[dict] = []
    sweep_skipped_reason: str | None = None
    if rows[0].get("relation_endpoint_kind") == "track_id":
        for value in threshold_values:
            candidate_settings = replace(base_settings, relation_threshold=float(value))
            replayed = list(replay(rows, candidate_settings))
            replayed_eval = evaluate(replayed, truth, tolerance=tolerance)
            sweep.append(
                {
                    "relation_threshold": float(value),
                    "events": replayed_eval["events"],
                    "false_alarms_per_hour": replayed_eval["false_alarms_per_hour"],
                }
            )
    else:
        sweep_skipped_reason = "requires_track_id_relation_endpoints"

    best = None
    if sweep:
        best = max(
            sweep,
            key=lambda item: (
                item["events"]["f1"] if item["events"]["f1"] is not None else -1.0,
                -item["false_alarms_per_hour"],
            ),
        )

    return {
        "summary": base_eval,
        "per_event_type": _per_type_breakdown(predicted, expected, pred_to_exp, exp_to_pred),
        "error_attribution": _attribute_errors(
            rows,
            predicted,
            expected,
            mapping,
            tolerance,
            pred_to_exp,
            exp_to_pred,
            base_settings.relation_threshold,
        ),
        "threshold_sweep": sweep,
        "sweep_skipped_reason": sweep_skipped_reason,
        "best_in_sample_threshold": best["relation_threshold"] if best else None,
        "notes": [
            "Metrics reflect exact matches within tolerance on this annotated session.",
            "Replay sweeps vary relation_threshold only; neural detector and VLM predictions are unchanged.",
        ],
    }


def _fmt_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def format_markdown_report(report: dict) -> str:
    """Format a calibration benchmark report as GitHub-flavored Markdown."""
    summary = report["summary"]
    ev = summary["events"]
    lines = [
        "# Calibration Benchmark Report",
        "",
        "## Overall Event Metrics",
        "",
        f"- Duration: `{summary['duration_seconds']:.2f}s` (tolerance `±{summary['event_tolerance_seconds']}s`)",
        f"- Counts: TP=`{ev['true_positive']}`, FP=`{ev['false_positive']}`, FN=`{ev['false_negative']}`",
        f"- Precision: `{_fmt_metric(ev['precision'])}` | Recall: `{_fmt_metric(ev['recall'])}` | F1: `{_fmt_metric(ev['f1'])}`",
        f"- False alarms/hr: `{summary['false_alarms_per_hour']:.2f}`",
        "",
        "## Per-Event-Type Breakdown",
        "",
        "| Event Type | TP | FP | FN | Precision | Recall | F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for ev_type, m in report["per_event_type"].items():
        lines.append(
            f"| `{ev_type}` | {m['true_positive']} | {m['false_positive']} | {m['false_negative']} "
            f"| {_fmt_metric(m['precision'])} | {_fmt_metric(m['recall'])} | {_fmt_metric(m['f1'])} |"
        )
    if not report["per_event_type"]:
        lines.append("| _(none)_ | 0 | 0 | 0 | n/a | n/a | n/a |")

    fn_list = report["error_attribution"]["false_negatives"]
    fp_list = report["error_attribution"]["false_positives"]
    lines.extend(["", "## Error Attribution", ""])
    lines.append(f"- Missed events (FN): {len(fn_list)}")
    for item in fn_list:
        e = item["event"]
        lines.append(
            f"  - `{e['type']}` on `{e['object_track_id']}` at `{e['timestamp_seconds']}s`: `{item['reason']}`"
        )
    lines.append(f"- False alarms (FP): {len(fp_list)}")
    for item in fp_list:
        e = item["event"]
        lines.append(
            f"  - `{e['type']}` on `{e['object_track_id']}` at `{e['timestamp_seconds']}s`: `{item['reason']}`"
        )

    lines.extend(["", "## Relation Threshold Replay Sweep", ""])
    if report["threshold_sweep"]:
        lines.extend(
            [
                "| relation_threshold | TP | FP | FN | Precision | Recall | F1 | FA/hr |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in report["threshold_sweep"]:
            m = row["events"]
            lines.append(
                f"| {row['relation_threshold']:.2f} | {m['true_positive']} | {m['false_positive']} "
                f"| {m['false_negative']} | {_fmt_metric(m['precision'])} | {_fmt_metric(m['recall'])} "
                f"| {_fmt_metric(m['f1'])} | {row['false_alarms_per_hour']:.2f} |"
            )
    else:
        lines.append(f"Skipped: `{report['sweep_skipped_reason']}`")

    lines.extend(["", "## Notes", ""])
    for note in report["notes"]:
        lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def _parse_thresholds(raw: str | None) -> list[float] | None:
    if raw is None:
        return None
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("--thresholds must contain at least one numeric value")
    return values


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path, help="Session JSONL file")
    parser.add_argument("--truth", type=Path, required=True, help="Reviewed annotation JSON file")
    parser.add_argument("--output-dir", type=Path, required=True, help="New output directory")
    parser.add_argument("--reasoning-config", type=Path, help="Optional reasoning TOML file")
    parser.add_argument("--tolerance", type=float, default=2.0)
    parser.add_argument(
        "--thresholds",
        help="Comma-separated relation thresholds for replay sweep (default: 0.4,0.5,0.6,0.7,0.8)",
    )
    args = parser.parse_args(argv)
    try:
        rows = read_records(args.session)
        truth = json.loads(args.truth.read_text(encoding="utf-8-sig"))
        if not isinstance(truth, dict):
            raise ValueError("truth JSON must be an object")
        settings = load_reasoning(args.reasoning_config) if args.reasoning_config else None
        thresholds = _parse_thresholds(args.thresholds)
        report = run_benchmark(
            rows, truth, settings=settings, tolerance=args.tolerance, thresholds=thresholds
        )
        args.output_dir.mkdir(parents=True, exist_ok=False)
        report_json = args.output_dir / "report.json"
        report_md = args.output_dir / "report.md"
        report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        report_md.write_text(format_markdown_report(report), encoding="utf-8")
        print(report_json)
        return 0
    except (OSError, ValueError, RuntimeError, TypeError, json.JSONDecodeError) as exc:
        print(f"camera-benchmark: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

# Architecture and staged roadmap

Phase 1 implements `image + validated regions -> RelationPredictor -> scored relations -> JSON`.
`Region` has a per-frame ID, original-pixel box, optional display label, and optional persistent `track_id`. `Relation` references region IDs rather than class names. The current JSON result has a schema version, frame ID, and nullable timestamp to support an explicit video contract later. The still-image runner now supports both manual regions and automatic YOLO detection (Phase 2). Phase 3 adds synchronous file/camera capture, timestamp provenance, per-frame JSONL output and model reuse. Camera backend latency and hardware drops are not measured.

The adapter isolates `relsgg` from the rest of the system. It reuses one loaded model, supplies a configured vocabulary, and converts upstream triplets into independent records. It rejects excessive region counts rather than relying on upstream truncation, which would change indices. The command validates inputs before loading/downloading weights. Timing is a single CPU/GPU call measurement, not a throughput benchmark.

| Phase | Addition | Acceptance criteria |
| --- | --- | --- |
| 2 | Detector adapter produces Region records | Correct original-image coordinate mapping; empty detections; bounded region selection; measured detector recall. Define score-fusion semantics explicitly. |
| 3 | File/video/camera frame source | Monotonic frame timestamps, explicit RGB conversion from OpenCV BGR, clean shutdown, bounded queues and dropped-frame reporting. |
| 4 | Tracker produces persistent track IDs | ID-switch and occlusion tests; relation endpoints use tracks across frames; distinguish frame IDs from track IDs. |
| 5 | Temporal relation aggregation | Timestamp-based windows, hysteresis, stale-track expiry; replay tests including dropped frames; tune on held-out validation videos. |
| 6 | Event reasoning | State machines for candidate events, duration/distance evidence, duplicate-alert suppression; no event claim from one image alone. |
| 7 | Evaluation | Annotated held-out clips; relation/event precision, recall, F1; false alarms/hour, missed events, latency percentiles, end-to-end FPS. Separate detection, tracking and relation errors. |
| 8 | Dashboard/API | Inspect evidence and scores, configuration/version provenance, upstream attribution and applicable source notices. |

The Region and Relation records and RelationPredictor protocol remain the extension boundary. Detection confidence is stored separately and is not fused with relation scores. The implemented baseline for tracking, temporal evidence, candidate events, local review/API and evaluation is described in [the current guide](prototype-guide.md).

Thresholds must be fitted per checkpoint, vocabulary and deployment domain. Compare image-only and temporal baselines on identical clips; prevent train/validation/test leakage between adjacent frames. Track false alarms by elapsed footage duration, not only frame count. No improvement percentage is claimed by Phase 1.

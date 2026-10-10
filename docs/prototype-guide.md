# Working prototype: tracking, temporal evidence, events and review

This local research prototype connects real detection/relation inference, video input, baseline tracking, temporal confirmation, a candidate unattended-bag rule, a session viewer/API, and annotation-based evaluation. It is not yet validated on representative CCTV footage.

## Run on your video

Run from the Work-mode repository directory:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,inference,detection,video]"
$env:HF_HOME = "$PWD/.hf_cache"
New-Item -ItemType Directory -Force .yolo | Out-Null
$env:YOLO_CONFIG_DIR = "$PWD/.yolo"
.\.venv\Scripts\camera-video.exe --video path/to/clip.mp4 --config configs/camera.toml --reasoning-config configs/reasoning.toml --max-frames 1000 --output outputs/session.jsonl
```

For a camera, replace `--video path/to/clip.mp4` with `--camera 0`. Ctrl+C preserves flushed records. Models load once. Output filenames must be new. The default limit is 100 processed frames; increase it for longer clips. CPU processing may be slower than real time. A physical webcam has not been tested here.

## Review results and local API

```powershell
.\.venv\Scripts\python.exe -m camera_system.review outputs/session.jsonl --output outputs/review.html
```

Open the HTML in a browser. The frame slider shows box layouts, IDs, raw/confirmed relations and candidate events. Source imagery is not embedded. To serve the snapshot locally:

```powershell
.\.venv\Scripts\python.exe -m camera_system.review outputs/session.jsonl --output outputs/served-review.html --serve --port 8765
```

Visit `http://127.0.0.1:8765/`. `GET /api/session` returns the loaded records. The server binds to localhost, has no write endpoints, serves no arbitrary files and stops with Ctrl+C. It is a snapshot viewer, not a live updating operations dashboard or internet deployment.

## Demo without model downloads

```powershell
.\.venv\Scripts\python.exe -m camera_system.demo --output-dir outputs/my-demo
```

The directory must be new. Open its `review.html`. This **simulates detections and scores**: a person holds a bag for three observations, then disappears while the bag stays visible. A single candidate alert occurs at timestamp 33 after 30 seconds of observed absence. `truth.json` and `metrics.json` exercise the evaluator. Their perfect scores are software fixture results, not model accuracy.

## Tracking and output contract

Video schema version **2** declares `relation_endpoint_kind: track_id`. Region `id` is local to a frame; `track_id` persists within a run. Video relations reference track IDs. Phase 1 image output remains schema version 1 with region-ID endpoints.

Tracking uses label-compatible greedy one-to-one IoU matching. Configure `--track-iou` (0.3), `--track-max-missed` (5 processed observations) and `--track-max-age` (2 seconds). Both expiry limits apply. IDs are never reused within a run. This baseline has no motion/appearance model or person re-identification; crossing objects, fast movement and occlusion can switch IDs. Tests establish simple continuity, reordering and expiry behavior, not tracking benchmark accuracy.

## Temporal confirmation

Defaults require three positive observations spanning at least 0.5 seconds within a 3-second window, each score at least 0.6, including a positive observation now. At most 256 samples are retained per edge. Duplicate edges in one frame count once. Stale evidence expires after 1.5 seconds; larger gaps reset evidence and event state. Configure `configs/reasoning.toml`.

These are uncalibrated defaults. Slow live CPU inference may produce gaps exceeding these settings and prevent confirmation. Measure the actual cadence before tuning gap tolerance and track age. More tolerance means greater risk of assuming continuity through missing evidence.

## Candidate unattended-bag rule

Recognized bag labels are `backpack`, `handbag`, and `suitcase`. The rule requires a confirmed `person -> holding/carrying -> bag` association, a continuously visible stationary bag, and no nearby detected person for 30 seconds from the first qualifying absence observation. Movement, renewed interaction or a nearby person resets the timer; a missing bag or long frame gap clears its state. One alert is emitted per continuous qualifying episode.

Distances use box centers divided by the image diagonal, not physical meters. Association is not proof of ownership. Missed people, detector jitter, track switches and bad relations can cause errors. No automatic messages or external alerts are sent.

## Evaluation

Create annotations for the same complete run/clip:

In a newly generated review page, click **Download annotation draft**. This works
in saved HTML too and uses the currently loaded tracked video. Images and invalid
video records disable the button. The draft contains navigation references only;
model events and human candidate verdicts are not copied into labels. Previously
generated HTML pages need to be regenerated to include the new button.

Alternatively, start a draft from the command line:

```powershell
.\.venv\Scripts\python.exe -m camera_system.review outputs/session.jsonl --annotation-template --output outputs/annotations-draft.json
```

The output must be a new file. It lists sampled frame timestamps and visible
track IDs as navigation references, without copying predicted events or relations.
Review the original footage independently, including events the model missed.
Fill in the first-to-last sampled evaluation interval, its duration, event labels, and any track-to-identity mapping.
Only use `events: []` after verifying no target events occurred. Set
`annotation_status` from `draft` to `reviewed` when finished; the CLI and upload
evaluator reject drafts even if their numeric fields are populated. Older annotation
files without this field remain supported.

The last sampled timestamp is not automatically the full clip duration. Check
whether processing stopped at a frame limit, and evaluate matching footage and
annotations. Reference track IDs can contain tracking errors and require review.
For optional relation labels, add both fields shown below after independently
reviewing those frames. `reference_frames` is not a list of annotated frames.

```json
{
  "duration_seconds": 120,
  "evaluation_interval_seconds": [0, 120],
  "track_mapping": {"track-3": "bag-A", "track-1": "person-A"},
  "events": [{"type": "possible_unattended_bag", "object_track_id": "bag-A", "timestamp_seconds": 60}],
  "annotated_frame_indices": [0, 10],
  "relations": [{"frame_index": 10, "subject_id": "person-A", "predicate": "holding", "object_id": "bag-A"}]
}
```

```powershell
.\.venv\Scripts\python.exe -m camera_system.review outputs/session.jsonl --truth path/to/annotations.json --tolerance 2 --output outputs/metrics.json
```

Events match one-to-one by type, mapped object ID and timestamp tolerance. Duplicate alerts count as false positives. Metrics include event precision/recall/F1, false alarms/hour and optional confirmed-relation metrics. Undefined ratios are null. `annotated_frame_indices` identifies completely annotated frames: omitted relations on those frames count as negatives. All annotated frames must exist in the run. The interval endpoints must equal the first and last sampled timestamps, and duration must equal their difference. This measures sampled coverage, not full clip duration. Partial runs require an explicit `evaluation_interval_seconds`. The example assumes samples span 0 through 120 seconds. Tune thresholds on separate data from held-out evaluation.

Evaluation requires at least two video frames with increasing sampled timestamps. For relation evaluation,
supply both `relations` and a nonempty `annotated_frame_indices` list containing
nonnegative integer indices. An empty `relations` list explicitly labels those
frames as having no target relations. Omit both fields for event-only evaluation;
an incomplete pair is rejected rather than silently ignoring relation labels.

## Tests and current limits

See [prototype-verification.md](prototype-verification.md). Earlier phase records are historical snapshots. Remaining work for deployment includes physical webcam testing, moving-object/occlusion evaluation, domain calibration and held-out event accuracy. A representative recorded camera clip and manual labels are the next practical inputs.

Lightweight tests: `.\.venv\Scripts\python.exe -m pytest -q`. For the real-model tests, set `RUN_MODEL_TESTS=1`, `RUN_DETECTOR_TESTS=1` and `RUN_VIDEO_TESTS=1`; keep `HF_HOME` pointed at the downloaded cache. `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` permit cached inference. Local YOLO weights must also be present.

Browser draft regression checks use Node.js when available (provided by CI);
it is not required to run the Python application or use the review page.

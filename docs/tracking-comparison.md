# Tracking diagnostics

The default tracker matches boxes by overlap with their last observed positions. The experimental `camera-video --track-motion` option predicts each box center from the last two matched observations and elapsed video time before matching. It preserves observed box coordinates, class gating, one-to-one matches and existing age/missed-frame expiry. It does not invent detections during gaps. At least two successful observations are needed to estimate velocity; abrupt motion, camera movement, bad detections and crossing objects remain difficult.

Tests cover motion through a missed observation, unequal timestamp spacing, expiry, and a synthetic crossing where velocity preserves two IDs. On 6 October 2026, 26 tracking/video/reasoning tests passed.

Replay the saved detections without loading models:

```powershell
.\.venv\Scripts\python.exe scripts/compare_tracking.py outputs/video-comparison-20261006/53902-476396222-detail.jsonl --output outputs/tracking-check.json
```

Use a new output path. Only detections are reused: saved IDs are discarded and fresh trackers start for each mode. Relationships and events are not recomputed or carried into changed tracks. The report verifies unique IDs per frame and unchanged observed boxes.

Results on the first 12 sampled frames of the three supplied videos:

| Scene | Previous-frame ID reuse: default | Motion prediction |
| --- | ---: | ---: |
| Office | 82.2% | 82.2% |
| Traffic | 16.4% | 14.6% |
| Boats | 65.3% | 69.5% |

These figures are diagnostics, not identity accuracy. A wrong match can still reuse an ID; unique track IDs do not count distinct real objects. Motion prediction did not solve traffic fragmentation and is therefore not enabled by default. The traffic sample needs closer examination of detection gaps and displacement at the sampled interval, followed by labeled tracks to evaluate a stronger association method. Local report: `outputs/tracking-motion-comparison.json`.

## Compare tracking frequency

```powershell
.\.venv\Scripts\python.exe scripts/benchmark_tracking_rate.py --video path/to/traffic.mp4 --output outputs/traffic-rate-new
```

This detects the first 67 source frames once and compares tracking every frame against every sixth frame. Results are scored only on shared frames (0, 6, ..., 66), with identical detections. Missed-frame expiry is scaled from 5 sparse observations to 30 dense observations; both trackers retain the same two-second age limit. Reports contain no relationship or event inference. Dense tracking requires roughly six times as many detector calls over the same source duration. HTML reviews and JSONL results contain only the shared frames.

Use `--start-frame 500` to test a later section: the comparison keeps original frame indices and timestamps and shares frames 500, 506, ..., 566. At 50 FPS this covers 10–11.32 seconds. Earlier frames are decoded sequentially to retain the source's timestamp behavior, so later offsets increase setup time. Tests check both zero and nonzero offsets against a known synthetic trajectory.

The comparison was validated on a synthetic moving box, where dense observations preserve one ID and sparse observations lose overlap. Eleven comparison/tracker tests passed. The real-video rate comparison has not run: the supplied traffic file `C:/Users/Sanket/Downloads/53902-476396222.mp4` was no longer present when checked. Saved sparse detections cannot reconstruct the missing intervening frames. No real-video improvement is claimed for this experiment.

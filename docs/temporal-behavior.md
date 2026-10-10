# Temporal behavior candidates

Enable `--reasoning-config configs/behavior.toml` with the video CLI, or replay
an existing tracked session without running the models again:

```powershell
.\.venv\Scripts\python.exe -m camera_system.replay outputs/office-40-scheduled-20261006.jsonl --reasoning-config configs/behavior.toml --output outputs/office-behavior.jsonl
.\.venv\Scripts\python.exe -m camera_system.review outputs/office-behavior.jsonl --output outputs/office-behavior.html
```

Use new output filenames. Person and vehicle dwell candidates require a continuous
track whose center stays within a configured radius of its initial image position.
The example requires three seconds and at least three observations, with radius
1% of the image diagonal. `dwell_seconds = 0` disables this optional feature.

One event is emitted per stationary interval, with track ID, start/end time,
duration, support count, and the current evidence box. Movement beyond the radius,
a missing track, class changes, resolution changes, or excessive time gaps reset
the interval. Existing zone and bag reasoning can be combined in the same config.

This is intended for fixed cameras. Camera movement, perspective, detector jitter,
and ID switches can produce errors. Dwell does not prove standing, parking,
loitering, or intent. Walking, running and fall recognition still need separate
validated temporal pose/action analysis. Scores are not invented for this rule;
duration and support counts are the evidence. Evaluate on labeled video before
using these candidates as operational alerts.

## Person movement window

With `--analyze-people`, movement uses a rolling one-second history per visible
person track (maximum 128 observations), requiring at least 0.25 seconds of
evidence. This allows consecutive 30 FPS frames to build sufficient evidence.
The output includes observation count, elapsed span and image-diagonals per
second. Missing people or changed image dimensions clear the history. Sampling
gaps above the window return unknown until enough new evidence accumulates.
The existing review panel displays the movement description; exported frame data
includes its supporting measurements. This remains image-plane movement, not
physical speed or a walking/running classifier.

## Person–vehicle distance trends

The behavior config enables closing/separating candidates at a net change of
3% of image diagonal. Each pair needs at least three samples over the configured
minimum duration, monotonic distance changes within the observation window,
and proximity at one endpoint (`person_near_fraction`). Events include both track
IDs, start/end distances, duration and observation count. Missing tracks, class
changes, long gaps or resolution changes discard pair evidence. Repeated events
in the same direction are suppressed until the direction reverses or the pair
is reset. Zero `pair_distance_change_fraction` disables the feature.

Either object may be moving. These are relative image-distance observations,
not evidence that a person intentionally approached or followed a vehicle.
Moving-camera footage and identity switches require particular care. Real-world
precision and recall have not been established for this rule.

## Review event evidence

Generate a review from the saved session to attach evidence-frame selectors to
events. Each selector lists frames inside the event interval where its required
tracks are visible. Selecting a timestamp moves the review timeline to that frame.
The saved input session is not changed.

```powershell
.\.venv\Scripts\python.exe -m camera_system.review outputs/traffic-behavior-pairs-20261008.jsonl --output outputs/traffic-evidence.html --serve --port 8771
```

Open http://127.0.0.1:8771/ and use **Review evidence frame** on an event card.
Older events without a start time can link only their trigger frame. Raw files
loaded through the browser's Open Results control need existing
`review_frame_indices` metadata to show these selectors; generate the review
with the command above when starting from an original model session.

Each event also has a **Your verdict** selector: Correct, Incorrect, Uncertain,
or Unreviewed. These are human judgments, kept separate from model observations.
Use **Export JSON** before closing or opening another session; edits are held in
memory until exported. Reopen the exported file with **Open Results** to resume.
The summary reports the correct percentage among decided candidates only.
Uncertain and unreviewed candidates are excluded. This is not full-video accuracy:
missed events and recall still require independent ground-truth annotations and
the existing `--truth` evaluation. Verdicts do not change that evaluation.

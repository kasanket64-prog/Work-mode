# Local upload and analysis website

From the repository root, after installing the project's inference, detection,
video and optional plate dependencies:

```powershell
.\.venv\Scripts\python.exe -m camera_system.app --port 8780
```

Open http://127.0.0.1:8780/. Upload a video or image, choose scene analysis or
video plate reading, set the maximum sampled frames, and start. When processing
finishes, open the investigation or download the JSONL result. Scene analysis
includes detection, tracking, person pose observations, temporal image movement,
scheduled relationships and the existing candidate-event reasoning.

The job list shows processed frame counts and a Cancel button for active jobs.
Cancellation stops the worker process; partial output remains in the job folder.
Failed, cancelled and interrupted jobs have a **Retry saved upload** button.
Retry copies the saved media into a new job, retains the mode and frame limit,
and runs from the beginning using the current server/model configuration.
The original result and log are preserved. Only one analysis can run at a time.
Each retry retains another media copy on disk; it does not resume partial inference.
Images produce one result rather than a sampled video frame count.

History also shows the last sampled timestamp and recorded coverage. A completed
job may have stopped at the frame limit; it does not mean the entire video was
analyzed. OpenCV cannot reliably distinguish end-of-file from a decode failure,
so that outcome is shown explicitly. Older outputs without an end reason report
unknown coverage. **Open processing log** displays worker diagnostics as plain
text, including for failed jobs, without needing to browse the output folder.

For a completed scene video, expand **Evaluate against reviewed annotations**
in its history card. Select a JSON file following the annotation format in
[the prototype guide](prototype-guide.md#evaluation), review the displayed
metrics, and download them. Evaluation uses a two-second event matching tolerance
and confirmed relationships. It does not score object detection or plate OCR.
Annotations must cover every sampled timestamp in the run. Single images are
not supported by this video evaluator. Empty annotations mean that a reviewer
verified there were no target events; do not use them as guessed ground truth.

New predictions undergo conservative object-class consistency checks. For
example, a car cannot be the object of `wearing`. Rejected candidates are kept
in `rejected_relations` with their original score and reason, and can be expanded
in the investigation. Unknown labels remain eligible; this is a rule-based
screen, not a trained accuracy improvement or a substitute for evaluation.

## Image-zone entry and exit

Copy `configs/reasoning-zones.example.toml` to your own configuration and set
each rectangle to the relevant area in your fixed camera image. Coordinates
are normalized `[left, top, right, bottom]` from 0 to 1. Choose the detector
class names to observe. Start the website with that reviewed configuration:

```powershell
.\.venv\Scripts\python.exe -m camera_system.app --port 8780 --reasoning-config configs/my-zones.toml
```

The same `--reasoning-config` option works with `camera_system.video_cli`.
New scene videos carry the rectangles in their results and show them in the
investigation. Image and plate jobs do not use zone reasoning. A track's
bottom-center must first establish a stable state and then a different stable
state; the example requires three observations across at least 0.5 seconds.
A boundary margin prevents brief edge jitter from accumulating confirmation.
Missing tracks, long timestamp gaps and resolution changes reset evidence.
Disappearance is not an exit event. These observations describe image regions,
not proof that a person entered a physical room. Moving cameras need separate
motion compensation before using this feature reliably.

To compare settings quickly using saved detections and tracks:

```powershell
.\.venv\Scripts\python.exe -m camera_system.replay outputs/session.jsonl --reasoning-config configs/my-zones.toml --output outputs/replayed.jsonl
.\.venv\Scripts\python.exe -m camera_system.review outputs/replayed.jsonl --output outputs/replayed.html
```

Both output filenames must be new. Replay preserves the original file, recorded
model timings and observations; it recomputes screened relationships, temporal
confirmation and events only. It cannot recover missed detections or improve
track IDs. The session marks `reasoning_replayed=true`. Zone event annotations
must include the matching `zone_id` as well as event type, track and timestamp;
evaluation cannot match events from different zones.

The application uses the project's `.hf_cache` and `.yolo` directories unless
the corresponding environment variables were explicitly set. First use may
download missing weights. The browser upload is local; model downloads do not
upload footage. Explicit official vehicle lookup sends the entered identifier
to that provider only after the user's confirmation.

## Limits and storage

- This server binds to loopback only. It is a single-user development workspace,
  not a public hosting server. No accounts or remote access are provided.
- One job at a time, 512 MB per upload, up to 600 sampled video frames.
- Sampling is every 15th frame. The default 60-frame limit may stop before EOF.
- Scene relationship inference runs every third sampled frame. Skipped frames
  do not repeat earlier relationship predictions.
- Each job has a one-hour process timeout. Keep the server running while a job
  is active. On restart, unfinished metadata is shown as interrupted.
- Uploads, JSONL, review HTML and processing logs are under
  `outputs/workspace/<job-id>/`. They are retained; no automatic deletion occurs.
- Files are named by the server; user filenames are not filesystem paths.
- Network stream support remains in the CLI. This web upload form accepts local
  files; it does not start external camera connections.
- Polished six-screen Figma design, accounts, deployment and camera management
  remain separate release work.

## Temporal observations

For people and vehicles, a bounded three-second track window reports image
displacement after at least three observations over one second. Missing tracks,
resolution changes and gaps greater than 1.5 seconds reset evidence. These are
image-coordinate observations, not measured speed, walking/running recognition,
fall recognition, intent or proof of entering/leaving a scene. Camera motion and
tracking mistakes can affect results. This layer needs labeled footage before
any accuracy claims.

## Verification

2026-10-08: full suite 85 passed, 4 optional tests skipped; Ruff passed.
Zone transitions, boundary jitter, missing tracks, time gaps, resolution changes,
cross-zone evaluation and preservation of replay inputs have runnable tests.
Replaying the saved 19-frame traffic session completed with zero confirmed zone
events and one screened relationship. This is execution evidence, not a labeled
accuracy benchmark. The example zone is not enabled in the default website.

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

Tests cover origin protection, upload extension validation, single-job admission,
command selection, motion continuity, missing tracks and resolution changes.
Mocked tests verify logic; real-model smoke checks separately verify wiring.

Verified on 2026-10-07: 75 tests passed, 4 optional tests skipped; Ruff passed
for the changed Python modules. Real uploaded image, three-frame scene-video,
and three-frame plate-video jobs completed and generated review pages. A real
scene job was cancelled successfully. Browser inspection confirmed the job
history and saved scene investigation load. The scene-video processing summary
reported 79.49 seconds including startup; this is not a real-time claim.

These smoke checks establish execution, not recognition accuracy. The traffic
sample includes an incorrect candidate relationship (a person "wearing" a car),
so relationship calibration and labeled evaluation remain release requirements.
Walking/running/fall recognition and production deployment are not complete.

# Phase 3: recorded video and camera input

The `camera-video` command loads YOLO and RelateAnything once per run, processes frames synchronously, and writes flushed JSON Lines (JSONL). It does not retain a whole video in memory. Tracking is not implemented: region IDs such as `det-0` are local to each frame.

## Setup and recorded video

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,inference,detection,video]"
$env:HF_HOME = "$PWD/.hf_cache"
New-Item -ItemType Directory -Force .yolo | Out-Null
$env:YOLO_CONFIG_DIR = "$PWD/.yolo"
.\.venv\Scripts\camera-video.exe --video path/to/clip.mp4 --stride 5 --max-frames 100 --output outputs/clip-relations.jsonl
```

`--stride 5` processes frame indices 0, 5, 10, and so on. The intervening frames are decoded and discarded, and counted as skipped. `--max-frames` limits processed frames, not decoded frames; the default is 100. Use `--stride 1` to process every frame. The output must be a **new filename**: existing results and inputs are never overwritten. No annotated video is produced.

Use the same `--config`, `--detector-config`, and `--device` options as the image pipeline. The current default vocabulary is still the horse-image smoke vocabulary. For other scenes, supply a relation configuration with suitable predicates and validate its scores before interpreting events.

## Local camera

```powershell
.\.venv\Scripts\camera-video.exe --camera 0 --max-frames 100 --output outputs/camera-session.jsonl
```

Choose another integer if the desired camera has a different index. The operating system must allow camera access. The camera opens after model loading. Ctrl+C releases capture and writes an interrupted summary when Python receives the interrupt. A blocked camera driver may delay interrupt handling. Camera disconnect/read failure records a failed summary and returns exit code 2. Interrupted runs return 130; completed runs return 0.

There is no preview window or audio capture. This turn verified recorded video and simulated camera behavior; a physical webcam was not activated or tested.

## Output and timestamps

Each line is an independent JSON object:

- `session`: source, model configuration, upstream code revision, sampling settings and attribution.
- `models`: model loading time, devices and detector weight SHA256.
- `frame`: source frame index, timestamp and its origin, image size, regions, relations and inference time.
- `summary`: completion status, end reason, decoded/processed/skipped counts, elapsed time and any error.

Video timestamps use OpenCV's position in milliseconds. Missing, negative or repeated/backward positions fall back to monotonically increasing estimates from source FPS. Each fallback is labeled `estimated_from_fps`; it is not a precise presentation timestamp for variable-frame-rate video. If neither timestamps nor FPS are usable, processing fails explicitly. Camera timestamps are elapsed host-monotonic time after each successful read, not hardware capture timestamps or wall-clock time.

Frames with fewer than two objects retain their detections and get an empty relation list plus a reason. A subsequent failure leaves already-flushed frame records available for inspection.

There is no application frame queue. Camera/backend buffering and device-dropped frames are not measured; `device_dropped_frames` is null. Processing may lag real time on CPU. No low-latency live-camera guarantee is made. OpenCV cannot reliably distinguish video EOF from a decode failure, so the terminal reason explicitly says `video_eof_or_decode_failure`. An entirely undecodable video fails.

## Verification, 5 October 2026

Windows / Python 3.12.14 / CPU. A local six-frame MJPEG video was generated from the attributed horse photograph at 5 FPS. This tests actual decoding, BGR-to-RGB conversion, timestamps, object detection and relation inference; it does not test motion understanding.

The bounded smoke run processed indices 0, 2 and 4 at 0.0, 0.4 and 0.8 seconds. Each returned three regions and five ranked relationships. It stopped at the three-frame limit, reporting five frames read and two deliberately skipped. Total wall time including model loading was 16.36 seconds on this run. This is not a throughput benchmark.

Recorded JSONL: [phase3-sample-result.jsonl](phase3-sample-result.jsonl).

Unit tests verify color conversion, stride accounting, timestamp fallback, capture release, camera-read failure, interrupted output preservation, model reuse and refusal to overwrite existing results. The real-video integration test additionally runs to EOF and checks all six frames were read with three skipped.

Full test suite: **40 passed**, including manual-image, automatic-image, and real-video model tests. Lint and dependency consistency checks passed. The installed camera-video command help was also verified.

To run all real-model tests after downloading the models:

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:RUN_MODEL_TESTS = "1"
$env:RUN_DETECTOR_TESTS = "1"
$env:RUN_VIDEO_TESTS = "1"
.\.venv\Scripts\python.exe -m pytest -q
Remove-Item Env:RUN_MODEL_TESTS, Env:RUN_DETECTOR_TESTS, Env:RUN_VIDEO_TESTS
```

Keep `HF_HOME` pointing at the warm cache and keep local YOLO weights. CI skips real-model tests by default. Next phase: persistent object tracking with explicit track lifecycle and ID-switch tests.

# Fullstack Completion Record (2026-11-22)

This note documents the end-to-end engineering upgrades across all four implementation tiers for `Work-mode` (FieldNote).

## Implemented Capabilities

### Tier 1: Correctness, Security & CLI Entry Points
- BOM-safe (`utf-8-sig`) JSON/TOML configuration reading across `config.py`, `detection.py`, `reasoning.py`, and `review.py`.
- `YoloDetector` and `PoseEstimator` support opt-in FP16 `--half` on available CUDA GPUs (`detection.py`, `people.py`).
- `PersonAnalyzer` filters interactions against `relation_threshold`; `replay.py` recomputes `person_analysis[*].interactions` consistently.
- `reasoning.py` emits `started_seconds` on `possible_unattended_bag` alerts; `review.py` validates `track_mapping` and `relations` symmetry, uses `ThreadingHTTPServer`, sets a strict `Content-Security-Policy` header, and ignores closed client sockets.
- `plates.py` adds `--device`, pre-load argument validation, friendly dependency errors, and guarantees complete `summary` records on interruption or failure.
- `cli.py` adds `--preview-dir` sidecar JPEG output (`image-0.jpg`).
- `pyproject.toml` registers `camera-replay`, `camera-plates`, `camera-demo`, and `camera-benchmark`.

### Tier 2: Fullstack Backend, SQLite Persistence & Warm Worker
- `store.py` introduces `WorkspaceStore` (SQLite WAL mode in `outputs/workspace/fieldnote.db`) managing `jobs`, `verdicts`, `zones`, `annotations`, and `evaluations`, synced with `job.json`.
- `worker.py` adds `ModelCache` and `WarmWorkerManager` (`--warm-worker`), reusing loaded YOLO, RelateAnything, pose, and ALPR weights across sequential jobs with cooperative cancellation and automatic restart after crashes.
- `app.py` adds:
  - `GET /api/system/status` (runtime readiness, device, extra availability, disk/job counts, warm-worker state).
  - `GET /api/events` (Server-Sent Events publishing live job state transitions).
  - `DELETE /api/jobs/<id>`, `GET`/`PUT /api/jobs/<id>/verdicts`, `GET`/`POST`/`DELETE /api/zones`, `POST /api/jobs/<id>/replay` (backing up `original-result.jsonl` and preserving saved verdicts), `GET`/`PUT /api/jobs/<id>/annotations`, `GET /api/jobs/<id>/evaluations`, and `GET /jobs/<id>/frames/<file>` sidecar thumbnail serving.
  - Upload controls (`X-Frame-Stride`, `X-Reasoning-Profile`, `X-Vehicle-Crops`, `X-Device`), `--strict-uploads` magic-byte checks, `--host` + `--auth-token-env` bearer/header/cookie authentication, and `--retention-days` / `--max-disk-mb` automatic pruning.

### Tier 3: Workspace & Review UI Studios + Live Stream Resilience
- `workspace.html` / `workspace.css` / `workspace.js`: live system readiness strip, upload configuration controls (stride/profile/device/vehicle-crops), `EventSource('/api/events')` updates with 3-second polling fallback, two-step job deletion, and evaluation F1 badges.
- `review.html`: sidecar `preview_path` thumbnails, keyboard navigation (`Left`/`Right`/`Space`/`B`/`P`/`1`/`2`/`3`/`0`), server-synced verdicts (`PUT /api/jobs/<id>/verdicts`), interactive canvas-drawn Zone & Replay Studio, and interactive Annotation & Evaluation Studio.
- `video.py` & `video_cli.py`: `--stream-credentials-env` (injected only into capture handles, never stored in `FrameSource.source` or JSONL outputs), `--live-buffer` latest-frame background reader tracking `dropped_frames`, `--reconnect-attempts` exponential backoff, `--auto-stride`, `--pose-model`, `--preview-dir`, `--half`, and `--static-diff-threshold`.

### Tier 4: Kalman Tracker, Calibration Benchmark & Containers
- `tracking.py`: adds `KalmanTracker` (constant-velocity 2D state `[cx, cy, vx, vy, w, h]`, optional `scipy.optimize.linear_sum_assignment` Hungarian matching with greedy fallback, preserving raw observed detection boxes) and `make_tracker("iou" | "iou-motion" | "kalman")`, wired to `video_cli.py --tracker` and `worker.py`.
- `benchmark.py` (`camera-benchmark`): evaluates session JSONL vs reviewed truth, computes per-event-type precision/recall/F1, attributes false negatives/positives, runs relation-threshold replay sensitivity sweeps (`threshold_sweep`), and writes `report.json` + `report.md`.
- `Dockerfile`, `docker-compose.yml`, `.dockerignore`: multi-stage Python 3.12-slim `cpu` and `cuda` targets, non-root `fieldnote` user, `/cache/hf` & `/cache/yolo` volume mounts, and `/api/system/status` healthcheck.

## Container Usage

```powershell
docker compose up --build
```

For NVIDIA GPU deployments, run:

```powershell
docker compose --profile gpu up --build
```

Override `FIELDNOTE_AUTH_TOKEN` in your shell or `.env` file before exposing the port beyond localhost.

## Honest Verification Scope

- Unit, integration-boundary, and HTTP end-to-end tests exercise all new modules using deterministic synthetic fixtures (`make_demo()`, NumPy arrays) and explicit stubs at the neural-network (`YOLO`, `RelateAnything`, `fast_alpr`) and `cv2.VideoCapture` boundaries.
- Live hardware accuracy benchmarks on proprietary CCTV streams, real GPU inference latency measurements, and container runtime builds require the optional `[inference,detection,video,plates]` model stacks, weights, and a live Docker/GPU host, and are **not** claimed from lightweight unit runs.

## Verified Results (2026-11-22)

- **Pytest (`.\.venv\Scripts\python.exe -m pytest -q`)**: `165 passed, 4 skipped` (run outside Windows terminal sandbox so loopback HTTP server & SSE socket tests bind locally).
- **Ruff (`.\.venv\Scripts\python.exe -m ruff check .`)**: `All checks passed!`.
- **Frontend JS syntax & unit checks**: `node --check` on `workspace.js` and inline `review.html` script plus DOM-free JS logic tests (`matchesJob`, `uploadError`, `linkEventFrames`, `annotationDraft`) passed inside `tests/test_e2e_browser.py`.
- **CLI Smoke Checks (Windows editable scripts under `.venv\Scripts\`)**:
  - `camera-relations.exe --image examples/horse.jpg --regions examples/horse.regions.json --validate-only`: exit code `0` (`"status": "validated"`).
  - `camera-demo.exe --output-dir <temp>\demo`: exit code `0` (`review.html` generated).
  - `camera-benchmark.exe <temp>\demo\session.jsonl --truth <temp>\demo\truth.json --output-dir <temp>\bench`: exit code `0` (`report.json` and `report.md` generated).

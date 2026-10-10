# Work-mode

Local camera-analysis prototype: detection, tracking, relationships, temporal
event candidates, and an upload/review website.

**Start here:** [project status, roadmap and documentation index](docs/README.md).
**Next implementation work:** [prioritized audit backlog](docs/audit-backlog.md).
The prototype runs locally; real-world accuracy is not yet measured. Substantial
work remains before deployment. See the [publication verification](docs/publication-2026-10-10.md).

**Upload website:** run `python -m camera_system.app` from this repository and open http://127.0.0.1:8780/. Upload a video/image, process it locally, open the investigation and export results. [Setup, limits and storage](docs/local-workspace.md).

**Vehicle details:** [official vehicle lookups](docs/vehicle-lookups.md) for Dutch plates, UK plates with an API key, and US VINs. [Local plate OCR](docs/plate-reading.md) includes an optional vehicle-crop experiment; readings require review. The review also links to India and Norway's official services.

**Office tuning:** a 40-object office profile, optional relationship scheduling, and per-stage timing are available. See the [detection comparison guide](docs/detection-comparison.md) and [performance measurements](docs/performance.md). These are local prototype measurements, not validated accuracy or real-time guarantees.

**Visible-person analysis:** optional pose, upper-body appearance, image movement and interaction observations, with a redesigned Fieldnote review page. [Run instructions and limits](docs/person-analysis.md).

**Current prototype:** detection → video/camera frames → track IDs → relations → temporal confirmation → candidate events → local review/API and evaluation. Start with the [current run guide](docs/prototype-guide.md) and [latest continuation verification](docs/continuation-2026-10-09.md). The phase sections below record earlier milestones.

The original Phase 1 milestone was relationship prediction from an image and manually supplied object regions. Its commands remain available below.

Based on RelateAnything by Maëlic Neau — https://github.com/Maelic/RelateAnything

## Quick start (Windows PowerShell)

Requirements: Git, Python 3.12 or 3.13, internet for dependencies and the first model download. CPU is the default; no camera or GPU is required. Allow several GB of disk space for PyTorch, model weights, and caches.

```powershell
git clone https://github.com/kasanket64-prog/Work-mode.git
cd Work-mode
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,inference]"
$env:HF_HOME = "$PWD/.hf_cache"
.\.venv\Scripts\camera-relations.exe --image examples/horse.jpg --regions examples/horse.regions.json --config configs/phase1.toml --output outputs/horse.json
```

On Linux/macOS, create the environment with `python3.12 -m venv .venv`, use `.venv/bin/python` and `.venv/bin/camera-relations`, and set `export HF_HOME="$PWD/.hf_cache"`. All example commands run from the repository root; no activation is necessary.

The real command downloads the pinned Hugging Face snapshot, loads the model and text encoder, and predicts ranked relations. `outputs/horse.json` contains subject/object region IDs, predicate strings, scores, image hash, effective settings, resolved device, and separate loading/prediction timings. Empty predictions are valid for some inputs and thresholds. Neither an empty nor a nonempty result is an accuracy measurement.

## Phase 2: automatic object detection

The image runner now accepts `--detect` instead of a manual regions file. YOLO11n provides a small CPU-friendly baseline; it is not fine-tuned for CCTV. Install the optional detector and run:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,inference,detection]"
$env:HF_HOME = "$PWD/.hf_cache"
New-Item -ItemType Directory -Force .yolo | Out-Null
$env:YOLO_CONFIG_DIR = "$PWD/.yolo"
.\.venv\Scripts\camera-relations.exe --image examples/horse.jpg --detect --detector-config configs/detector.toml --output outputs/horse-detected.json
```

YOLO downloads `yolo11n.pt` on first use. Keep it for subsequent runs; weights are ignored by Git. The result records its SHA256. Model paths in `configs/detector.toml` resolve from the working directory. The detector returns original-image pixel boxes; no second resize or coordinate scaling is applied by our adapter. Invalid/degenerate detections are discarded, valid boxes are clipped to image bounds, sorted by confidence and capped before IDs are assigned. The effective cap cannot exceed the relation model's `max_regions`.

Detector confidence is stored as `detection_score` separately from relation `score`; it is not multiplied into relation scores. `det-*` IDs are unique within one image only, not tracking IDs. Zero or one detection produces an empty relations list with a skip reason, without loading RelateAnything. `--detect` and `--regions` are mutually exclusive. `--validate-only` remains for manual regions.

The detector uses Ultralytics' documented [prediction API](https://docs.ultralytics.com/modes/predict/). See [Phase 2 verification](docs/phase2-verification.md). Video input and persistent tracking are now implemented; see the current run guide.

## Phase 3: video and camera input

See [video/camera instructions and verification](docs/phase3-video.md). Install `.[dev,inference,detection,video]`, then run from the repository root:

```powershell
.\.venv\Scripts\camera-video.exe --video path/to/clip.mp4 --stride 5 --max-frames 100 --output outputs/clip.jsonl
.\.venv\Scripts\camera-video.exe --camera 0 --max-frames 100 --output outputs/camera.jsonl
```

Both models load once. Results are flushed per frame, followed by a summary. Output paths must not already exist. The camera path has no preview window; Ctrl+C stops processing. Region IDs are local to a frame; persistent track IDs connect observations and relation endpoints across frames.

## Lightweight checks (no weights)

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\camera-relations.exe --image examples/horse.jpg --regions examples/horse.regions.json --validate-only
```

`--validate-only` reports `status: validated` and makes **no predictions**. Unit tests use an explicit stub at the model boundary; production inference never substitutes a mock. The real model integration test is opt-in:

```powershell
$env:HF_HOME = "$PWD/.hf_cache"
$env:RUN_MODEL_TESTS = "1"
.\.venv\Scripts\python.exe -m pytest tests/test_integration.py -q
Remove-Item Env:RUN_MODEL_TESTS
```

## Your own image and regions

Copy `examples/horse.regions.json` and supply at least two objects. Each needs a unique string `id` and `box: [x1, y1, x2, y2]` in **original image pixels**. `label` and `track_id` are optional. Coordinates must be finite, nonnegative, inside the image, with positive box area. Coordinates refer to the stored image orientation; the runner does not rotate it using EXIF metadata. Labels are display metadata, not model features.

```powershell
.\.venv\Scripts\camera-relations.exe --image path/to/image.jpg --regions path/to/regions.json --config configs/phase1.toml --output outputs/custom.json
```

Edit `configs/phase1.toml` to choose a vocabulary, model revision, top-k, region cap, or score filter. `min_score = 0.0` keeps the smoke test independent of an unvalidated decision threshold. New vocabularies require their own validation/calibration. Scores are not guaranteed CCTV probabilities. Single images cannot establish events such as following or leaving a bag.

Inputs exceeding the configured region cap are rejected to preserve ID mappings. Detector confidences are stored separately from relation scores. The video runner constructs one predictor and reuses it across frames.

## Layout

```text
src/camera_system/   inference, tracking, reasoning, evaluation, CLI and local website
configs/            model, detector and event/zone settings
examples/           attributed test image and its manual regions
tests/              unit tests and opt-in real inference smoke test
docs/               project overview, audit backlog, guides and verification records
scripts/            launch script and detection/tracking/performance diagnostics
outputs/            local generated sessions and reviews (ignored by Git)
.github/workflows/  Windows/Linux lightweight CI
```

## Dependencies and upstream

The `inference` extra installs `relsgg[hub]` directly from upstream commit `06766fdf56752ca535fc9b971fca99ce563676d0`; model code is not vendored or modified. The example pins model snapshot `2db90096be5217bdc7a9003c042950f45723d105`. Dependency ranges come from upstream; see the verification record for versions actually tested. Code and model pins alone do not lock every transitive dependency.

For CUDA, install a matching CUDA-enabled PyTorch/torchvision build satisfying upstream requirements, verify `torch.cuda.is_available()`, then pass `--device cuda`. `--device auto` selects CUDA when available and CPU otherwise. Explicit CUDA requests fail clearly if unavailable.

For offline use, complete one real run using the same `HF_HOME`, then set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. Preserve the entire cache, including snapshot metadata and text encoder. A cold cache still needs internet. If installing fails, check Python version, Git access, package-index connectivity, and upstream's dependency constraints; do not silently loosen the model requirements.

See [architecture and next phases](docs/architecture.md), [verification](docs/verification.md), and [third-party notices](THIRD_PARTY_NOTICES.md). CI exercises the lightweight package only; it does not download weights or certify GPU support.

# Phase 2: automatic detection verification

Verified on Windows, Python 3.12.14, CPU, 5 October 2026. The sample image now runs without a manual regions file. Ultralytics 8.4.173 / YOLO11n detects objects and the existing pinned RelateAnything model predicts their relationships.

## Observed sample result

- Horse detection confidence: 0.927902.
- Person detection confidence: 0.793851.
- Background object labeled car: 0.401113 (not independently annotated).
- Person -> riding -> horse relation score: 0.740043.
- Horse -> carrying a rider -> person relation score: 0.344596.

See [full recorded result](phase2-sample-result.json), including all five ranked relations, boxes, configuration and detector weight SHA256. The score differs from the manual-box run because the boxes and object set differ; this is not evidence of an accuracy improvement. Background detections and low relation scores still require evaluation.

The first run spent 12.84 seconds loading/downloading the detector and detecting objects, 24.62 seconds loading the relation model, and 1.57 seconds predicting relations. These are single-run timings including cold startup, not measured steady-state throughput.

## Repeat the smoke test

Run from the repository root after installing `.[dev,inference,detection]`:

```powershell
$env:HF_HOME = "$PWD/.hf_cache"
New-Item -ItemType Directory -Force .yolo | Out-Null
$env:YOLO_CONFIG_DIR = "$PWD/.yolo"
.\.venv\Scripts\camera-relations.exe --image examples/horse.jpg --detect --output outputs/horse-detected.json
```

For the two real-model tests, first download both models with the command above, then:

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:RUN_MODEL_TESTS = "1"
$env:RUN_DETECTOR_TESTS = "1"
.\.venv\Scripts\python.exe -m pytest -q
Remove-Item Env:RUN_MODEL_TESTS, Env:RUN_DETECTOR_TESTS
```

The offline flags apply to Hugging Face; YOLO weights must already exist at the configured path. The automatic integration test also parses stdout as JSON to catch unexpected library messages in the result stream. Default CI skips both real-model tests and requires no model downloads.

## Validation

- 30 lightweight tests passed before the real-model tests.
- Real automatic image inference completed successfully.
- Full regression suite including manual-region and automatic-detection real inference: 32 passed in 41.15 seconds. Hugging Face offline flags were enabled and local YOLO weights were reused.
- Lint and dependency consistency checks passed.

Unit tests cover clipping and original-pixel coordinates, score sorting, confidence filtering, region caps, malformed configuration, empty/single detections, and preservation of detection confidence separately from relation scores. The manual Phase 1 entry point remains available unchanged.

Both OpenCV distributions are installed because upstream RelateAnything requires headless OpenCV while Ultralytics requires full OpenCV. They expose the same `cv2` module; this combination was exercised by the real tests here. Avoid uninstalling one independently in this environment; rebuild the virtual environment if OpenCV becomes inconsistent.

This phase does not implement video, tracking, alerts, detector accuracy evaluation, or GPU performance verification. Next: recorded-video input with timestamped frames and reused model instances.

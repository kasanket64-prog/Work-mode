# Visible-person analysis and the Fieldnote review

The optional person-analysis layer works with images and video. It provides visible observations with separate evidence/status fields; it does not infer personal identity, mood or intent. The analyzer does not use facial recognition or an emotion classifier.

## Run an image

The existing `detection` extra includes the Ultralytics pose runtime. First use downloads the official `yolo11n-pose.pt` weights. Subsequent runs reuse the local file. Model name, device and weight SHA256 are recorded.

```powershell
$env:HF_HOME = "$PWD/.hf_cache"
$env:YOLO_CONFIG_DIR = "$PWD/.yolo"
.\.venv\Scripts\camera-relations.exe --image examples/horse.jpg --detect --analyze-people --include-preview --output outputs/people.json
.\.venv\Scripts\python.exe -m camera_system.review outputs/people.json --output outputs/people.html --serve --port 8766
```

Open `http://127.0.0.1:8766/`. Output paths should be new. For other scenes use your own image and an appropriate relation vocabulary (for example `--config configs/camera.toml`). The sample retains the horse/riding vocabulary.

## Run video

```powershell
.\.venv\Scripts\camera-video.exe --video path/to/clip.mp4 --config configs/camera.toml --reasoning-config configs/reasoning.toml --analyze-people --include-preview --max-frames 100 --output outputs/people-video.jsonl
.\.venv\Scripts\python.exe -m camera_system.review outputs/people-video.jsonl --output outputs/people-video.html
```

Pose loads once alongside the existing models. More inference increases runtime. Video motion estimates need a continuing track, unchanged dimensions, and a 0.25–2 second observation interval. Larger gaps yield unknown movement. This is image displacement, not meters/second or recognition of walking/running. Camera motion is not compensated.

## What observations mean

| Observation | Evidence | Limits |
| --- | --- | --- |
| Upper-body color | Dominant hue group in an interior pixel crop | Not clothing segmentation; skin, background, lighting and occlusion may affect it. Small regions or mixed color are marked uncertain/unknown. |
| Posture | COCO shoulders/hips/knees/ankles from YOLO pose; all required keypoint scores ≥ 0.5 | Simple joint geometry, not trained activity classification. Bent knees do not establish sitting; uncertain configurations remain unknown. |
| Movement | Track-center change divided by elapsed time and image diagonal | Camera movement, detector jitter and ID switches affect results. Single images cannot provide movement evidence. |
| Interactions | Relation model candidates scoring at least 0.6 | Scores are not calibrated accuracy. The existing temporal-relation panel separately identifies multi-frame support. |

Pose matching is one-to-one by box overlap, with a minimum IoU of 0.3. Unmatched people retain color/interaction observations but unknown posture. The body model returns 17 landmarks; this is not facial expression or gaze estimation.

## Review experience

Fieldnote is the local project's review UI: source frame overlays, box/pose toggles, person observation cards, model interactions, a frame timeline, event navigation, result-file loading and JSON export. It accepts both image JSON and video JSONL. Opening a result file reads it in the browser; this does not trigger inference or upload it anywhere. The API continues to represent the server's originally loaded session.

`--include-preview` explicitly embeds JPEG thumbnails (up to 960 × 640) in result files. Without it, the canvas displays box layouts. Previews stay local but increase output size and travel with any exported/shared file; choose the flag accordingly. The viewer limits browser-loaded files to 100 MB. Long runs should be reviewed in smaller segments.

The page labels simulated data, frame-only candidates and unknown observations. It makes no claim to analyze everything or to determine a person's actual mood. Its counts and scores are observations from the selected run, not dataset performance metrics.

## Upstream and validation

- [Ultralytics pose API](https://docs.ultralytics.com/tasks/pose/)
- [YOLO11 model variants](https://docs.ultralytics.com/models/yolo11/)
- Weights: https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n-pose.pt
- Ultralytics code/model licensing follows the existing third-party notices. No pose weights are committed.

Unit tests cover crop size/color, missing keypoints, posture geometry, single-image uncertainty, movement and timestamp gaps, and image-result compatibility. Real pose inference and browser verification are recorded in the current verification notes. Real CCTV posture/activity accuracy remains unmeasured.

Verified locally on 6 October 2026: the full suite passed 64 tests with real model checks enabled. After adding person analysis and embedded previews to the video integration check, the focused video/person/review suite passed all 9 tests. The video fixture repeats the sample photograph; this verifies processing and persistence, not real-world action accuracy. The sample image was also processed through the detector, relation model and pose model, and inspected in the browser review. No physical camera or labeled CCTV benchmark was evaluated.

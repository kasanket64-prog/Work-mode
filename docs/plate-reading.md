# Local video plate reading

Install and run from the repository in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[plates]"
.\.venv\Scripts\python.exe -m camera_system.plates "C:\path\traffic.mp4" --output outputs/plates.jsonl --stride 15 --max-frames 60
.\.venv\Scripts\python.exe -m camera_system.review outputs/plates.jsonl --output outputs/plates.html --serve --port 8769
```

Outputs must be new paths. Open http://127.0.0.1:8769, step through frames, and
review plate crops in Vehicle details. “Use text for review” fills the lookup
field; select the correct service, correct the number, and explicitly confirm.
US service accepts VINs, not plates. No country is inferred automatically.

The optional [FastALPR](https://github.com/ankandrew/fast-alpr) package uses
YOLOv9-t-384 and cct-xs-v2-global models. First use downloads public weights;
subsequent inference is local. Video and crops are not uploaded. Retained JSONL
and HTML contain plate crops and should be treated as local footage data.
Review upstream package and model licenses before redistribution.

Readings are candidates, never verified identifiers. Minimum character score
below 0.8 is marked uncertain; even a high score requires review and is not a
calibrated correctness probability. Exact text repetition across sampled frames
is displayed, but is not tracking or temporal identity confirmation. Small,
blurred and angled plates may be missed or misread. No registry query runs
automatically. This bounded scanner samples at most 60 frames by default; it
does not perform full scene relationship analysis or live camera monitoring.

## Vehicle crop experiment

Install `.[plates,detection]` and add `--vehicle-crops` to the scanner command
to run the existing YOLO11n detector at 1280 pixels with up to 40 detections.
Cars, buses, trucks and motorcycles are cropped from the original frame before
plate detection/OCR; plate boxes are mapped back to original coordinates.
Overlapping plate detections are deduplicated spatially. The default full-frame
mode remains available for comparison. Cropping can miss vehicles the detector
does not find, and costs additional CPU time. More candidates are not proof of
better accuracy. Neither mode invents missing image detail or queries registries.

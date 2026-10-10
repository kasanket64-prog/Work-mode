# CPU performance investigation

Video frame records now include `stage_seconds` for detection, tracking, relations, reasoning, person analysis and preview generation. These wall-clock measurements exclude decoding and JSON writing. Model startup is recorded separately in the models record; session elapsed time includes startup. CUDA execution can be asynchronous, so these counters are CPU diagnostics, not synchronized GPU kernel benchmarks.

On 6 October 2026, the 40-object office profile processed the same 12 frames at stride 7 with every analysis enabled:

| Stage | Mean seconds/frame |
| --- | ---: |
| Detection | 0.177 |
| Tracking | 0.001 |
| Relationships | 0.491 |
| Temporal reasoning | <0.001 |
| Person analysis including pose | 0.078 |
| Preview | 0.018 |
| Complete measured frame | 0.765 |

The full run including startup took 19.37 seconds. Detection boxes/IDs and relation results exactly matched the previous saved run. The previous run averaged 2.75 seconds/frame: this variation is not evidence of a code optimization. No model, resolution, confidence threshold, or CPU-thread default was changed in this performance step. The latest measurement is still slower than the 0.28-second source sampling interval.

A fixed-image relation benchmark tried 8, 4, 2 and 1 CPU threads, with one warm-up and three measured predictions per setting. Median times were 0.431, 0.412, 0.531 and 0.901 seconds respectively. Candidate sets were identical; the maximum score change across settings was below 0.000001. A roughly 4% single-image gain at four threads does not justify a new global default. Benchmark ordering and system load can affect results.

```powershell
.\.venv\Scripts\python.exe scripts/benchmark_relation_cpu.py --image outputs/video-inspection-20261006/27091-361827476_medium.jpg --session outputs/office-40-verified-20261006.jsonl --output outputs/relation-cpu-new.json
```

The benchmark image must correspond to the first session frame and its original dimensions. Use a new output path. Current local results are in `outputs/relation-cpu-benchmark.json` and `outputs/office-40-profiled-20261006.jsonl`. Seven video tests passed, and the real run's completion, timing fields, and prediction consistency were checked.

An order-balanced run on the supplied office clip used four fresh processes in order 1, 3, 3, 1. It processed 12 frames at stride 7 per run, excluding the first three frames from warm averages. Every-frame scheduling averaged 2.750 seconds/frame; every-third-frame scheduling averaged 1.182 seconds/frame, a measured 57.0% reduction. All four runs had identical detection/tracking records. Relationship calls fell from 12 to 4 per run. This is a short single-clip CPU benchmark, not an accuracy result; scheduling can miss brief relationships. Report: `outputs/scheduling-balanced-20261006/comparison.md`.

## Optional relationship scheduling

For an order-balanced speed comparison, run from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts/benchmark_scheduling.py --video "path/to/office.mp4" --output outputs/scheduling-comparison
```

Use a new output directory and the office clip (at least 78 frames). This runs the real CLI in four fresh processes in interval order 1, 3, 3, 1, with identical settings otherwise. Each run processes 12 frames at stride 7. Warm averages exclude the first three processed frames, leaving three complete scheduling cycles. The report compares detection/tracking records and checks that skipped frames contain no relation evidence. Two repeats per mode on one clip are a diagnostic, not a statistically robust throughput or accuracy benchmark. Avoid other CPU-intensive work during measurement.

`camera-video --relation-every 3` runs fresh relationship inference on processed frames 0, 3, 6, and so on. The default is 1, preserving every-frame behavior. This interval counts processed frames after `--stride`, not original video frames. Detection, tracking, optional person analysis and geometric event monitoring still run on every processed frame.

Skipped frames have empty relations and confirmations, with `relation_skip_reason = "scheduled_skip"`; the review displays a scheduled-pause message. Previous predictions are never copied forward or counted as fresh evidence. The reasoner still observes object visibility, motion and proximity on skipped frames; previously established bag associations can still support geometric unattended-event monitoring. This is not equivalent to observing an absence of a relationship. Fewer relationship checks can miss brief interactions or delay/prevent confirmation. Person interaction lists are empty on skipped frames too.

For the 25 FPS office clip, stride 7 and relation-every 3 give a fresh check every 0.84 seconds, within the default 1.5-second relation history gap. Choose intervals compatible with the reasoning window, required hit count and gap settings. No automatic threshold adjustment is made. Validate event behavior on labeled video before deployment.

```powershell
.\.venv\Scripts\camera-video.exe --video "path/to/office.mp4" --detector-config configs/detector-office.toml --config configs/camera.toml --stride 7 --max-frames 12 --relation-every 3 --analyze-people --include-preview --output outputs/office-scheduled.jsonl
```

Verified locally on 6 October 2026: all 12 frames completed, with exactly four fresh relationship checks and eight explicitly skipped checks. Skipped frames contained no relations or confirmations; object counts remained at most 35. In this run, fresh-check frames averaged 3.48 seconds and skipped frames 0.99 seconds (overall 1.82 seconds/frame; startup plus processing 50.10 seconds). CPU timings varied substantially from the earlier run, so these separate runs do not establish an overall speedup percentage. Scheduling reduces relation-model calls by two thirds, with the coverage tradeoff above; it still does not meet the 0.28-second source interval. Sixteen video/reasoning tests and three review tests passed. The local API test required running outside sandbox socket restrictions. Review artifact: `outputs/office-40-scheduled-review.html`.

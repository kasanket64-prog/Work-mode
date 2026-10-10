# Publication verification — 10 October 2026

## Scope

GitHub initially contained only the README at
`856737e2cde5c583040e3e09ca0ec5f375153f6f`. The local
`codex/phase1-relateanything` checkout already contained the application,
configuration, examples, tests, diagnostics and documentation as uncommitted work.
This publication preserves and verifies that implementation rather than replacing it.

The requested first milestone is implemented: Python package/setup instructions,
pinned RelateAnything adapter, strict input validation, an attributed sample image,
manual region input and real pretrained CPU inference with JSON output. Automatic
YOLO detection and bounded video/camera input are also implemented. Existing
tracking, temporal candidates, local upload/review, evaluation and optional person/
plate helpers are included as prototype features, with the limits below.

Upstream remote HEAD was checked against the dependency pin:
`06766fdf56752ca535fc9b971fca99ce563676d0`. The installed package provenance
matches that commit. Model snapshot:
`2db90096be5217bdc7a9003c042950f45723d105`.

Changes during this continuation:

- Fixed stale progress after a completed result is replaced with a larger file.
  Incremental reads now apply only to active jobs. Existing regression tests
  cover replacement, partial lines, incremental reads and restart caching.
- Made Ultralytics import classification explicit for consistent lint results
  across the inference and fresh lightweight environments.
- Updated publication/setup status and evaluation-scope documentation, reconciled
  the six audit entries with their existing fixes, and renamed the CI workflow.

## Checks

- Fresh Python 3.12 lightweight environment: `pip install -e ".[dev]"` succeeded;
  125 tests passed, four optional model tests skipped; `pip check` passed.
- The installed `camera-relations --validate-only` command succeeded there,
  without inference dependencies. Validation is not a prediction.
- Real CPU integration tests cover manual sample regions, automatic detection,
  generated-video decoding/tracking/temporal output and person pose inference.
  They use actual cached weights with Hugging Face offline mode, without a mock
  fallback. Final complete suite: **129 passed in 34.80 seconds**, no skips.
- Ruff passed in both environments (including the freshly installed version);
  dependency consistency also passed in the inference environment.
- A separate real CLI run reproduced `person-1 -> riding -> horse-1` with score
  0.55174905, and the reverse `carrying a rider` result with score 0.01989860.
  Loading took 15.34 seconds; prediction took 0.66 seconds. These single-run
  timings and scores are not accuracy measurements or steady-state benchmarks.
  Local output: `outputs/publish-horse-20261010.json` (ignored).
- Initial sandbox execution could not use test temporary files and loopback
  sockets. Tests were rerun with approved local access; no assertions were removed.

## Limits

No physical webcam, network camera, CUDA or Linux execution was verified locally.
Generated video repeats the sample photograph and does not validate motion,
occlusion or real-camera accuracy. No fresh cold-cache model download was needed;
the real inference environment and model cache already existed. The clean setup
check covers the lightweight package only. Optional plate OCR and external lookup
providers were not exercised live. Browser appearance was not rechecked.

Evaluation covers first-to-last sampled timestamps only. Independently labeled
footage and calibration remain necessary before using event candidates operationally.
The website is a loopback-only prototype, not an internet deployment. Model weights,
caches, user videos, local output sessions, and the virtual environment are ignored
and are not part of the source publication. Synced `sources/` files were untouched.

The Windows/Linux CI workflow runs lightweight tests only. Its remote outcome
must be checked separately from these local results.

## GitHub outcome

Published as commit `717831fc7420fed1d9b6b18481592ab3991b6a79` on `main`.
[GitHub Actions run 38034253037](https://github.com/kasanket64-prog/Work-mode/actions/runs/38034253037)
completed successfully on all four combinations of Windows/Linux and Python 3.12/3.13.

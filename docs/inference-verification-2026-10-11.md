# Inference continuation — 11 October 2026

## Starting state

Fetched GitHub before editing. Local `codex/phase1-relateanything` and
`origin/main` both started at `f8aaa40`, with a clean working tree. The requested
local package, pinned pretrained adapter, setup instructions, horse image and
manual regions, CLI, and integration test already existed. Detection, video,
tracking, temporal rules and the local review/API prototype also existed.
This continuation preserves those implementations rather than recreating them.

## Change

The manual-region loader now accepts UTF-8 JSON with or without a byte-order
mark. A Windows-style UTF-8 file previously failed with `Unexpected UTF-8 BOM`
before inference. The existing CLI test now exercises both encodings through
validation, stubbed inference and input-overwrite protection. README instructions
describe supported encoding and validation before loading the model.

## Checks

- Phase-one tests after the fix: **20 passed**.
- Full lightweight suite after the fix: **133 passed, 4 optional model tests skipped**.
- Real pretrained integration run: **133 passed, 3 skipped**, including
  `test_real_sample`, with `RUN_MODEL_TESTS=1`, `HF_HUB_OFFLINE=1`,
  `TRANSFORMERS_OFFLINE=1` and `HF_HOME` pointing at `.hf_cache`.
  This run started before the encoding change; the final phase-one regression
  separately covers that change. One pytest cache-write warning did not affect
  test execution. The final lightweight run used a fresh temporary directory
  and disabled pytest's cache provider, with no warnings.
- Manual validation of the BOM-bearing sample: passed after reproducing failure.
- Ruff, `pip check`, and `git diff --check`: passed.

The real CLI demo also passed after the fix, saving the local ignored artifact
`outputs/horse-20261011.json`. It predicted `person-1 riding horse-1` with score
0.551749 and the reverse `carrying a rider` with score 0.019899. Loading took
69.11 seconds and prediction 3.31 seconds in this single CPU run. These are
uncalibrated model scores, not accuracy estimates or sustained throughput.

Reproduce from the repository root after the README setup (use a new output path):

```powershell
$env:HF_HOME = "$PWD/.hf_cache"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
.\.venv\Scripts\camera-relations.exe --image examples/horse.jpg --regions examples/horse.regions.json --config configs/phase1.toml --output outputs/horse-new.json
```

Offline mode requires a previously populated cache; omit the two offline
variables for the initial download.

Environment: Windows, Python 3.12.14, torch 2.14.1, torchvision 0.29.1,
relsgg 1.0.0, transformers 5.18.0. The upstream code and model snapshot pins
remain unchanged. Existing dependencies and cached weights were reused;
a fresh internet installation was not repeated.

The first sandboxed test attempt could not create pytest temporary files and
could not connect to localhost test servers. Rerunning with those permissions
resolved these infrastructure failures; no application workaround was added.

## Remaining roadmap

The [project overview](README.md) remains the source of current work order.
Detector integration, camera/video input, tracking, temporal verification,
event candidates, optimization and dashboard/API have prototype implementations.
They still need representative evaluation and hardening. Next substantive work
is independently reviewed ground truth, baseline precision/recall and false-alarm
measurement, then measured improvements to the weakest component.

Physical cameras, GPU execution, Linux execution and recognition accuracy were
not verified in this continuation. A successful horse-image prediction is an
execution check, not evidence of real-world event accuracy.

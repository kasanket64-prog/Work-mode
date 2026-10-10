# Audit backlog

Audit date: 9 October 2026. Assumed workload: a single local user, one analysis
worker at a time, with retained job history. **All six findings have fixes and regression coverage as of 10 October 2026.**
The descriptions below preserve the original failure cases. See
[publication verification](publication-2026-10-10.md) for current checks.

## Must fix

### AUD-01 — Relationship scheduling exceeds evidence expiry

Status: fixed; covered by `tests/test_audit_regressions.py`. Code: [app.py](../src/camera_system/app.py),
[reasoning.py](../src/camera_system/reasoning.py).

Website defaults sample every 15 frames and infer relationships every third
sample. At 29.97 FPS, fresh evidence is 1.5015 seconds apart, exceeding the
1.5-second expiry. A controlled high-confidence sequence produced zero
confirmations versus 28 with inference on every sample. This does not diagnose
the latest command-line video's zero confirmations; that run inferred every sample.

Fix: make scheduling compatible with the evidence window, required hits and
expiry. Acceptance: sustained high-confidence evidence confirms at 25, 29.97 and
30 FPS under supported website settings; skipped frames never count as fresh evidence.

### AUD-02 — Client disconnect releases a running job's slot

Status: fixed; covered by `tests/test_audit_regressions.py`. Code: [app.py](../src/camera_system/app.py).

The upload handler starts a worker before sending its response. A response write
failure enters cleanup and clears `jobs.active` while the worker still runs.
A mocked disconnect reproduced a started worker, a processing job and an empty
active slot. This permits overlapping analysis and breaks cancellation ownership.

Fix: transfer slot ownership to the worker after successful startup; response
failure must not release it. Acceptance: a disconnected response cannot admit a
second job, startup failure releases the slot, and completion clears only its own slot.

### AUD-03 — Image output overwrites existing files

Status: fixed; covered by `tests/test_audit_regressions.py`. Code: [cli.py](../src/camera_system/cli.py).

Selected inputs are protected, but `write_text` overwrites other existing paths.
An intercepted write accepted `README.md` as output; the audit did not overwrite it.

Fix: use exclusive creation as the video/review commands do. Acceptance: an existing
output or unrelated file is rejected and remains byte-for-byte unchanged.

### AUD-04 — Evaluation duration need not match processed coverage

Status: fixed; covered by `tests/test_audit_regressions.py`. Code: [review.py](../src/camera_system/review.py).

The evaluator only checks that sample times fit within the supplied duration.
One false alert in the same truncated 10-second run scored 360 alarms/hour with
duration 10, but 1 alarm/hour with duration 3600. Both inputs were accepted.

Fix: record and validate matching evaluated coverage; require explicit matching
annotation scope for partial runs. Acceptance: mismatched full-video annotations
cannot dilute a partial run's false-alarm rate, while valid complete and explicitly
scoped partial evaluations remain supported. Do not equate last sample time with
exact clip duration without accounting for the trailing sample interval.

### AUD-05 — Standalone review GET routes lack hostname checks

Status: fixed; covered by `tests/test_audit_regressions.py`. Code: [review.py](../src/camera_system/review.py).

Unlike the main website, the standalone review server accepts arbitrary Host
headers on `/` and `/api/session`. A handler check returned session data and HTTP
200 for an untrusted hostname. This is a potential DNS-rebinding exposure; a real
browser exploit was not attempted.

Fix: apply the local hostname allowlist to GET requests. Acceptance: untrusted
Host values receive 403 without session data; legitimate localhost reads still work.

## Should fix

### AUD-06 — History polling parses every saved result repeatedly

Status: fixed; covered by `tests/test_audit_regressions.py`. Code: [app.py](../src/camera_system/app.py),
[workspace.html](../src/camera_system/templates/workspace.html).

Every three seconds, history polling scans saved results, including embedded
preview data. At audit time, 10 workspace result files totaled approximately
9.7 MB. Work increases with retained footage even when completed jobs never change.

Fix: persist final progress/coverage metadata and read active progress incrementally.
Acceptance: repeated polls do not reread completed result bodies; restart and partial
line handling preserve correct counts. Support older jobs with a one-time metadata fill.

## Verification and scope

Audit checks: 106 tests passed, four optional model tests skipped; Ruff and
dependency consistency passed. Focused probes demonstrated AUD-01 through AUD-05;
AUD-06 follows from the polling and snapshot paths and saved-file inventory.

Application modules, entry points, configuration and diagnostic scripts were
inspected; tests were inspected selectively. No application code was changed by
the audit. Fresh model/GPU runs, physical cameras, browser exploitability,
dependency vulnerability advisories and upstream model internals were not checked.

For each fix, add the smallest regression covering its stated failure, run the
relevant checks, then record the result here before marking the item resolved.

## Fix verification — 10 October 2026

The local implementation already contained the six fixes. Website relations run
on every sample, sparse file-video schedules are rejected, worker slot ownership
survives disconnects, image outputs use exclusive creation, evaluation uses an
explicit first-to-last sampled interval, review GET routes validate Host, and
progress is cached with incremental reads for active jobs.

This continuation also fixed progress refresh after a completed result is replaced
by a larger file: only active jobs use append-only reads. The existing test in
`tests/test_app.py` exposed this and now passes. The full lightweight suite passes
(125 tests). Evaluation intentionally excludes unsampled leading/trailing footage;
it does not certify full clip coverage. Live-camera scheduling still needs hardware
measurements.

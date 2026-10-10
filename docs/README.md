# Project overview

Status snapshot: 10 October 2026. Work-mode is a working local prototype, with
unmeasured real-world accuracy. Start with this page; dated verification files
record individual runs rather than the current feature list.

## Current state

| Area | Implemented | Still needed |
| --- | --- | --- |
| Detection and tracking | YOLO regions, persistent track IDs, optional motion matching | Labeled recall and identity-switch evaluation |
| Relationships | Pinned RelateAnything model and configurable vocabulary | Domain calibration and measured precision/recall |
| Event candidates | Dwell, distance trends, zones and unattended-bag reasoning | Independent event labels and false-alarm measurement |
| Person observations | Pose, approximate appearance and image movement | Domain validation; walking/running/falls are not implemented classifiers |
| Website | Local uploads, history, cancel/retry, evidence review and exports | Fresh browser verification |
| Plates and lookups | Optional OCR; selected official lookup integrations | Reviewed readings and provider-specific access where required |
| Evaluation | Event/relation scoring, annotation draft export | Independently reviewed ground truth |
| Publication | Verified source prepared for GitHub | See [publication verification](publication-2026-10-10.md) |

Current checks: **129 passed, including all four real-model integration tests**.
A fresh lightweight setup also passed 125 tests with four model tests skipped. Lint and dependency
checks passed. These verify execution, not recognition accuracy. The six audit
findings have regression coverage; see the [audit backlog](audit-backlog.md) and
[publication verification](publication-2026-10-10.md).

## Work order and completion criteria

| Order | Work | Complete when |
| --- | --- | --- |
| 1 | Reliability and security | Six audit regressions now covered; validate against representative workloads |
| 2 | Finish supplied-video review package | Matching review HTML and annotation draft exist for `user-27183-20261009`; website opens successfully |
| 3 | Establish the baseline | Independent annotations cover the evaluated footage; precision, recall, F1 and false alarms/hour are saved |
| 4 | Improve the weakest component | Error analysis identifies detection, tracking, relations or rules; improvements are measured on separate test footage |
| 5 | Optional THID experiment | Same labeled frames and comparable metrics demonstrate whether it helps; THID is researched only, not installed/integrated |
| 6 | Live-camera and release work | Target hardware meets agreed cadence, camera/recovery tests pass, storage/access arrangements are defined, and source/configuration changes are reviewed |

Inputs needed: priority event types, acceptable false-alarm/missed-event rates,
independently reviewed footage with quiet periods and difficult scenes, and target
camera/hardware details. Keep tuning and final evaluation clips separate. GPU
purchase is not a prerequisite for labeling or baseline evaluation; assess it
after profiling the chosen workload.

## Run and use the project

| Task | Guide |
| --- | --- |
| Install dependencies and run the first image | [Repository README](../README.md) |
| Start the local upload website | [Local workspace](local-workspace.md) |
| Analyze video, generate review or annotation drafts, evaluate | [Prototype guide](prototype-guide.md) |
| Understand components and original phases | [Architecture](architecture.md) |
| Review dwell, distance trends and event evidence | [Temporal behavior](temporal-behavior.md) |
| Review person observations | [Person analysis](person-analysis.md) |
| Read plates / look up vehicle details | [Plate reading](plate-reading.md), [vehicle lookups](vehicle-lookups.md) |
| Understand camera source references | [Camera sources](camera-sources.md) |
| Compare detector or tracker settings | [Detection comparison](detection-comparison.md), [tracking comparison](tracking-comparison.md) |
| Understand measured runtime | [Performance](performance.md) |
| Check attribution and dependencies | [Third-party notices](../THIRD_PARTY_NOTICES.md), [package configuration](../pyproject.toml) |

Run commands from `Work-mode/` using its `.venv`. The website entry point is
`python -m camera_system.app`, normally at `http://127.0.0.1:8780/`. Its most recent
opening attempt was blocked by approval-service limits; this document does not
assert that a server is currently running.

## Saved work and evidence

Paths below are relative to the repository. `outputs/` is ignored by Git, so these
artifacts are local and will not appear in a fresh clone.

| Artifact | Location | State |
| --- | --- | --- |
| Latest supplied-video analysis | `outputs/user-27183-20261009/session.jsonl` | Completed: 736 decoded frames, 50 analyzed samples |
| Reference image from that video | `outputs/user-27183-20261009/reference-12s.jpg` | Saved |
| Processing diagnostics | `outputs/user-27183-processing-20261009.log` | Saved |
| Latest video's review and annotation draft | `outputs/user-27183-20261009/` | Not generated yet |
| Earlier traffic annotation draft | `outputs/traffic-annotations-draft-20261009.json` | Unreviewed; different 19-frame traffic session |
| Website-managed jobs | `outputs/workspace/<job-id>/` | Upload, job metadata, results, logs and generated review |

The latest video took 63.4 seconds including startup: **0.79 analyzed FPS** overall,
or 1.27 FPS for measured per-frame processing alone. Its 720 detections are repeated
frame observations, not unique objects. There are 2,471 raw relations, zero temporal
confirmations and 33 unverified event candidates. No real accuracy score exists.

Keep each new analysis in its own directory under `outputs/`. Preserve the original
session and use new filenames for reviews, annotations, replays and metrics. Do not
move old artifacts merely to conform to this convention: scripts and records may
refer to their existing paths. Image-command output filenames must not already exist; exclusive creation preserves existing files.

## Historical verification

- [Latest continuation and annotation-export checks](continuation-2026-10-09.md)
- [Original prototype verification](prototype-verification.md)
- [Phase 3 video checks](phase3-video.md)
- [Phase 2 detector checks](phase2-verification.md)
- [Phase 1 checks](verification.md)
- Environment snapshots: `verified-environment-windows-py312.txt` and
  `verified-environment-phase2-windows-py312.txt`.
- Small phase fixtures: `phase1-sample-result.json`, `phase2-sample-result.json`
  and `phase3-sample-result.jsonl`.

## Repository boundaries

Application code stays in `src/camera_system/`, tests in `tests/`, settings in
`configs/`, and diagnostics in `scripts/`. Keep existing module paths stable.
Weights, caches, the virtual environment and generated outputs remain ignored.
The sibling `upstream-inspection/` is the upstream reference checkout. The parent
project's `sources/` directory is synced, read-only reference material.

When work changes, update this page's status and the corresponding audit item.
Record exact validation results in the relevant dated record; do not convert old
test results or synthetic scores into claims about current real-world accuracy.

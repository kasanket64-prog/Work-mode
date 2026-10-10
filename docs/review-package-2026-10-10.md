# Supplied-video review package — 10 October 2026

The next roadmap milestone is complete: the saved supplied-video analysis now
has a matching interactive review and an independently fillable annotation draft.
No inference rerun, threshold change or new dependency was necessary.

## Local artifacts

Under `outputs/user-27183-20261009/`:

- `session.jsonl`: original saved analysis, unchanged.
- `review.html`: standalone review with embedded previews and event evidence links.
- `annotations-draft.json`: 50 frame references, interval `[0.0, 24.5245]` seconds;
  status remains `draft`, duration and event labels remain unset.
- `review-browser.jpg`: browser verification screenshot.

These files remain local and ignored by Git. No footage, previews or annotations
are published to GitHub. The source session SHA256 before and after verification:
`a6f34c62a8fd4062b8a65dcc507040da498577df1d81a9c1dfb408e88ac57139`.

## Verification

- Review opened with 50 processed frames and 33 unreviewed candidate events.
- Next-frame navigation advanced to frame 15 at 0.501 seconds.
- Event 1 evidence selection navigated to frame 45 at 1.502 seconds.
- Setting a temporary Uncertain verdict updated the human-review count; Export
  JSON retained that verdict. The downloaded `camera-session.json` is a UI test
  artifact, not ground truth. The browser verdict was restored to Unreviewed.
- Draft checks confirmed matching frame count and interval, with no copied model
  event labels. The evaluator rejected the draft as unreviewed.
- The original session hash remained unchanged.
- The upload workspace opened and rendered existing history, coverage indicators
  and investigation links. No new upload was submitted during this smoke check.
- Neither browser page reported JavaScript console errors during the check.

## Next step: independently label the footage

Review the original footage across the exact sampled interval, including missed
and absent events. Complete duration, event annotations and any identity mapping
in a copy of the draft; set `annotation_status` to `reviewed` only when complete.
Candidate verdicts alone cannot measure recall. Use the evaluator instructions in
[the prototype guide](prototype-guide.md); keep tuning clips separate from final
held-out evaluation. No accuracy or false-alarm score is claimed by this package.

Physical/network cameras, GPU performance and live external lookups were not
retested. This milestone verifies the review workflow, not event correctness.

# Fieldnote website refresh — 11 October 2026

## Delivered

- Responsive ivory/green homepage with an interactive, explicitly saved real
  inference example. Its person/horse regions are manual; its relationship is
  predicted. Image credits, model attribution and original JSON remain available.
- Drag-and-drop or file-picker upload, size/type checks, optional settings,
  upload progress, truthful preparation/processing stages and clearer job cards.
- Existing cancellation, retry, download and annotation evaluation remain wired
  to the same backend. Polling preserves unchanged cards and their open controls.
- Updated review styling, collapsed secondary tools, and original-video playback
  with a jump to the selected sampled evidence frame. Loading another result
  removes the original player to avoid mixing recordings.
- Allowlisted packaged homepage assets and local-only original-media streaming
  with single byte ranges. Existing standalone HTML snapshots remain unchanged;
  the workspace serves the latest review template against saved result data.

## Verification

- Full lightweight suite: **136 passed, 4 optional model tests skipped**.
- Final focused website suite after range/error handling refinement: **3 passed**.
- Ruff, JavaScript syntax, dependency consistency and diff whitespace checks passed.
- Wheel built successfully; all five homepage/demo resources verified in archive.
- Browser: real demo selection and box visibility toggles worked; desktop and
  mobile layouts inspected. At 390px viewport there was no horizontal overflow.
- Real browser upload of a generated three-second WebM using the credited horse
  photograph completed through the existing detector, relationship and pose
  pipeline. 30 frames decoded, two sampled frames analyzed, 85.96 seconds total.
  This is a repeated-image execution fixture, not an accuracy or motion benchmark.
- Browser review loaded, video decoded (readyState 4, duration 3 seconds), and
  selected evidence sought to 1.5 seconds. No browser console errors observed.
- Test artifacts remain ignored under `outputs/website-preview-20261011/` and
  `outputs/website-wheel-20261011/`. Test servers were stopped afterward.

The browser upload permission review initially timed out without a decision;
inspection confirmed the file selection had completed, and submission then
succeeded. No security controls were bypassed.

## Limits and next design pass

No public deployment, external upload access or model behavior changes. The
finished local app runs at port 8780; restart via `scripts/start-workspace.ps1`
when needed. Browser video codec support varies; unsupported media retains its
sampled evidence view. Real-world accuracy and public hosting remain unfinished.

The user plans to explore a Figma design next. Preserve this functional baseline;
integrate approved desktop/mobile designs without replacing the inference,
upload, cancellation, review or evaluation contracts.


## Homepage usability follow-up

Added an original split hero using the credited demo photograph, illustrated feature cards, clearer benefit copy, workflow strip and FAQ. Uploads now have local image/video previews, removal, and 30/120-frame presets synchronized with the existing sample limit. Workspace cards can be searched by type/date/ID and filtered by status without discarding review inputs. No dependencies or inference behavior changed.

Validation: 137 pytest tests passed, 4 optional tests skipped; Ruff and JavaScript syntax check passed. Browser checks confirmed 120-frame preset, search empty state, active-status filter, loaded image preview and removal. Desktop 1440x1000 and mobile 390x844 inspected; no mobile horizontal overflow. Screenshot: outputs/fieldnote-polished-homepage.jpg (local only). Existing real-inference validation above remains applicable; no new accuracy benchmark or inference run was performed in this UI follow-up. Public hosting remains outstanding.

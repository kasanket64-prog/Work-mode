# Third-party notices

Based on RelateAnything by Maëlic Neau — https://github.com/Maelic/RelateAnything

RelateAnything is installed as an external dependency, pinned to commit `06766fdf56752ca535fc9b971fca99ce563676d0`. Its code carries AGPL-3.0-only plus the attribution/name terms in its NOTICE. Preserve the attribution above in future interactive interfaces. This repository does not relabel that dependency under a permissive license. Model weights and text student have separate DINOv3-derived terms.

Authoritative files at the pinned revision:

- https://github.com/Maelic/RelateAnything/blob/06766fdf56752ca535fc9b971fca99ce563676d0/LICENSE
- https://github.com/Maelic/RelateAnything/blob/06766fdf56752ca535fc9b971fca99ce563676d0/NOTICE
- https://github.com/Maelic/RelateAnything/blob/06766fdf56752ca535fc9b971fca99ce563676d0/THIRD_PARTY_NOTICES.md
- https://huggingface.co/maelic/relsgg-vits16plus/blob/2db90096be5217bdc7a9003c042950f45723d105/README.md

## Included test photograph

`examples/horse.jpg`: **A horse rider in Maidan , Kolkata.jpg**, by **Subhrajyoti07**, licensed **CC BY-SA 4.0**.

- Source: https://commons.wikimedia.org/wiki/File:A_horse_rider_in_Maidan_,_Kolkata.jpg
- License: https://creativecommons.org/licenses/by-sa/4.0/
- Retrieved unchanged from upstream `assets/reel/images/horse.jpg` at the pinned commit. Upstream distributed dimensions: 1280 x 907; no additional modifications here.
- SHA256: `549bd965a4ac9aa682f70e40757c5841d160c4e84eb86c2948d5bac5dedecf29`
- Per-image upstream credit: `assets/reel/credits.json` at that commit.

The example's person/horse coordinates follow upstream's quickstart. No training dataset, detector weights, or model weights are committed here.

## Phase 2 detector

The optional `detection` extra installs Ultralytics 8.4.173. Its code and YOLO model distribution carry their own licensing terms (AGPL-3.0 / commercial licensing); no detector code or weights are redistributed in this repository. Preserve applicable notices when deploying a combined application.

- https://github.com/ultralytics/ultralytics/blob/v8.4.173/LICENSE
- https://www.ultralytics.com/license
- Weights used for the baseline: https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt

The detector weight hash actually used is recorded in each automatic inference result.

The optional person analyzer also loads official `yolo11n-pose.pt` from the Ultralytics v8.4.0 assets release, under the same upstream licensing framework. The pose weight SHA256 is recorded in results. Appearance and posture heuristics in this project are separately described in `docs/person-analysis.md` and are not claimed as upstream capabilities.

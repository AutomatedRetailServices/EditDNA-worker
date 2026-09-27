# Yaskira 01–10 saved-result inventory — 2026-09-27

Read-only assessment; no new provider call, render or paid run. Baseline artifacts: runs 36321269066 and 36324899917. These are saved historical results, NOT qualification of current V17/V18 code. Selected duration below is timeline duration, not measured MP4 duration.

| Source | Raw seconds | Selected seconds | Technical QC | Automated review | Human reference |
| --- | ---: | ---: | --- | --- | --- |
| Yaskira/01.mp4 | 115.561 | 103.504 | PASS | NOT_DELIVERABLE_WATCH_LISTEN_BLOCKED:perceptual=FAIL | Not recovered for this exact source; do not invent approval |
| Yaskira/02.mp4 | 99.262 | 93.798 | PASS | NOT_DELIVERABLE_WATCH_LISTEN_BLOCKED:perceptual=FAIL | Not recovered for this exact source; do not invent approval |
| Yaskira/03.mp4 | 19.623 | 17.9 | PASS | DELIVERABLE_PENDING_HUMAN_WATCH_LISTEN:watch_listen=HUMAN_REVIEW_REQUIRED | Not recovered for this exact source; do not invent approval |
| Yaskira/04.MP4 | 54.025 | 33.632 | PASS | DELIVERABLE_PENDING_HUMAN_WATCH_LISTEN:watch_listen=HUMAN_REVIEW_REQUIRED | Not recovered for this exact source; do not invent approval |
| Yaskira/05.MP4 | 139.965 | 37.949 | PASS | DELIVERABLE_PENDING_HUMAN_WATCH_LISTEN:watch_listen=HUMAN_REVIEW_REQUIRED | Existing approved reference |
| Yaskira/06.MP4 | 112.002 | 22.426 | PASS | NOT_DELIVERABLE_WATCH_LISTEN_BLOCKED:perceptual=FAIL | Not recovered for this exact source; do not invent approval |
| Yaskira/07.mp4 | 366.997 | 159.891 | PASS | NOT_DELIVERABLE_WATCH_LISTEN_BLOCKED:perceptual=FAIL | Not recovered for this exact source; do not invent approval |
| Yaskira/08.mp4 | 149.533 | 52.0 | PASS | DELIVERABLE_PENDING_HUMAN_WATCH_LISTEN:watch_listen=HUMAN_REVIEW_REQUIRED | Not recovered for this exact source; do not invent approval |
| Yaskira/09.mp4 | 123.812 | 44.136 | PASS | DELIVERABLE_PENDING_HUMAN_WATCH_LISTEN:watch_listen=HUMAN_REVIEW_REQUIRED | Existing approved reference |
| Yaskira/10.MOV | 97.972 | 51.293 | PASS | DELIVERABLE_PENDING_HUMAN_WATCH_LISTEN:watch_listen=HUMAN_REVIEW_REQUIRED | Not recovered for this exact source; do not invent approval |

## Findings and next review
- All ten saved baseline renders passed technical QC. No automatic editorial acceptance follows.
- 01,02,06,07 blocked on PERCEPTUAL_RESET_DEBRIS_AT_EDGE. These are mapped source motion candidates, not direct decoded-output confirmation. Visual review must distinguish actual recording resets from intentional gestures before changing the gate.
- 03,04,05,08,09,10 require human review in these baseline artifacts. Later 05/09 runs exist and must not be overwritten with baseline status.
- 09: use the later V17/deferral checkpoint in CUTSELL_EDITORIAL_DEFECT_DEFERRAL.md. No new isolated paid run.
- 05: existing approved run31 reference remains separate from this batch's 37.949-second selection. Do not call this baseline identical to the approved 33.234-second output.
- Historical claims mapping 07 to Video00 or10 to MOV require source identity verification before importing their gold intervals.
- Next human review: Yaskira/01.mp4. Selected 103.504s from115.561s. Source31.918–36.760 was discarded: “I Haven't worked with a cameraman ever in my life. So I used to work for years and”. The following selected explanation starts37.880. Ask whether this is intentional false-start removal or loss of useful setup; transcript alone cannot establish performance intent.
- Saved preview: yaskira-eval-main-36321269066/previews/01-01.mp4. Times above refer to RAW source, not preview timeline.
- No defect fix cycle consumed by this read-only inventory. Existing two-cycle-per-class limit applies to future corrections.


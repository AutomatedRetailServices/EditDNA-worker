# Yaskira08 V22 — technical render with editorial and QC failures

Branch commit `865ffcb4`, run `36424286975`, artifact `10971382348`. Qualification tests passed, and the source-word union passed Freeze. The rendered edit is **not deliverable**: post-render QC detected 1.397 seconds of accidental silence, output 66.691–68.088, mapped to source 117.128–118.525.

| Measure | V20 | V22 |
|---|---:|---:|
| Gold 120–147 retained | 26.52 s | 23.245 s |
| Gold lost | 0.48 s | 3.755 s |
| Outside Gold retained | 95.30 s | 71.209 s |
| Recall / precision | 98.22% / 21.77% | 86.09% / 24.61% |

V22's structured competition removed three older fragments against a complete delivery, improving the extra-duration metric, but still retained substantial early material. The model selected 120.33–141.39 and 145.095–146.95, while labeling the short phrase at 142.93–144.77 (“levantada de cola”) as failed. It grammatically completes the selected sentence ending “para darte”; the missing interval includes a pause and should be visually reviewed against Gold's rounded 120–147 label.

The silence reappeared because V2 continuity restoration extended a selected clip across 116.57–120.33 on a broad AV “audience demonstration” label, despite source-measured silence events 116.386–120.731 and 117.128–119.584. V23 must let localized visual demonstration evidence preserve legitimate silent actions, while rejecting broad labels when measured dead air contradicts them. The missing short phrase remains an unresolved editorial decision. An attempted grammar-only rescue failed independent QA because it could restore a failed take; it was reverted. The silence fix requires independent QA and a new live render before acceptance.

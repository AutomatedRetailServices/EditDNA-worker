# Yaskira08 V23 — bounded success, further regression pending

Branch commit `c1a855ba`, GitHub Actions run `36425985553`, attempt 2, artifact `10972131823`. Attempt 1 failed before Selection because Watch + Listen received HTTP 503; the bounded rerun succeeded. The same 134-workflow-test suite passed before the live edit.

| Measure | V20 | V22 | V23 attempt 2 |
|---|---:|---:|---:|
| Gold 120–147 retained | 26.52 s | 23.245 s | 25.199 s |
| Gold lost | 0.48 s | 3.755 s | 1.801 s |
| Outside Gold retained | 95.30 s | 71.209 s | **0.000 s** |
| Recall / precision | 98.22% / 21.77% | 86.09% / 24.61% | 93.33% / 100% |
| Delivery | Pending human review | Blocked by silence | Pending human watch/listen |

Selected source intervals: 120.39–141.388, 142.709–145.1, 145.1–146.91. The missing 1.801 seconds of the rounded Gold are edge margins (120–120.39 and 146.91–147) and the 141.388–142.709 gap. Source silencedetect found 141.268–142.829 at the relaxed floor and 141.922–142.783 at the strict floor, so the internal gap is measured silence rather than a known lost spoken phrase. The model kept the short completion “una levantadita de cola” and CTA. One structured take comparison declared the later complete delivery equivalent to nine preceding candidates, and the final Selection contains no earlier intervals.

Render length 24.367 seconds; an eight-frame visual contact sheet shows the creator speaking and demonstrating front/back of the jumpsuit without a visible wrong scene. Automated QC reports zero findings and `DELIVERABLE_PENDING_HUMAN_WATCH_LISTEN`; a human watch/listen remains required before editorial acceptance. The selected source duration of 25.199 seconds differs from render duration because physical Boundary removes pauses.

This is one successful stochastic live case, not proof of repeatability or the ten-video gate. Next: rerun Yaskira08 alongside Yaskira09 on the identical engine commit/config to check Selection stability and whether the measured-silence correction preserves Video09's silent product demonstration. If those pass, run the unchanged 01–10 batch and compare each against its own Gold.

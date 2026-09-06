# Video00 reference oracles

## RAW source
`Editdna longform validation/VIDEO-2026-07-30-09-18-03.mp4`

## Human Gold oracle
`Editdna longform validation/5E01F214-A364-4F4B-8F25-D39B1E2B21D2.MP4`

Purpose: authoritative editorial target for Selection parity, composite construction, retry resolution, narrative sufficiency, and final Human Watch+Listen review.

## Cut.ai commercial baseline
`Editdna longform validation/D40F1D43-7391-44D5-8D83-09CB62FBF397.MP4`

Purpose: commercial baseline for recording-process cleanup, obvious retry removal, take quality, basic continuity, and visual cleanliness. Cut.ai is not Gold; Human Gold supersedes it when they disagree editorially.

## Comparison hierarchy
1. CutSell worse than Cut.ai = Level 1 regression; fix first.
2. CutSell ~= Cut.ai but worse than Human Gold = Level 2 editorial gap.
3. CutSell ~= or better than Human Gold = protect; do not regress.

Both reference videos are QA-only. Never provide either video, transcript alignment, or oracle decision directly to production Selection/Boundary code.

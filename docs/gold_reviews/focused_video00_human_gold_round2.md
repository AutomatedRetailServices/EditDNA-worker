# Focused Gold Review — Video 00 — Human Round 2 / Round 4 Rejection

Source: `Editdna longform validation/VIDEO-2026-07-30-09-18-03.mp4`

This review is authoritative human editorial Gold for Clean Cut. Timestamps below refer to the focused preview timeline reviewed by the user.

## Core editorial rules

1. **Sentence-complete audiovisual boundary cleanup**
   - Once a spoken thought is complete, trailing dead air, facial reset/mueca, body reset, camera disengagement, or unrelated movement should be removed.
   - This is an audiovisual rule, not an audio-only silence rule.
   - Do not preserve post-sentence visual garbage merely because no spoken word is being cut.

2. **Broken attempt loses to clean retake**
   - A take that starts well but fumbles/repeats later is still the losing attempt if a nearby clean retake delivers the same idea completely.
   - Compare complete communication attempts, not just a good prefix.

3. **Best Take must resolve final competing deliveries**
   - Keep the cleaner complete ending and remove the later broken/repeated version.

## Human-marked cuts / winners

- Around `0:12`: remove dead air / visual reset immediately after sentence completion.
- `0:14–0:22`: starts well, but fumbles around `0:21` and begins repeating the same idea.
  - **LOSE / DELETE** this broken attempt.
  - `0:23–0:32` is the clean retake and should **WIN / KEEP**.
- `0:33–0:34`: cut dead air / visual reset.
- `0:42–0:43`: cut dead air / visual reset.
- `1:03–1:04`: cut dead air / visual reset.
- `1:04–1:07`: obvious fumble/repetition before clean retake.
  - **DELETE 100%**.
  - Retake beginning around `1:08` should be kept.
- `1:17–1:19`: sentence has ended; remove visual reset/dead air.
- `1:46–1:47`: same audiovisual boundary problem; cut.
- `1:49–2:00`: remove the unwanted section according to human editorial review.
- `2:05–2:07`: cut dead air / visual error after completed thought.
- `2:16–2:17`: cut dead air / visual reset.
- Final competing delivery:
  - `2:25–2:43` = **WINNER / KEEP**.
  - `2:45–3:01` = **LOSER / DELETE**; it contains a fumble near `2:59` and ends abruptly.

## Round 4 human verdict — REJECTED

Round 4 (`GitHub Actions run 32542084376`) was technically successful but did **not** pass human editorial review. The same core Gold remains authoritative.

The real Round 4 report exposed the exact integration failure behind the repeated sonography error:

- source `108.56–111.86` remained selected: `Ahí fue cuando me mandaron a hacer sonografías de tiroides`;
- clean retake source `120.11–124.15` was discarded: `a hacer sonografías de tiroides y otras sonografías`;
- the clean retake was removed by `semantic_short_alternate_covered_by_neighbors`;
- overlapping Hybrid windows had judged that clean retake `alternate=0.75` in one window and `keep=0.80` in another;
- the Gold reconciliation threshold of `0.80` therefore failed to run against the reducer's retained `alternate=0.75` state even though retry setup was independently confirmed around source `112.01` at confidence `0.86`.

Round 4 also exposed a separate availability issue near the end of Video 00: the last Hybrid window was unavailable because the per-edit paid inference ledger was exhausted before the test budget ceiling. Focused benchmark runs must allow the explicitly approved test budget without changing the lower production per-edit COGS default.

Finally, the renderer still allowed long objectively silent post-roll to survive when the trailing silence exceeded the former 3-second safety ceiling. Human Gold requires those proven long dead-air tails to be removed rather than rejected merely because they are longer.

## Regression intent

A successful future preview should visibly demonstrate all three behaviors:

- frame-tight audiovisual sentence boundaries, including long proven post-roll dead air;
- failed attempt vs clean retake resolved at the attempt level, including the exact `108.56–111.86` vs `120.11–124.15` inversion;
- one clean final ending selected, with the broken alternate removed.

CI or benchmark completion alone is not editorial approval. Only the reviewed preview can pass this Gold.

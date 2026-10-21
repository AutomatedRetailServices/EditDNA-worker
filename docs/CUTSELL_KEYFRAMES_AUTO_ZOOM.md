# Keyframes (like CapCut) and Auto Zoom AI — 21 Oct 2026

Branch `feat/keyframes-smooth` (server + engine) and `feat/ios-build24` (iPhone). Not deployed until the
Product Owner says so (rollback for server/engine: `a0addc8`).

## Keyframes
- Draft field `keyframes` (cutsell_worker/keyframes.py). Each keyframe has `curve`:
  `ease_in_out` (default, not written), `linear`, `ease_in`, `ease_out` — how the value moves from it to the next one.
- **One track for the whole video.** Video keyframes sit where their source moment plays on the edited timeline
  (`timeline_track`); each clip draws its own piece of that track (`clip_track`: the keyframes inside it plus the
  nearest one on each side, re-mapped to its source seconds). The picture glides across joins instead of
  snapping (the old per-clip interpolation reset at every join). Held before the first and after the last.
- iPhone mirrors this exactly (`EditorB17Model.videoTrack`, `KeyframeCurve`). UI: ◇ adds/removes at the white
  line; once the target has a ◇, a pinch / drag / twist anywhere else makes a ◇ there; ‹ › jump between ◇;
  tapping a ◇ on the timeline shows the curve bar; the white line stays in the middle and the tracks slide under it.

## Auto Zoom AI
- Moments come from the existing word-fix call (no extra call): hook, product, price, cta, **look**, **benefit**;
  at most one every 3 s; a video with a product almost always gets one.
- Each zoom = 4 keyframes: in 0.4 s (ease in-out), hold, out 0.4 s; 110–125 % by kind; centred.
- A moment near the end of a kept part starts its zoom earlier inside the part (was dropped).
- `diagnostics.simple_engine.sources[].zoom_moments`: word, kind, and `added` / `too_close` / `part_too_short` / `cut_out`.
- A new cut of a draft that already has `keyframes` replaces the AI keyframes with the new zooms
  (`draft_merge._refresh_ai_keyframes`); the creator's keyframes stay; a deleted AI zoom (`zoom_removed`) never returns.
  The app adopts AI zooms it does not carry yet even when keyframes exist.

## Evidence
- `tests/test_cutsell_keyframes_smooth.py` (track across a join, ffmpeg expression = value_at, curves, merge).
- `tests/test_cutsell_build17_end_to_end.py`: real renders — zoom grows through a join with no jump; 100→130→100 %.
- Her 29 drafts (read-only probe): the "0 from Auto Zoom AI" cuts had no AI zooms from the engine (empty moment list
  allowed, or cut before Auto Zoom existed). New call on 8 of her cuts: 1–5 zooms each.

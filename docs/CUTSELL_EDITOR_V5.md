# CutSell Editor v5: state, draft fields, deploy

Branches:
- App: `feat/figma-part1`.
- Server: `feat/editor-v4-mirror-export`, at 183df94, waiting for deploy approval.
- Live: Render and Modal run 6d5efb9.

No merge to `main`. No audio, video or keys in the repository.

## One model for every clip

Video, voice-over, music, CutSell sounds and overlays all live on the **draft**, so each change is one save (anti-409 queue) and the draft's own Undo / Redo covers it. The export renders the same draft.

Voice-overs placed before v5 (D-282A timeline) join the draft once, as `audio_clips`, the first time the editor opens.

### Draft fields (app ↔ API ↔ export)

The Product Owner's names (15 Oct 2026) map to the stored names below.

| Product Owner name | Stored field | Values |
|---|---|---|
| clip `speed` | `selected[].speed` | 0.5 · 1 · 1.5 · 2 · 3 (absent = 1) |
| clip `volume` | `selected[].audio_volume` | 0–2 |
| clip `muted` | `selected[].audio_muted` | bool |
| clip `mirror` | `selected[].mirror` | bool |
| clip `rotation` | `selected[].rotation` (+ draft `rotation` for the whole video) | 0 · 90 · 180 · 270 |
| clip `source_in` / `source_out` | `selected[].start` / `end` (seconds of the original) | |
| `master_volume` | `video_volume` | 0–2 (absent = 1) |
| `clean_up_voice` | `video_clean_voice` | bool |
| voiceover `start` / `in` / `out` | `audio_clips[].start` / `source_in` / `source_out` (`kind: "voice"`, `asset_id`) | seconds |
| voiceover `volume` | `audio_clips[].volume` | 0–2 |
| voiceover `clean_up` | `audio_clips[].clean` | bool |
| voiceover `duck_video` | `audio_clips[].duck_video` | bool |
| voiceover `enhanced_at_record` | `audio_clips[].enhanced_at_record` | bool |
| overlay `layer` | `media_overlays[].layer` | 0–50, higher on top |
| overlay `type` | `media_overlays[].kind` | photo · video |
| overlay `asset_id` | `media_overlays[].uri` (tenant-checked at export) | |
| overlay `start` / `end` | `media_overlays[].start` / `end` | final timeline seconds |
| overlay `in` / `out` | `media_overlays[].source_start` / `source_end` | |
| overlay `layout` | `media_overlays[].layout` | float · side_by_side · stacked · left · right · top · bottom · full |
| overlay `rect` | `x`, `y` (centre, 0–1), `width` (0.1–1; height from the media's shape) | |
| overlay `rotation` | `media_overlays[].rotation` | −360…360 (Float layout) |
| overlay `edge` | `media_overlays[].edge` | sharp · soft (12 px) · extra_soft (28 px) |
| overlay `muted` / `volume` | `mute_audio` / `volume` | bool / 0–2 |
| overlay `fill` | `media_overlays[].fill` | bool |
| SFX `sfx_id` / `start` / `in` / `out` / `volume` | `audio_clips[]` with `kind: "sfx"`, `sfx_id`, `start`, `source_in`, `source_out`, `volume` | |
| music | `audio_clips[]` with `kind: "music"`, `asset_id` | |
| `zoom_keyframes` | `zoom_keyframes` (+ `zoom_removed` tombstones of deleted AI zooms) | |

## Export (server, 183df94)

| Option | How it is rendered |
|---|---|
| Speed | `setpts` on the video, chained `atempo` on the audio (pitch kept). Applied after captions, zoom and orientation, so captions stay on their words. Output windows are divided by speed. |
| Volume / Mute | Each clip's `audio_volume`, then `video_volume` on the whole video sound. |
| Clean up voice | afftdn + 80 Hz highpass + loudnorm −16 LUFS. On the video sound when `video_clean_voice` is on, and on a clip's own window when it has `clean`. |
| Lower video sound | −12 dB with 120 ms ramps while each ducking clip plays. |
| Mirror / rotation | Base video only, before captions, overlays and logo. |
| Overlays | Drawn in layer order, with layout regions, fit or Fill, soft edges (looped blurred mask), rotation, and the overlay's own sound when not muted. |
| CutSell sounds | Fetched from `assets/sfx/v1/` in the media bucket (Amazon S3, `script2clipshop-video-automatedretailservices`) and mixed at their time and volume. |
| Voices / music | Resolved through the tenant-safe timeline asset registry. A missing one fails the export with `audio_clip_unavailable` (never dropped silently). |
| Zoom keyframes | Smooth push-in and release, as before. AI zooms near a tombstone are not drawn. |
| Overlap audio | Per the draft flag, as before. Not applied to sped clips. |
| QC / delivery review | What the creator placed on purpose is not treated as an accident: B-roll, a hand-dragged tail, a sped clip, a muted clip with speech, or a video at volume 0. |

**Reprocessing:**
- A draft the creator has edited keeps its cut and every editor setting. Only the AI zoom keyframes (minus tombstones) and the diagnostics are refreshed, and nothing is written when they did not change.
- A draft that was never edited takes the new cut.

## Deploy (needs the Product Owner's approval)

1. **Modal engine:** push 183df94 to `feat/simple-engine-v1`. The deploy workflow runs its tests, then `modal deploy`.
   - Rollback: rerun the deploy workflow from 6d5efb9.
2. **Render `cutsell-api`:** manual deploy of 183df94.
   - Rollback: Render deploy dep-db80pa8u01pc738flu30 (6d5efb9).

## CutSell sounds

The sounds are generated from the two CC0 packs with `tools/make_cutsell_sounds.py`; see `docs/LICENCIAS_sonidos.md`. The `cutsell-sfx-upload.yml` workflow uploads them to `assets/sfx/v1/`. It runs only after approval, and only to the production bucket.

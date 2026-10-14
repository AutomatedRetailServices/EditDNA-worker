# CutSell iOS build 15 — state, measurements, pending deploys

Branches:
- App: `feat/figma-part1`
- Server: `feat/editor-v4-mirror-export`, at 6d5efb9. It is a straight continuation of the deployed `feat/simple-engine-v1` (c89d338), 8 commits ahead.
- No merge to `main`.
- Nothing deployed without the Product Owner's approval.

## Fixed in the app (build 15)

| # | Report (build 14) | Fix |
|---|---|---|
| 1 | Double header ("‹ Cut 14 Oct") | The editor hides the system bar (`DraftEditorView`). Processing no longer forces it visible. |
| 2 | Overlay | Overlay opens the phone's photos and videos straight away; the "Add to the video" menu and text are gone (text belongs to Captions). The picked file is copied to the phone first and shows on the video at once. It uploads with a %, and a failure leaves "Tap to retry". The real picture or video frame is shown at its real shape. One finger moves it, two fingers resize it. The overlay row shows real thumbnails, with trim from the edges, hold-and-drag to move, and Delete. |
| 3 | B-roll | The video row "+" opens the gallery (videos only). The video goes into the main track at the cursor, splitting the clip under it. It is kept in the draft's `broll_sources`. The old B-roll sheet is no longer reachable. Technical messages are mapped to plain words. |
| 4 | Drag and drop | Hold a clip to lift it and drop it elsewhere; the others make room. Voice-overs and overlays are held and dragged sideways, with soft magnets (cursor, clip joins, other items' edges) and a light haptic. One save on release. Undo and Redo cover voice moves too (one shared history). |
| 5 | Frames vanish | A frame request cut short no longer marks the video as broken. Frames are kept and retried. |
| 6 | Export | Starts on tap and waits for pending edits and uploads first. Shows the server's real %, with "You can leave this screen". An app-wide tracker sends a banner or notification when the export is ready. Exporting again with nothing changed returns the last video at once. |
| — | Processing | Continuous angular-gradient ring, 4.5 pt, one turn every 2.5 s, glow behind it. Reduce Motion: still ring with a breathing glow. The same ring is used on the logo loading screens. The cut name is shown, never the file name. |

## Export timings (live server, before)

Measured by `.github/workflows/cutsell-export-timing.yml`. Seconds from tapping Export.

| Cut | Submit | Queue + start | Sources | Render + QC | Deliver | Total | MP4 to phone |
|---|---|---|---|---|---|---|---|
| 13 s cold | 2.2 | 6.2 | 2.2 | 4.2 | 6.9 | 21.7 | 0.8 (6.6 MB) |
| 13 s warm | 2.2 | 5.8 | 2.3 | 4.5 | 5.1 | 20.0 | 0.9 |
| 60 s cold | 2.2 | 5.9 | 4.8 | 8.5 | 6.7 | 28.1 | 1.3 (28.8 MB) |
| 60 s warm | 2.2 | 6.2 | 2.5 | 11.2 | 5.1 | 27.2 | 1.4 |

## Server changes waiting for deploy

All on `feat/editor-v4-mirror-export`:

- **Submit:** the worker is woken in the background, about −2 s per export.
- **Silence QC:** decodes audio only. 15–18× faster per check; it runs twice per export (QC and review).
- **Probes:** each unchanged file is probed once.
- **Render + QC, local benchmark** (2 vCPU, same code before and after): 13 s cut 17.0 → 14.4 s; 60 s cut 67.3 → 60.3 s.
- **Progress:** the % moves 35 → 84 during render and QC, and per-stage `timings` come back in the result.
- **Signed links:** overlay photos and videos and B-roll sources get signed links when a draft is read, so any phone shows them.
- **Export QC:** media the creator placed by hand is not trimmed as accidental silence or a frozen or black picture. On the live server, a silent or still B-roll makes the export fail after 3 repair attempts. Reproduced locally, and fixed with tests.
- **Already on the branch, never deployed:** Auto Zoom AI, Overlap audio, Mirror/Rotate, Mirror per clip, trim by the edges (`manual_end`).

## Auto Zoom AI: where the chain breaks

- The app sends `auto_zoom_enabled`.
- The live API (`feat/simple-engine-v1` c89d338, `FlowBSubmitRequest`) has no such field, so the flag is dropped silently.
- The Modal engine deployed from c89d338 has no zoom code. So the draft never carries `zoom_keyframes`, nothing shows as ◇, and the export does nothing.
- Overlap audio is also not active in the live engine: there are no pre/post rolls in c89d338.

## Deploys to approve

1. **Modal engine:** fast-forward `feat/simple-engine-v1` c89d338 → 6d5efb9. The push starts `cutsell-modal-worker-deploy.yml`, which runs tests, then `modal deploy`.
   - Rollback: run the same workflow from c89d338, or `modal app rollback`.
2. **Render API (`cutsell-api`):** manual deploy of 6d5efb9. Auto-deploy is off.
   - Rollback: in Render, roll back to the previous deploy.

Deploy the engine first or together with the API: the API turns on `auto_zoom_enabled` by default.

# CutSell state, 2026-10-16: build 18

Continues `CutSell_estado_2026-10-16.md`. Details: `docs/CUTSELL_BUILD18.md`.

## Summary

- **iPhone app:** build 18 (TestFlight), branch `feat/figma-part1`.
- **Server:** unchanged (db3f61d). Build 18 needs no deploy.
- **No merge to `main`.**
- **New Simulator probe:** builds the app optimized like TestFlight and reads what the preview really mixes and what the player plays (an audio tap per sound).

## Checklist for Swanny's iPhone

Do it twice: first with the silent switch ON (orange visible), then OFF. Raise the iPhone's volume.

1. **Typing at 0:02.**
   - Open a cut. Move the cursor to 0:02.
   - Tap Audio → Typing → "Add at playhead".
   - Go back to 0:00 and tap Play: Typing must be heard from second 2.
2. **Good.** Add Good at 0:05 and play from 0:03: the "ding" is heard at 5 s.
3. **Move a sound.**
   - Tap the Typing block (white border).
   - Drag it with your finger (no holding) to 0:04.
   - Play from 0:03: it is heard at 4 s.
4. **Split a sound.** With Typing selected, put the cursor in its middle and tap Split. Play: both pieces are heard.
5. **Mute one clip.**
   - Tap the first video clip → Volume. It says "Selected clip". Tap Mute.
   - Play: the first clip is silent, the second one plays.
   - Close and reopen the cut: the first clip is still silent.
6. **Video row speaker.** Tap it 6 times, playing after each tap:
   - Off: the video is silent, but Typing and Good still play.
   - On: everything plays.
7. **Zoom with two fingers.**
   - Tap a clip, put the cursor at 0:01.
   - Pinch on the video (it zooms) and tap ◇: it stays filled.
   - Move to 0:03, turn two fingers on the video (it rotates) and tap ◇.
   - Play from 0:00: the zoom and the turn change smoothly.
8. **Auto Zoom AI.** Play a part with AI ◇ on the timeline: the video zooms in (the "◇ Zoom %" badge shows).
9. **Photo overlay.**
   - Overlay → Photo: the full gallery opens (all photos, albums, search).
   - Pick a photo that lives only in iCloud: "Getting it from iCloud… %".
   - On the timeline, with the overlay selected:
     - drag it (no holding);
     - stretch and shorten it from both edges;
     - Split it with the cursor inside;
     - delete one piece.
   - With two fingers on the video: make it bigger or smaller and rotate it.
10. **Video overlay.** Same as 9 with Overlay → Video.
11. **Diagnostics.**
    - You → tap "CutSell 0.1.0 (18)" five times quickly → Diagnostics.
    - Check that Audio says Category: Playback.
    - Each sound must say "file on iPhone · in mix · heard while playing -XX dB".
    - Tap "Send report" and send it to me.

If something fails, open Diagnostics right then and send the report: it shows whether the sound file is on the iPhone, whether it is in the mix, and how loud it really played.

# Clon (beta) — was "Split clone" (iOS) — branch `feature/split-clone`

**29 Oct 2026 (PO): Split is now "Clon" (BETA) and joins with the SILHOUETTE method.**
- Camera rail: `person.2.fill` + "Clon" + a small BETA tag (#1769FF). Same id `cameraSplit`.
- Start screen (white): "Clon" + BETA, "Record yourself twice. We put both of you in the same room.", 5 tips,
  *Start part 1*. Labels "Stand on the LEFT" / "Now stand on the RIGHT".
- Join (`SplitCompositor`, blend `.silhouette` by default): the WHOLE of part 2 as the background and only the
  person of part 1 on top (`VNGeneratePersonSegmentationRequest`, `.accurate`, switching to `.balanced` when a
  frame takes > ~0.12 s), soft 3–6 px edge, the shift measured between the two RECORDED parts (live check as a
  backup), part 1 matched to part 2 (per-channel gains from the average colour, ±18 %), length = part 2, the
  last frame of a shorter part 1 held, both sounds even. Rendered in sRGB with Rec. 709 tags.
- Fallback `.halves` (soft ~60 px middle) when the cut-out fails (Vision fails on ≥ 20 % of frames or finds
  no person on ≥ 50 %), takes > 240 s, or the phone moved > 8 % — with "We used a simpler blend for this video".
- "Creating your clone…" with the BrandRing; result: preview, *Retake*, *Use this video* (same flow as a Record
  video; the engine is unchanged). Diagnostics: `clone_joined` (method, seconds, video length, shift).
- Self-test (`split_selftest`, synthetic videos made in the test): silhouette over part 2 (also past the
  middle), part 2's person and room kept, no line, colour match, soft edge, length, sound, fallback.
  Speed: the simulator has no Neural Engine (~0.8–1.6 s per frame), so the real speed is read on the iPhone.
- TestFlight build 64 = `build/clon-and-style` (3D style + big icon + Clon). No merge to `main`, no deploy.

---

# Split clone (iOS) — branch `feature/split-clone`

PO request 27 Oct 2026. Two "yous" in the same room from one fixed shot, joined **on the iPhone**.
The engine is not touched; the joined video goes through the same flow as any Record video
(style sheet → upload → engine). No merge to `main`, no deploy, no new paid services.

## Entry and flow
- Record camera (Create tab only, `CreateCameraView(allowsSplit: true)`): rail control **Split**
  (`cameraSplit`). The Mix camera is unchanged.
- `SplitCloneView` steps:
  1. **Split screen** (white): only Left / Right, 5 tips, *Start part 1*.
  2. **Part 1** (full-screen camera): dotted middle guide (white 40 %), "Stay on the LEFT",
     `● REC 0:06 · Part 1`, Record's button and teleprompter. Exposure, white balance and focus lock
     when part 1 starts and stay locked for part 2.
  3. **Part 1 ready**: preview, *Retake part 1*, *Record part 2*.
  4. **Part 2**: part 1 as a 35 % ghost, "Now stay on the RIGHT"; alignment check twice a second
     (`VNTranslationalImageRegistrationRequest`, live frame vs part 1's first frame): > 2 % of the
     width → "Your phone moved — line it up with the ghost", else a green "Lined up". On REC, part 1
     plays from 0 at the same host time the recording starts (`AVPlayer.setRate(_:time:atHostTime:)`
     + `AVAssetWriter.startSession` at that instant). Part 1's sound only to headphones; without
     headphones it is muted and "Use headphones to hear part 1" shows.
  5. **Joining** (BrandRing), then **Your split video**: preview, *Try Pro blend (beta)*,
     *Use this video*. After Pro: Basic / Pro picker.

## Joining (`SplitCompositor`)
- Basic: part 1 left, part 2 right, horizontal soft mask ~60 px at 1080 wide (no line); the measured
  shift is applied to part 1; length = part 2; a shorter part 1 holds its last frame; both sounds
  mixed at an even level (the louder part turned down, never below 30 %). `.mov`, HEVC when part 2
  is HEVC.
- Pro (beta): `VNGeneratePersonSegmentationRequest` per frame on part 1, the person over all of
  part 2. Failure or > 3 min → back to Basic with "Pro blend not available for this video".

## Tests (synthetic media made inside the app; nothing in the repo)
- `testSplitCompositor` (`-CutSellScreen split_selftest`): seam (monotonic, no dark/bright stripe,
  soft edge), length, held last frame, sound levels (440/660 Hz Goertzel), alignment shift measured
  (0.048 vs 0.050) and applied (square at 97.0 px vs 97.5), Pro runs. All PASS on the simulator.
- `testSplitScreens` (screenshots of every step) and `testSplitEntry` (Record › Split › Part 1 › ✕).
- The camera itself (capture, lock, ghost sync, headphones) needs a real iPhone.

## Known limits
- Alignment measures only a shift (not rotation or zoom).
- With Bluetooth headphones the phone's own microphone records (A2DP output).
- Pro is slow on the simulator (~90 s for a 7 s clip); real iPhones run Vision on the Neural Engine.

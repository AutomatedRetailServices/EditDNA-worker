# CutSell iOS · White Indigo (23 Oct 2026)

Product Owner request: switch the whole app from the dark look to the approved white theme
"White Indigo" (Figma `PZnh8DuFcnhMUqPScHHL5q`, page `12_Prototype`; logo file `MFrDJ4TR19mEjCXNx3CRNO`).
App-only change: no engine, upload, Batch or server change; no keys; no user media in the repo.

Branch: `feature/white-indigo` (from build 39 = `61f3724`). Not merged to `main`.
It does NOT contain the later `feat/ios-build24` fixes that are still waiting for approval
(busy → Queued, push registration, edited Photos version).

## Where things live
- Tokens: `Theme.swift` (`LightPalette` maps every old dark `Color(hex:)` to its white role;
  `CS.accent`, `CS.wi*` tokens; Plus Jakarta Sans through `CS.inter`).
- Shared White Indigo pieces: `WhiteIndigo.swift` (`wiCard`, `WIHeader`, `WIHeaderIcon`, `WIScreenTitle`,
  `WISegmented`, `WISectionTitle`, `WINote`, `WIBackBar`, `WIStatusBadge`, `WIOriginalSafeCard`,
  `WIPillButton`, `WISheetCard`).
- Icons from Figma: `Assets.xcassets/WhiteIndigo/*` (template SVGs, tinted in code). Logo:
  `Assets.xcassets/Figma/LogoHorizontal` and `LogoIcon` (vector PDFs).
- Font: `Fonts/PlusJakartaSans-*.ttf` + OFL licence.
- Spanish: scratch tables `l10n/es8.py` → `L10nES.swift` and `es.lproj/Localizable.strings`.

## Screens rebuilt from Figma
Create 1164:2 · tab bar · Edits 1166:4 (full-width Processing · Ready · Products · Drafts; "All" removed) ·
Edits › Processing 1177:2 · Plan 1166:107 (no Sales Mix / Bulk cards) · Profile 1174:2 ·
Single Processing 1179:2 (BrandRing: 4 gradient arcs with neon glow, turning; transparent centre with the
logo icon; also Combine Processing, editor opening and Export) · Single Ready 1179:30 ·
errors 1176:2858 (ProblemScreen and the failed Processing state) · sheet 1176:2888 (LimitScreen) ·
permission screens · gallery-access card · intro and guide.
Everything else takes the tokens through `LightPalette`. The teleprompter stays as approved.

## Builds 41–44
- Create v3 (1164:2) and the product sheet "What are you promoting?" (1215:50, `ProductPicker.swift`) before every
  upload/record; the choice travels in the source metadata and is assigned to the catalog right after the project is made.
  Engine branch `feat/product-given` (NOT deployed) skips Haiku detection when the product is given.
- Edits Ready 1213:2, Product detail 1213:191, Assign 1213:293; Plan 1209:2/70/170 (colors only); tab bar 66 pt.
- Editor (page 04_Editor, frames "White Indigo · Editor · …"), colors only: small 26 pt back square (`EditorBackButton`,
  #EEF3FF, accent 35 % border); toolbar #F3F5F9 with black icons; timeline #E9ECF1 / rows #DDE1E8; voice-over and overlay
  tracks solid blue (selected = white with blue border); white panels (captions, style, overlay, volume, speed, audio,
  voice clip); voice-over sheet white with black pause/stop, blue mics, Play/Retake outlined, Save solid blue.
  The video and everything drawn on it (captions, overlays, countdown, record button) are unchanged.
- Blue is solid #1769FF everywhere (`CS.ctaGradient` is one color).
- Processing error (1233:3123): red "Retry processing" calls the deployed `POST /v1/jobs/{job_id}/retry`
  (`CutTracker.retry`); Edits › Processing uses the same retry.
- Not done yet: ERROR frames Queued 1233:3155, Export failed 1233:3199, Bulk/Plan Needs attention 1233:3243/3354;
  VideoTooLong/FileTooLarge stay as the 1176:2888 sheet (the 1233:3089 card differs — waiting for a decision).

## Screenshots
Workflow `cutsell-ios-white-indigo-shots.yml` (push to `feature/white-indigo`): small and large iPhone,
list `DesignScreenshots.whiteIndigo`, pushed to the branch `design-screenshots-wi`.

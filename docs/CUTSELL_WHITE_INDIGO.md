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
Everything else takes the tokens through `LightPalette`. Voice-over panels and the teleprompter stay as approved (dark).

## Screenshots
Workflow `cutsell-ios-white-indigo-shots.yml` (push to `feature/white-indigo`): small and large iPhone,
list `DesignScreenshots.whiteIndigo`, pushed to the branch `design-screenshots-wi`.

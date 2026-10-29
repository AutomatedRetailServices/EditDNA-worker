# CutSell "3D bubbly" style — 28–29 Oct 2026 (PO)

Branch `feature/home-bubbly` (from `build/split-and-speed`). App only, no server change, no merge to main.
TestFlight: build 61 = new Create home; build 62 = the same line on every main screen and the editor.

## Create (home)
- No logo in the corner. Hero = the CutSell logo itself (same bag and name), matte black with a soft puffy
  volume (no shine), bouncing like jelly; the letters of "CutSell.ai" hop one by one; a few gold coins.
  Livelier the first ~9 s, then calmer; Reduce Motion = still. Assets: `Assets.xcassets/Home/`.
- PO decision: no fake sales/commission pop-ups (could read as earnings claims) — only coins.
- Record (ink) + Upload (blue) 3D pills; Bulk / Combine / Sales Mix / Faceless / Plan my week as white 3D bubbles.
- "Stats" (ES "Estadísticas"): one white 3D card — blue 3D clock whose hands turn while the time saved counts up,
  the time saved, and a pink 3D bubble with the videos edited. Real numbers only (EditStats, 20 min per video).

## Every screen
- `Puff3D` / `puff3D(_:in:depth:)` (HomeHubView.swift): darker base under the face, face gradient, soft inner
  light on top and shade at the bottom, soft floor shadow. Looks: ink, blue (#2F7BFF→#0A55F0), pink, white.
- Shared components use it: PrimaryButton, SecondaryButton, WIPillButton, FilterChip, WISegmented, OptionCard,
  CircleIconButton, WIHeaderIcon, wiCard (default fill), Profile cards. Backgrounds: `HomeBackdrop`.
- Editor: `EditorBackdrop` (the logo area's pink/lilac/blue glows, still), lighter translucent panels, 3D back /
  quality / Export buttons, every tool in a small 3D bubble (the open one blue).

Known: the UI probe's "Plan my week" check and testLightMode's "prefAppearance" step fail for test reasons
(the screens open), not because of this style.

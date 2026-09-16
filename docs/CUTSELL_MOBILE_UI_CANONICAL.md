# CutSell.ai — Mobile UI Canonical

Status: active visual/product-flow canonical for the iOS app.
Branch: `cutsell/mobile-v1-clean`.

## Visual system

- Dark premium iOS UI.
- Primary brand colors: neon cyan / bright blue / electric purple / magenta on deep navy-black surfaces.
- Official CutSell.ai logo and branded icon family are used for Create, Edits, Stack, You, Gallery, and Processing.
- Native iOS symbols remain native for system actions such as Back, Help, Flash, and Flip Camera.
- Bottom navigation is always: `Create` / `Edits` / `Stack` / `You`.
- Active navigation state uses bright cyan/blue glow; inactive states remain visually quieter.
- Processing and export progress screens reuse the CutSell shopping-bag/play mark inside the circular progress ring.

## Canonical Create / Gallery flow

### Single Sales Cut

`Create Camera`
→ `Gallery`
→ native iPhone Photos picker
→ select exactly one video
→ `Add video` confirmation
→ `Processing`
→ result lands in `Edits`

Notes:
- Use the native iOS Photos picker rather than a custom full-library browser.
- No full `PHPhotoLibrary` permission is required merely for the user to browse/select media through the system picker.
- A permission-style mockup may be used in Figma to explain the flow, but production behavior follows the native picker/system authorization model.

### Multiple Sales Cuts

`Create Camera`
→ `Gallery`
→ native iPhone Photos picker
→ select multiple videos
→ `Add videos`
→ `Arrange your sales clips`
→ optional `Overlap audio`
→ show clip count + total duration
→ `Create my cut`
→ `Processing`
→ result lands in `Edits`

## Canonical Processing flow

Processing screen includes:
- video preview using the real selected video frame;
- mini filmstrip;
- circular progress ring with the CutSell bag/play icon;
- percent complete;
- visible stage progression, currently represented as:
  - Upload complete
  - Trimming clips
  - Transcribing audio
  - Cleaning pauses
  - Building final edit
- ETA when available;
- helper copy that the user may leave and return later;
- `Edits` active in bottom navigation.

## Ready / Export flow

When processing completes:

`Your cut is ready`
→ `View in Edits` OR `Export video`
→ `Export settings`
→ `Exporting`
→ `Export complete / Share`

Export settings may expose:
- resolution;
- frame rate;
- bitrate;
- estimated file size;
- Export / Cancel.

Exporting screen reuses the same branded progress language as Processing and shows stages such as:
- Preparing export
- Rendering final file
- Saving to Photos

## Export destinations — V1 vs pending integrations

### V1 required

- `Save to Photos` / save locally to the iPhone.
- Standard iOS share sheet (`More…`) may be used for installed apps and destinations supported by iOS.

### TikTok Share Kit — PENDING, NON-BLOCKING

TikTok Share Kit integration is intentionally deferred until the CutSell app itself is otherwise complete/stable.

Planned future flow:

`Export complete`
→ `Share to TikTok`
→ hand the exported video to TikTok through TikTok Share Kit
→ user finishes caption/sound/privacy/posting inside TikTok.

This is a future integration task and **does not block current mobile UI, editor, processing, export, or TestFlight development**.

Do not implement Direct Post API/OAuth publishing at this stage.

### Instagram / Facebook

Direct platform API publishing is not required for V1.
Use local export + iOS share sheet first. Any dedicated Meta publishing integration is a later product decision.

## Current design sequence

Screens are approved one by one before being rebuilt as editable Figma UI:

1. Create Camera
2. Add Video bottom sheet
3. Native iPhone video picker representation
4. Single-video `Add video` confirmation
5. Multiple-video arrange screen
6. Processing
7. Your cut is ready
8. Export settings
9. Exporting
10. Export complete / Share
11. Edits
12. Editor
13. Voice Over
14. Stack
15. You

## Product rule

The canonical backend/app behavior defines **what each screen does**. Approved visual references define **how it looks**. Visual redesign must not silently alter the real product flow or backend authority contracts.

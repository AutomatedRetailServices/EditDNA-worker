# CutSell simple engine v1.0

Status: implemented behind a switch, **off by default**. Not deployed. Not yet run end to end inside the app.

## What it is

A small editorial engine that replaces the legacy V2 decision stack for a Flow B job when
`CUTSELL_ENGINE=simple`. It was built and calibrated outside this repository between 29 Sep and
5 Oct 2026 against the Product Owner's Human Gold, then ported here unchanged.

```
source video
  -> audio (ffmpeg)                       cutsell_worker/simple_engine/asr.py
  -> Deepgram Nova-3 word times           (punctuation + filler words, no smart formatting)
  -> pass step      (1 Claude call)       is this one take, or several full passes?
  -> decision step  (1 Claude call)       which transcript lines to hide, and why
  -> loose-piece rule                     a kept part of 0-1 words under 2 s is hidden
  -> silence refine                       cut edges snap to real silence; never inside a word
  -> shot-change snap                     an edge next to an existing visual cut moves onto it
  -> splits + captions                    full non-destructive timeline
```

Everything after the two Claude calls is plain code. Default model: `claude-sonnet-5-5`.

## Files

| File | Role |
|---|---|
| `cutsell_worker/simple_engine/engine.py` | The engine. `process(words, duration, audio_path, video_path=None)` |
| `cutsell_worker/simple_engine/prompts/` | `pass_v1.txt`, `decision_v1.txt`. Tuned together with the constants in `engine.py` |
| `cutsell_worker/simple_engine/llm.py` | Anthropic call |
| `cutsell_worker/simple_engine/asr.py` | Audio extraction + Deepgram call |
| `cutsell_worker/simple_engine_adapter.py` | Maps the engine output to the existing `ProcessingResult` / `DraftTimeline` contract |
| `cutsell_worker/config.py` | `selected_engine()` reads `CUTSELL_ENGINE` |
| `cutsell_worker/worker_job.py` | The one branch in `run_flow_b_job` |
| `benchmarks/simple_engine_gold/gold_v1.json` | Human Gold: 25 videos, 555 keep/delete decisions, words, and the frozen v1.0 cuts for 24 of them |
| `scripts/simple_engine_eval.py` | Scores cuts against the Gold |
| `tests/test_cutsell_simple_engine.py` | Engine, adapter, switch, scorer. No network |

## Environment variables (names only)

| Variable | Meaning |
|---|---|
| `CUTSELL_ENGINE` | `legacy` (default) or `simple`. Anything else fails the job at start |
| `ANTHROPIC_API_KEY` | Required when `simple` |
| `DEEPGRAM_API_KEY` | Required when `simple` |
| `CUTSELL_SIMPLE_ENGINE_MODEL` | Optional. Default `claude-sonnet-5-5` |

With `simple`, the worker does not build the brain runtime and does not load Whisper. No GPU is needed.

## How the output maps to the draft the app already reads

| Engine output | Draft |
|---|---|
| visible split | one clip in `selected` (source in/out = split start/end, with its words) |
| hidden split that contains speech | one clip in `alternates`, `take_group_id = null`, so `draft_edits.restore_clip` can bring it back |
| hidden silence | not on the timeline |
| word-level captions, caption groups, reasons, token usage | `draft.diagnostics.simple_engine` |

`caption_text` of each clip is the clip's words, so the existing per-clip caption burn-in keeps working.

## Measured

- Human Gold agreement of the frozen v1.0 run: **478 / 514 decisions (93.0 %)** on 24 videos
  (`python scripts/simple_engine_eval.py`). Other runs of the same version scored 93.8-95.1 %;
  run-to-run variation is about one point.
- The code in this repository reproduces that frozen run exactly on all 24 videos when given the
  saved model answers and audio decoded from the source file. Decoding from an intermediate MP3
  instead moves three cut edges by 10-20 ms.
- Cost measured on the Gold set (Deepgram + Claude Sonnet 4.6): about USD 0.02 for a 1-minute
  source, about USD 0.07 projected for a 6-minute source.

## Known limits

1. **Export has not been run with this engine.** `export_job.run_export_job` rebuilds a
   `CanonicalEditPlan` and runs post-render QC and the perceptual watch-listen gate, which defaults to
   human review. A simple-engine draft carries none of the legacy diagnostics those gates read.
   Test export end to end before enabling the switch for users.
2. **Word-level captions are not rendered.** The renderer and the iOS app only know one caption per
   clip. The word-level data is stored in `diagnostics` for when they do.
3. **Output is still 1080x1920 at 30 fps.** That is the renderer and the iOS upload preparation, not
   the engine.
4. **Open editorial cases** (seen on the Gold and on six unmarked sales videos): a line from an
   earlier take kept before the hook; repeated takes that are not merged; the pass step is unstable
   on one Gold video (Y08) in about one run out of four.
5. `decision_v2` (abandoned-tail rule) was tested and **not** adopted; it is not in this repository.

## Changing it

The prompts and the numeric constants were tuned together. Any change to either needs a full re-run
on the Gold and a comparison of the score, per video, before it is merged.

## 1.2.1: 1.2 + sale review (2026-10-17, branch feat/simple-engine-review, NOT deployed)
After 1.4 (understanding as the editor) lost to 1.2 on the Gold's sales videos, the PO approved this plan: 1.2 keeps
deciding; a senior-editor reading of the 1.2 cut may only fix known problems in a SALE, applied only when 2 of 3
readings agree, never losing a number/price, never removing more than 25 % of the kept words (40 % for a restarted
pitch). Measured per fix kind on the Gold's 21 sales videos, 2 runs (the 3 answers replayed per kind):

| kind | run 1 | run 2 | any video worse? |
|---|---|---|---|
| none (1.2) | 90.4 % | 96.0 % | -- |
| restart (abandoned first run kept next to the full run) | +2 | +1 | no -> enabled |
| mixed (half sentence from an abandoned take) | +1 | 0 | no -> enabled |
| repeat (same claim twice) | 0 | -1 | yes (B03, B08) -> off |
| missing (hook/price/CTA/closer) | 0 | 0 | never fired -> off |

(1.2 itself moved 90.4 -> 96.0 % between the two runs: run-to-run variance is larger than any fix.)
Perfume video: the restart fix removes the abandoned first run (31-72 s) and keeps hook + problem + the full second
run, ending on her closer. Mode: CUTSELL_SIMPLE_ENGINE_MODE=review (default in code: rules = exactly 1.2).
Cost: one extra call per sale video (~USD 0.01), three when a fix is proposed (~0.04).

## 1.2.3: same cut, fewer tokens (2026-10-17, LIVE: deployed 7ba78c6 with Product Owner approval; rollback CUTSELL_SIMPLE_ENGINE_EFFORT=high or redeploy 213dcad)
- Sale review and haul review: when the first two readings agree, the third is not asked. A third reading
  cannot change a 2-of-3 vote, so the cut is identical (exhaustive test).
- Word fix: the last complete JSON answer is read; trailing text no longer forces a second call.
- `CUTSELL_SIMPLE_ENGINE_EFFORT` (thinking effort per step). Default `pass=medium,haul_review=low`;
  decision, sale review and word fix stay at the model default (high). `high` = every step as in 1.2.1
  (rollback without a redeploy).
- Measured on the Gold (26 videos x 2 runs, Sonnet 5.5), Claude cost per video and agreement on sales videos:

| | cost | sales | all |
|---|---|---|---|
| live 1.2.1 | 4.10-4.43 c | 95.2-95.7 % | 94.6-95.4 % |
| A: pass medium, haul review low, word fix low | 3.45-3.66 c | 94.9-96.0 % | 94.9-95.1 % |
| B: pass/haul/word fix low, decision + sale review medium | 2.72-2.76 c | 93.4-94.7 % | 92.2-92.6 % |

  B is rejected: 5 sales videos below the live minimum (G01 26/37). A is within the run-to-run variance;
  its one lower video (B07, 15/16) came from the decision step, which A does not change. The word fix at
  `low` once wrote a wrong brand ("acicalfares" -> "Aquaphor", the live engine wrote "Cicalfate"), so the
  default keeps the word fix at high. Expected cost: about 3.6-3.8 c per video, 13-15 % less than 1.2.1.
- Prompt caching was not adopted: within one video the shared prompt is read again only when a review needs
  a second reading (about 1 video in 5). At low volume, the cache-write premium on every first call costs
  about as much as the reads save.

## 1.2.4 candidate: token measurement + prompt caching (2026-10-18, branch feat/engine-token-cost, NOT deployed)
Measured per Claude call on the Gold (26 videos x 2 runs), prices from platform.claude.com/docs/en/about-claude/pricing
(Sonnet 5.5: input $2, output $10 incl. thinking, cache write 5 min $2.50, cache read $0.10 per million tokens).
- Live 1.2.3: 3.84 c per video. Input 52 % of the cost, thinking 35 %, visible answer 13 %. Per step: decision
  1.40 c, sale review 1.07, pass 0.49, haul review 0.45, word fix 0.44. max_tokens 16000, longest output 2763 (a
  ceiling only, not changed). No malformed-JSON retries.
- Changes kept: (a) each step's fixed instructions are sent as their own content block marked for prompt caching
  (the text the model reads is byte-identical, checked on the 26 Gold videos); (b) decision, haul review and word fix
  answer through structured output (JSON only). Every call logs step, tokens, cache tokens, max_tokens and thinking.
- Changes reverted because they changed the cut: pass step JSON-only (Y08 read as one pass, 3 of 3 runs, free text:
  several passes, 3 of 3) and sale review JSON-only without "why" (fixes proposed 14 vs 8 of 52, cuts changed).
- Not done: word-index answers for the decision (the answer is already line ranges, ~70 tokens; the input would grow),
  one transcript shared by the 4 calls (each step reads a different text: lines, KEEP/CUT lines, index:word; sharing
  would change what the model reads), thinking/effort/max_tokens untouched.
- Cut: the live engine itself gives the identical cut in only 11 of 26 videos between two runs (355 words decided
  differently). Candidate vs live stays inside that band (223-469 words); Gold agreement 92.7-94.1 % vs live 93.8-95.1 %.
  A control with caching and JSON switched off (= live requests) moved Y05/B07 exactly as much.
- Cost per video: 2.78 c when the cached instructions are read (another video within 5 minutes, e.g. Bulk),
  4.15 c when they must be written (a lone video). Break-even: about 1 video in 5 finding the cache warm.
- Rollback without a redeploy: CUTSELL_SIMPLE_ENGINE_CACHE=0 (one plain text block again), CUTSELL_SIMPLE_ENGINE_JSON=0
  (free-text answers again). Both at 0 = the live 1.2.3 requests.
- 2026-10-19, Product Owner approval to deploy 1.2.4 with: prompt caching ON only for a video of a Bulk send (the app
  sends metadata.bulk_size / bulk_id on each source, iOS build 26+; a lone video never caches), JSON-only on decision /
  haul review / word fix, pass and sale review unchanged, token log per call. The draft diagnostics record
  `prompt_cache` per source. If anything fails: CUTSELL_SIMPLE_ENGINE_CACHE=0 and CUTSELL_SIMPLE_ENGINE_JSON=0.

## B-roll over the voice (2026-11-10, branch feature/broll-after-stop-engine, NOT deployed)
PO spec 9-10 Nov: one project = the talking part + N B-roll clips. Code: `cutsell_worker/broll_overlay.py`, wired in
`simple_engine_adapter.py` and `worker_job.py`; tests `tests/test_cutsell_broll_overlay.py`. `engine.py`, `prompts/` and
`simple_engine/__init__.py` are untouched (guard test in `test_cutsell_mix.py`), so the cut itself is exactly 1.2.5.

**Contract (source `metadata.clip_type`).** `"broll"` (camera B-roll mode or a gallery clip marked B-roll) = B-roll;
`"primary"`, absent or anything else = voice. A request whose sources are ALL marked B-roll is cut as voice (there is
nothing to put them over). With no B-roll source nothing new runs: the output is byte-identical to before (test
compares against the pre-change adapter from git).

**B-roll is never cut.** No speech-to-text, no pass/decision/review/word-fix call, no silence refine, never in
`selected` / `alternates`. The voice cut, its captions and its zooms are identical with and without B-roll (tested).

**Draft format (no app change).** Each placed clip is one `draft.media_overlays` item, the type the iOS editor already
draws on the video and lets the creator move, trim and delete (Overlay rows under the video; `/v1/overlays/update|remove`):
`overlay_id` (`ai-broll-<source_order>`), `kind: "video"`, `uri` (the B-roll source's own URI), `start` / `end` (final
timeline seconds), `source_start` / `source_end` (in-point inside the clip), `mute_audio: true`, `layout: "full"`,
`fill: true`, `x 0.5`, `y 0.5`, `width 1.0`, `layer 0`. `draft.diagnostics.broll_overlay` records per clip what it
shows, the suggested phrase, where it went and why (`matched` / `product` / `spread`) or why not.
`overlay_uploads.validate_overlay_uri` now also accepts a VIDEO in this same user's same project upload scope
(`scoped_upload_prefix`), so the export and the editor's `preview_url` accept the B-roll source as an overlay. Another
user's or another project's object is still refused (tested).

**Matching: one extra small Claude call**, not inside the cut decision. The decision prompt is calibrated on the Gold
and frozen by the guard test; B-roll must also match the phrases that SURVIVED the cut, which exist only after the
decision; and it needs images. So after the cut: 3 frames per clip (1/6, 3/6, 5/6 of it, 512 px long side) + the kept
phrases with their times + the product name -> per clip "what it shows", the best phrase, the best frame, and where
the closing CTA starts. (v1 used the engine's model; **v2 uses Haiku, see below**.) Effort low, JSON schema,
max_tokens 2000. Normal API (not the Bulk Batch API); not cached / not remembered.

**Rules (enforced in code, whatever the answer says).** No B-roll in the first 3 s, nor from the start of the phrase
holding the closing call to action (model's `cta_phrase` or the engine's `cta` moment in the second half; at least the
last phrase and the last 3 s); 2-4 s each, starting on a phrase start and ending on a phrase end (else a word end, never
inside a word); at most 50 % of the video; no two overlap; each clip at most once, using its steadiest window
(4 fps 32x32 grey frame difference) among those containing the frame the model picked. Clips with no matching phrase go
where the product is mentioned (`metadata.brand` / `product` from "What are you promoting?", else `album_hint`, plus the
engine's `product` moments), the rest are spread evenly between the hook and the CTA. Clips under 2 s are not used.
**Fallback:** no answer, an error, an invalid answer or `CUTSELL_BROLL_AI=0` -> the same rules without suggestions. Any
error in the B-roll step leaves the cut without overlays and records it; it never fails the cut. A B-roll clip the
source format gate rejects is dropped from the cut instead of blocking it. A B-roll clip sent audio first: its video
is fetched; if it is not on storage yet the clip is placed without being looked at.

**Cost per video** (Sonnet 5.5: input $2, output $10 per M; image tokens ~ w x h / 750 ~ 200 per frame; thinking at
effort low estimated 0-600 tokens; not measured on real calls yet):

| | input tokens | output tokens | Claude USD |
|---|---|---|---|
| 30-60 s video, 3 B-roll clips | ~2.2-2.6 k (9 frames ~1.8 k + text) | ~100-600 | ~0.005-0.011 |
| 30-60 s video, 6 B-roll clips | ~4.0-4.4 k (18 frames ~3.6 k + text) | ~200-700 | ~0.009-0.016 |

That is about +0.5-1.6 c on top of the ~3.6-3.8 c of the cut. Deepgram is cheaper than sending the same clips today
(B-roll is no longer transcribed). Frames and steadiness are local ffmpeg (a few CPU seconds).

**Switches (names only).** `CUTSELL_BROLL_OVERLAY` = `auto` (default: on only for a request that has B-roll sources) or
`0` (off: every source is a voice source, exactly as before this change). Default `auto` because the iPhone app already
sends `clip_type: "broll"`, and without this step those clips are cut as if they were talk (or the cut fails with "no
speech" on a silent clip); a request without B-roll never reaches the new code either way. `CUTSELL_BROLL_AI` = `1`
(default) or `0` (no extra call, deterministic placement only). No new secret: it uses `ANTHROPIC_API_KEY`.

**To deploy (Product Owner decision; nothing deployed).** Both services: the Modal worker (`modal_cutsell_worker.py`:
the cut + the export read `broll_overlay.py`, the adapter, `worker_job.py`, `overlay_uploads.py`), deployed by
`.github/workflows/cutsell-modal-worker-deploy.yml` on a push to `feat/simple-engine-v1`; and the Render API
(`cutsell_app`, which signs the overlays' `preview_url` through `overlay_uploads`; without it the blocks are on the
timeline but the editor has no picture for them). Before enabling for users: one real end-to-end cut + export with B-roll
and a human watch of the result.

### B-roll v2 (2026-11-11, same branch, NOT deployed)
PO spec 11 Nov. Same files plus: `contracts.py` / `serde.py` (overlay `motion` fields), `media_overlay_render.py`
(Ken Burns), `uploads.py` / `overlay_uploads.py` (JPEG photo sources), `export_job.py`, `cutsell_app/main.py`,
`cutsell_app/auth_middleware.py`. `engine.py`, `prompts/`, `simple_engine/__init__.py` untouched; the cut call and
`CUTSELL_SIMPLE_ENGINE_MODEL` unchanged (tested).

**Source metadata contract (iPhone -> server, all string values, free-form `metadata`).**
- `clip_type`: `"primary"` (talking video; SEVERAL allowed, joined in source_order) or `"broll"`.
- `broll_kind`: `"video"` | `"photo"` (B-roll only). A photo source is a JPEG (`image/jpeg`, `.jpg`/`.jpeg`).
- `broll_order`: `"1".."N"` (B-roll only; the creator's order). `broll_keep_order`: `"1"` | `"0"` (same on every B-roll).
- Sources order: all primary first, then B-roll. In the audio-first flow B-roll sources carry NO `audio_uri`.
A source is a photo when it is marked `broll` AND its uri is `.jpg`/`.jpeg` (the overlay kind must match the stored
object, so `broll_kind` alone never turns a video file into a photo).

**Several talking videos.** Only voice sources go to the engine; each is cut and the kept parts follow one another in
source_order (the Combine path, unchanged). Overlay times are on that joined sum-of-clips timeline (the one the
renderer and `draft_from_dict` use); tested with a B-roll matched to a phrase of the SECOND talking video.

**Haiku for the images.** `match_with_claude` uses `CUTSELL_BROLL_MATCH_MODEL` (default `claude-haiku-5-5`), read at every
call (no redeploy to change it). `diagnostics.broll_overlay.ai` records `model` and `cost_usd`. Price table
`broll_overlay.PRICES`: Haiku 5.5 $0.10 in / $0.50 out per M tokens, Sonnet 5.5 $2 / $10 -- the values already in the
repo (`post_captions.py`, `product_tagging.py`, `mix_engine.py`, `simple_engine/llm.py`); **not re-checked against the
live Anthropic price page for this change: verify before relying on the totals below.**

**Frames.** Clip < 10 s: 3 frames (1/6, 3/6, 5/6). Longer: one every ~2.5 s (`round(duration / 2.5)`), max 10 (20 s ->
8 frames). 512 px long side. The model answers per clip `uses: [{phrase, at}]`, `at` = the second of the clip where
the content is; the piece used is the steadiest 2-4 s window that contains `at`. A clip of 10 s or more may be used
up to `min(3, duration // 5)` times, on different parts: a piece overlapping an already used part of the same clip is
never placed. A second piece of a clip gets id `ai-broll-<source_order>-2`. **PO 11 Nov: two pieces of the SAME clip are
at least 4 s apart on the timeline** (`SAME_CLIP_GAP_SEC`, end of one to start of the next; face or another shot in
between), enforced in code: a piece that cannot respect it is dropped, never placed closer.

**Photos.** One photo = one image for the model (no `-ss` seek: it drops a still's only frame). It is never sent to
speech-to-text, never goes through the format gate, has no filmstrip/waveform (`timeline_assets.status:
"not_applicable"`) and counts **0 minutes** (worker meter, `check_measured`, the adapter's allowance check and the API's
declared-length guard). Overlay fields (what the iPhone editor/export must draw the same way):
`{"overlay_id": "ai-broll-002", "kind": "photo", "uri": <the photo source uri>, "start": s, "end": e (2-4 s),
"x": 0.5, "y": 0.5, "width": 1.0, "source_start": 0.0, "source_end": null, "mute_audio": true, "layout": "full",
"fill": true, "motion": "ken_burns", "motion_zoom_from": 1.0, "motion_zoom_to": 1.08}`.
Meaning: the photo covers the 9:16 frame (scale to cover, centre crop), and its scale goes LINEARLY from
`motion_zoom_from` at `start` to `motion_zoom_to` at `end`, centred. Server render (`media_overlay_render.ken_burns_chain`):
one frame of the photo, scaled to cover 2x the region, `zoompan` (z linear in the output frame, centred x/y, d = frames
of the overlay, 30 fps) to the region size. `motion` is accepted only on photo overlays (`serde`); an overlay without
`motion` serializes and renders byte-identically to before (tested against the previous renderer from git).
Uploads: `.jpg`/`.jpeg` names accept only `image/jpeg` (or octet-stream), max 40 MB; video names never accept an image
type. `validate_product_source_uri(..., allow_photo=True)` only where a photo is expected (cut request when the source
is a B-roll photo, worker download, export sources -- skipped, it is drawn as its overlay --, playback links,
video-ready); every other reader (Mix, audio clean...) still refuses a JPEG. `validate_overlay_uri` accepts this same
user's same project's JPEG source as a `photo` overlay.

**Order 1-2-3.** `broll_keep_order "1"`: the shots appear in `broll_order` (1 before 2 before 3). PO 11 Nov: a long
clip's pieces stay inside its slot with the face in between: 1A -> face (>= 4 s) -> 1B -> 2 (the assignment requires
the next piece of the same clip to start >= 2 + 4 s after the previous one, and piece A must end >= 4 s before B). `best_ordered_assignment` (DP) picks strictly increasing phrases maximizing the number of shots
placed, then keeping the model's picks (cost 0 on its phrase, 1 + distance elsewhere), clips without a pick near their
even-spread target; reasons `matched` / `reordered` / `spread`. The model is also told the order. `"0"`: build-79 rules
(the model's matches, product mentions, even spread). Every rule still holds (face first 3 s and over the closing CTA,
2-4 s blocks, <= 50 % coverage, no two overlap); the fallback (no answer / invalid / `CUTSELL_BROLL_AI=0`) respects the
order when keep_order is on.

**Audio first.** The app sends no `audio_uri` for B-roll. The worker fetches each B-roll file directly; one that is not
on storage yet never fails the cut. **PO 11 Nov: the cut waits for it.** After the voice is cut (so the cut's own time
is not added), the worker polls storage for every still-missing B-roll file every ~3 s, with ONE total budget for all
of the job's missing files: `CUTSELL_BROLL_UPLOAD_WAIT_SEC` (default 60, 0 = never wait, max 300). A file that arrives
gets the same checks as at the start (photo readable, B-roll video through the format gate, <= 10 min) and is looked at
by the AI normally (its minutes are counted; its filmstrip is made). One still missing after the budget is
`broll_pending_upload`, gets 0 minutes and is placed from its declared length without being looked at (a pending video
with declared 0 s is not placed); one that arrived but is unusable stops being polled. `broll_upload_wait` in the
result (and `diagnostics.broll_overlay.upload_wait`) records `cap_sec`, `waited_sec`, `polls`, `arrived`, `rejected`,
`still_missing`. Total rather than per file: with 3 clips still uploading in parallel a per-file cap could hold the
cut 3 minutes; the total keeps the worst case at 60 s while parallel uploads still all get the full window.
The export reads a pending file once it is there.

**Cost per video, B-roll matching only** (estimates: image ~ w x h / 750 = ~200 tokens per 512 px frame; text ~600-800
tokens; output incl. low-effort thinking 100-800 tokens; prices as above, not measured on real calls):

| case | images | input tokens | Haiku 5.5 (v2) | Sonnet 5.5 (build 79) |
|---|---|---|---|---|
| (a) 3 short clips (< 10 s) | 9 | ~2.4 k | ~$0.0003-0.0005 | ~$0.005-0.011 |
| (b) 3 long clips (~20 s) | 24 (8 each) | ~5.4 k | ~$0.0006-0.0009 | ~$0.013-0.019 (same 24 frames) / ~$0.005-0.011 with build-79's 3 frames |
| (c) 3 photos | 3 | ~1.2 k | ~$0.0002-0.0004 | ~$0.003-0.007 |

About 15-20x cheaper than Sonnet: under 0.1 c per video on top of the ~3.6-3.8 c cut (if Haiku's repo price is right).

**To deploy (Product Owner decision; nothing deployed).** Modal worker (cut + export: `broll_overlay.py`, the adapter,
`worker_job.py`, `export_job.py`, `media_overlay_render.py`, `serde.py`, `contracts.py`, `uploads.py`,
`overlay_uploads.py`) via `.github/workflows/cutsell-modal-worker-deploy.yml` (push to `feat/simple-engine-v1` only), AND
the Render API (`cutsell_app`: JPEG presign/multipart, the cut-request check, the export-request check, the usage
guard, photo `preview_url` / `playback_url`). Both must ship together: an API without it refuses the photo upload; a
worker without it fails the cut on a JPEG source. Optional env: `CUTSELL_BROLL_MATCH_MODEL` (default Haiku), no new
secret. Optional `CUTSELL_BROLL_UPLOAD_WAIT_SEC` (default 60). Before users: one real cut + export with long clips, photos and keep_order, watched by a human; check the real
Haiku token usage / price in `diagnostics.broll_overlay.ai`.


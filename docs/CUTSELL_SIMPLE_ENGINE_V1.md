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

Everything after the two Claude calls is plain code. Default model: `claude-sonnet-4-6`.

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
| `CUTSELL_SIMPLE_ENGINE_MODEL` | Optional. Default `claude-sonnet-4-6` |

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

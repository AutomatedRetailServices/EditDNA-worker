# Mix (engine mix-1.0.1)

One product video in parts (Hook, Body, Benefit, CTA; 1-6 sections, 1-10 clips each, at most 100 videos). Every video
= one clip of each section, in order. The normal cut (simple engine, its prompts and steps) is not touched: Mix only
reuses its finishing functions (`tests/test_cutsell_mix.py::test_the_cut_engine_is_untouched_by_mix`).

## Flow
1. App: the clips are uploaded into one project (the Mix id; state `mix`, never listed), voice first.
   `POST /v1/mix/jobs` saves the Mix, creates one project per video (state `processing`), places each in the
   product (Edits · Products, `source: user`, `mix_id`, `mix_code`) and enqueues ONE job (`mix_job.run_mix_job`).
2. Job, clean phase (once per clip): Deepgram per clip (a short clip heard empty is heard again in the Mix's
   language) + ONE `mix_clip` call for all the clips (`mix_prompts/mix_clip_v2.txt`, model `CUTSELL_MIX_MODEL`,
   default `claude-haiku-5-5`), then the cut engine's finishing per clip.
3. Videos phase: when every clip's video is on storage (else `waiting_for_videos`; `/v1/projects/{mix}/video-ready`
   starts the job again), a clean clip per clip (common 1080x1920 or 1920x1080 canvas, 30 fps, -16 LUFS two-pass,
   30 ms fades at every edge), stored under the Mix project (`mix/clean-*.mov`).
4. Combinations (no AI, 3 at a time): clean clips joined (video copied, sound encoded once); each becomes a normal
   draft (one clip per part; captions and Auto Zoom moments moved onto its timeline). First and last ready times,
   Deepgram / Claude usage and `MIX_UNITS_PER_VIDEO` (plan units per video, default 1, recorded only) in the record.
5. A failed clip is named (`failed_clips`); `POST /v1/mix/{id}/retry` redoes only that clip; nothing made is redone.

## Order of the videos ("Pick how many")
`mix_engine.plan`: every clip of a section used a similar number of times (hooks and CTAs weigh most), two videos
in a row never share their start or their end when avoidable, pairs of clips repeated as little as possible.

## Real test (2026-10-20, 9 clips from two real Old Navy recordings, read-only probe)
24 videos: first ready 38 s after start, last 88 s; Pick 12: 38 s / 61 s. 0 clicks in 72 joins; every part
-16.8..-15.8 LUFS; 90 % of caption words within 0.08 s of a fresh transcription of the final video.
Cost per Mix of 24: Deepgram $0.0044 + Haiku $0.0024 + Modal ~$0.0031 = ~$0.010 (~$0.0004 per video).
Haiku vs Sonnet on mix_clip: Haiku 3.7x cheaper and more consistent on retakes (Sonnet kept stumbles in one run).

## Rollback
Worker: redeploy the previous commit (Mix code is only reached through `/v1/mix/*` and Mix projects).
API: redeploy the previous commit; `/v1/capabilities` then has no `mix` and the app's Mix screen fails to start
cleanly ("Your Mix couldn't start").

## Production check (21 Oct, after deploying a0ab25c)
- Production Mix with the 3+2+2+2 test clips: 24/24 videos Ready in 178 s. 0 clicks at 72 joins. Every part at -16.0 to -16.1 LUFS. Captions within a median of 0.01–0.06 s of a fresh transcription.
- Defect found: 1 of the 9 clips (6 s, English), heard alone, came back as Dutch, so it lost its words. Fix on `fix/mix-language`: any clip not heard in the Mix's language (the language most of its words were heard in) is heard again in that language. Not deployed yet; waiting for permission.

# Plan my week

Uses only Ready videos (Single, Combine, Bulk or Mix). The cut is not touched (`test_the_cut_engine_is_untouched*`).

## Data (Redis `cutsell:v1:plan:{user}`)
- `prefs`: `per_day` (1-4), `days` (Monday = 0), `reminders` [{label, time "HH:MM", on}], `tz`.
- `videos[project_id]`: `source` (single | combine | bulk | mix; set by the worker when the cut / Mix video is made,
  older videos: several sources = combine, else single), `posted_at`, `scheduled_for`, `caption`,
  `caption_source` (ai | user), `caption_log` (tokens, cost). Product and Mix code come from the catalog.
- `week`: {start (Monday), built_at, slots [{date, time, project_id, pinned}], missing}.

## Building the week (`post_plan.build`, rules, no AI)
Ready and unposted, oldest first; never the same product in two slots in a row (a product with so many videos left
that they could no longer be spread out goes first); one video of a Mix per day; the same Mix hook never on two days
in a row; slot times = her reminders that are on, then 9:00 / 13:00 / 19:00 / 21:00. Past days, posted videos and
videos she moved stay when the week is built again. `missing` = days left short and how many videos would fill them.
Checked on 300 random libraries (`test_the_rules_hold_on_many_random_libraries`).

API: `GET /v1/plan`, `PUT /v1/plan/prefs`, `POST /v1/plan/week/build`, `POST /v1/plan/slots/move`,
`POST /v1/plan/slots/remove`, `POST /v1/plan/posted`, `PUT /v1/plan/caption`.

## Caption (`post_captions`, Claude Haiku 5.5, a separate step)
Written once when a video is Ready (after its brand / product step; for a Mix, after all its videos are made; for
older videos, by a worker job when they are first planned). English, short, 3-5 hashtags, no price / discount /
promise the video does not say (prompt + a code check). Her edited caption is never overwritten.
`CUTSELL_POST_CAPTIONS=0` turns it off. Real test (12 of her videos): $0.00027 per video on average.

## App
Plan · Home "This week" (dots per planned video, today's plan, "Plan my week ›" → Goal, or Week once built),
Goal / Week / Day (Figma 1138:2749 / 2798 / 2872), local "Time to post" reminders that open their day, the night
before (WiFi + charging, BGProcessingTask `ai.cutsell.app.prefetch-posts`) tomorrow's videos are exported and kept
on the iPhone; otherwise Share exports with its progress.

## 21 Oct: Product Owner decisions (deployed 72ecfa5)
- One cut per recording. A recording is identified, per source, by its length (±1 s) plus the first words heard in it. Which cut represents it: a posted cut first, then the cut already in this week, then the newest cut.
- Personal videos: a per-video flag in the plan store. `POST /v1/plan/personal {user_id, project_id, personal}` sets or clears it, and `GET /v1/plan` lists the flagged videos in `personal`. Personal videos stay in Edits and are never planned. App: Edits ••• → "Personal · don't plan" / "Use in Plan".
- The creator's catalog: the products she named are assigned as `source: user`, so the AI never moves them. The "Cut AI" videos are CutSell · CutSell.ai. 10 personal videos are flagged. Plan has 11 usable videos.

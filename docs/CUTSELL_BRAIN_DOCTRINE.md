# CutSell Brain Doctrine — Raw to Postable

This document is the authoritative editing doctrine for the clean CutSell architecture.
It replaces transcript-first or rigid-funnel interpretations. Implementation must stay
modular and evidence-driven; do not add per-video hardcoded exceptions.

## Master decision order

**Understand globally → decide locally with global context → compose globally → review globally.**

The quality gate is not `pipeline completed`, structural validity, or blooper count.
The product target is a draft a competent short-form editor could post without repairing.

## 1. Universal Watch + Listen core

Before destructive editing, understand the complete source recording:

- creator intent and main topic;
- product/subject when one exists;
- beginning/development/payoff or available sales story;
- visual, verbal, combined visual+verbal hooks and re-hooks;
- retries and relationships between alternate takes;
- face/expression over time;
- eye contact and camera engagement;
- hands/body/gesture movement and congruency with speech;
- product handling and demonstrations;
- speech clarity, hesitation, stumbles and search-for-words behavior;
- intentional vs accidental pauses/reactions;
- performance evolution through the recording.

Burned-in captions, stickers, prices or text may exist in training/validation footage.
They are not the primary signal for editing decisions. CutSell-generated captions are
an editor ON/OFF feature and are outside Clean Cut reasoning.

## 2. Human-performance bad takes

A transcript can be complete and still be a failed take. Timestamp and reason about:

- false start;
- wrong take;
- verbal fumble / forgotten words;
- hesitation caused by losing the line;
- visible frustration;
- breaking character;
- recording-process joke or accidental laughter;
- looking away to search for what to say;
- facial reaction that communicates "I got that wrong";
- body reset / retry setup;
- camera adjustment;
- product-handling mistake;
- visual fumble;
- unintentional dead air.

Do not label authentic personality, intentional humor, normal mannerisms, or a reaction
that belongs to the actual story as recording errors.

## 3. Precision cuts

Do not force every candidate into whole-take KEEP/DELETE. If the spoken line is good but
the creator makes a bad reaction/body reset immediately before or after it, preserve the
line and trim the bad edge. Interior destructive edits require stronger evidence because
blindly cutting through speech can mutilate meaning.

## 4. Dead air in every mode

Remove long empty space when it has no speech, visual hook, reaction, story, comedic,
dramatic, emphasis, or reveal value. Typical removable cases include waiting, silence
after a failed attempt, silence before a retry, or searching for words.

Keep a short intentional pause when it contributes meaning, timing, personality, tension,
reaction, emphasis, reveal, or a visual hook.

**Rule:** unintentional empty space → remove; meaningful intentional pause → may stay.

## 5. Retry grouping and Best Take

Group retries by the same *specific underlying communication attempt*, not by broad topic
or sales role. Wording may change while the idea remains the same.

Best Take must combine:

- clarity and completeness;
- confidence and naturalness;
- pacing and audio quality;
- camera engagement / useful eye contact;
- facial expression;
- body and gesture congruency;
- appropriate energy;
- framing and continuity;
- low distraction/fumble;
- product presentation and sales effectiveness only when relevant.

Do not reward exaggerated delivery merely for energy.

## 6. Edit-mode routing

The whole-video pass routes the footage after understanding it:

### Sales Edit

For genuine TikTok Shop / UGC / product or service sales intent. Use the universal core
plus product and sales-story understanding. Available beats can include product reveal,
problem, discovery, feature, benefit, demo, proof, reaction, testimonial, result,
objection, CTA, and others.

Do **not** force `HOOK → PROBLEM → BENEFIT → PROOF → CTA`. Discover the strongest coherent
sales story that actually exists in the footage. Natural source order is the starting
point, but reorder when it clearly improves comprehension, hook strength, demonstration,
payoff, or sales coherence without inventing speech or claims.

### Natural / Non-Sales Edit

For yapping, storyteller/storytime, talking head, routine/lifestyle talking, commentary,
educational/explainer, vlog, and personal updates. Do not invent a product or conversion
objective.

Understand the main topic and natural story logic. Preserve engaging yapping,
story-building detail, personality and intentional humor. Remove or compress redundant
repetition, unrelated tangents that kill momentum, recording mistakes, wrong takes and
unintentional dead air.

### Mixed Edit

Preserve the natural story while integrating genuine product/sales beats when present.
Do not make mixed footage feel artificially commercial.

## 7. Hooks

Hooks can be:

- visual only;
- verbal only;
- visual + verbal together;
- later re-hooks.

A visually compelling silent opening must not be deleted merely because ASR sees no speech.
Hook strength is evaluated in the context of the complete story, not as an isolated first
sentence.

## 8. Global coherence

A locally good clip can still be wrong for the final video if it repeats a better take,
breaks chronology, kills momentum, contradicts the story, or weakens the sales/natural
narrative. Conversely, a non-sales personality/story beat may deserve to stay because it
creates context, trust, humor, relatability or payoff.

Never invent speech, claims, product facts, or events that were not present in the source.

## 9. Post-composition review

After assembling the draft, evaluate the finished sequence again as one story:

- does it make sense from beginning to end?
- is the hook appropriate to the actual footage?
- are there redundant ideas?
- did any wrong take, frustration, awkward reaction or body reset survive?
- are cut boundaries natural?
- are meaningful pauses preserved and dead spaces removed?
- are transitions coherent?
- for sales: does the product story progress convincingly without a forced funnel?
- for natural: does personality survive while the story/topic remains engaging?

If the answer is no, revise the responsible layer instead of declaring success.

## 10. Validation doctrine

Pure blooper compilations are an **error-recognition dataset**, not raw-to-clean sources.
Raw mixed sessions containing good material + errors + retries validate Clean Cut,
boundaries, retry grouping and Best Take. Finished good ads validate sales-story and edit
patterns. Raw→human-final pairs are the strongest direct supervision/evaluation when
available.

A benchmark passes quality only when the output itself is postable; structural execution
success is necessary but not sufficient.

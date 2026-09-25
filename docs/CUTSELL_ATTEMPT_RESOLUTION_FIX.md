# Attempt classification and retry consumption — 2026-09-25

Scope: `feat/gpt-whisperx-video00`, based on `f7aceb0`. Offline correction,
not production promotion or a new media qualification.

## Observed failure and root cause

The saved MOV audit contains failed attempts, mixed preparation/delivery and
one long audience-labelled replacement whose text still contains preparation.
Existing coverage protection cannot correct a false audience classification.
Code inspection found additional actionable gaps:

- The Gemini prompt requested exactly one winner inside a session window,
  although the window can contain several unrelated communication attempts.
- The typed boundary accepted `winner` together with `mixed` or `recording_only`.
- The initial replacement proposal searched only the current window and did not
  require directional claim coverage.
- The final proven-retry authority inspected the kept pool but required the
  specific `winner` label, excluding a clean independent delivery labelled `keep`.
- Short exact abandoned openings with only two content tokens could not enter
  the existing relation comparison despite later literal completion.

These are verified code gaps, not proof that they explain every MOV mistake.

## General correction

1. Prompt separates intended audience content, execution usability and retry
   comparison. Mixed candidates cannot be clean winners; unrelated valid ideas
   use keep. Chronology is supplied as source start/end metadata, never output
   edit commands. Humor, reactions and product facts remain protected.
2. Validation normalizes winner+mixed/recording_only to uncertain, retaining
   proposed_label, original role, recording confidence and word boundaries.
   This withholds winner authority; it does not relabel the candidate failed,
   delete it, or exempt any audience content from loss checks.
3. Collect the same planned judge calls first, in the same order and budget.
   Replacement proposals then consult all returned windows of the SAME creator
   session, before downstream composite resolution. Conflicting decisions are
   not flattened into implicit positive consensus. No extra provider request.
4. The proposal checks directional coverage before the existing complete-take
   identity guard. Failure is reported as COVERAGE_NOT_VERIFIED where appropriate.
   The existing final authority can accept a high-confidence keep only with
   explicit audience evidence and all-window positive consensus. Winner peers
   with recorded judgments also require consistent confidence and no contrary
   decision. Historical no-row winner callers retain compatibility.
5. Preserve inferred AND original recorded partition identity. Earlier hooks
   can remove boundary-adjacent candidates; recomputation on the reduced pool
   must not erase a known separation between creators/scenes.
6. Short incomplete openings require an exact ordered prefix of a complete
   later delivery (at least four words/two content tokens). Local confirmed
   retry evidence, coverage, chronology and source/partition restrictions still
   apply. A phrase, gesture or profanity alone grants no deletion authority.

## Verification

- 22 new EN/ES regressions: cross-window keep consumption, mixed winner
  normalization, preservation of trim evidence and humor, exact incomplete
  prefixes, unique information/numbers/negation, contradictory windows,
  partition isolation before/after pruning and real composite-chain consumption.
- Relevant regression: 623 tests passed. No paid inference, GPU run or render.
- Independent QA: 126 targeted tests passed. QA reproduced and caused repairs
  to the partition-crossing and winner-consensus defects before acceptance.
- Older identity/observability positive fixtures omitted the referral meaning
  from their replacement. Their positive examples now preserve the original
  information; the old abbreviated example remains an explicit negative test.
  One incomplete-peer diagnostic now correctly reports insufficient overlap.
- Existing dense AV payload, transport budget, planner, recording trims,
  semantic protections and hybrid pipeline tests passed.

## Remaining qualification

The prompt is unqualified on new Gemini inference. A model can still falsely
label preparation audience; absence of a mixed label is not proof of cleanliness.
No saved baseline result is rewritten or claimed as a new output. This change
does not implement full final perceptual repair, remove freeze protections,
force prosody on every family, or settle every cross-group semantic conflict.
Repeat real-source qualification and inspect the outputs before claiming MOV
cleaning success, generalized editing improvement or release readiness. A new
paid run remains subject to AGENTS.md validation step 9.

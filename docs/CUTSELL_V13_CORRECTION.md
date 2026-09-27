# Yaskira 09 v13 correction — offline evidence, not approval

## Live v16 result

Run36345575257, commit a662be83: technical success, delivery status
NOT_DELIVERABLE_NEEDS_HUMAN_REVIEW. Demo96.65–106.6 is no longer in renderer
trim audit; the visual-floor correction has live mechanical evidence.
Opening aborted prefix and final recording aside remain. Tail proposal is
3 words at.85 confidence and was refused. The second selected take extends
to30.2, including an unfinished next phrase. ASR output and candidate decisions
vary across runs; no claim of stable calibration or final approval is justified.
V14 hash failure remains unproven. Next work needs targeted source/candidate
evidence and a general safe semantic correction, not lower safety thresholds
or hardcoded human-gold cuts. The rejected accent-folding fix must stay removed.

## Live v15 and bounded follow-up

Run 36344493562 completed and rendered. It recovered opening and demo selection,
but is NOT approved: opening retains an aborted accented prefix; final recording
aside remains; renderer trimmed the approved demonstration by 8.185 seconds
(106.66 -> 98.475). The v14 semantic failure did not reproduce; NOT resolved.

Follow-up v16:
- Proposed accent-folded aborted-prefix correction was REJECTED by independent
  QA: it could delete valid 'tomo café' after 'tomo cafeína'. It was removed
  before publication. Opening debris remains unresolved; negative tests retain
  complete words and faithfully document that the recorded opening is not fixed.
- Renderer receives a source/clip-bound trailing trim floor for an approved
  audience demonstration. Silent action is retained, including fragment lineage
  and coalescing. Ordinary speech/legacy segments retain existing trimming.
- Tail classification has its own optional confidence rather than sharing take
  selection confidence. The .97 floor and independent audio/alignment/protected
  word checks remain; no live success claimed. Proposal/confidence now audited.
  V2-only output reserve accounts for the field; dollar and token caps unchanged.
- 445 boundary/selection/render/engine/replay tests pass locally. Real model
  adherence, video review and the intermittent v14 contract failure remain open.

## Live v14 follow-up

Run 36343804777 (commit 7e6fb8b) passed workflow tests and runtime credential
loading, but failed before rendering: Boundary changed frozen Selection
semantic content, expected=311549ccb522 actual=f7814c5f5d46. No video was
produced; the three editorial corrections are NOT live-qualified.

The failure artifact only contained error hashes, not replayable clips.
The v15 diagnostic change preserves frozen and final clip/word snapshots and
an allowlist of physical-operation diagnostics in a separate failure JSON.
It does not weaken Freeze, refreeze altered content or change editing behavior.
The recorded demo with real post-Freeze callbacks and individual recorded
clips did not reproduce the failure. Actual v14 cause remains unproven.
Independent inspection identifies word/text mismatch during splitting,
crossing word intervals and ordering of overlapping parents as hypotheses.
The diagnostic patch passes 39 targeted engine/replay/tail tests locally;
independent QA found no blocker and separately passed 12 engine tests.

Branch: feat/editorial-engine-v2-whole-video-rebuild. Reference run:
36340955692. Production and main are unchanged.

## Proven causes

- The saved selection labels the useful opening failed despite audience AV
  evidence. The previous safeguard required three new content tokens AND 40%
  novel vocabulary against the whole selected story. The percentage condition
  misses an opening containing unique information but much shared vocabulary.
- The demonstration's first instruction receives redundant_retry at .80.
  A broad shared topic is not proof that the corresponding visual action is
  redundant. Losing this candidate also prevents the existing two-neighbor
  physical continuity restoration from retaining the demonstration gap.
- The final ASR words span a measured pause. A pause-only CTA trim would delete
  legitimate safety qualifiers too. Independent QA reproduced this with
  'Solo para adultos'; that proposed fix was rejected, not shipped.

## Bounded corrections

1. Preserve AV-majority failed candidates with the existing three-uncovered-
   content-token floor, without the percentage-of-entire-clip requirement.
   Merge overlapping AV spans before measuring coverage; reject invalid spans.
2. Preserve a low-confidence (<.90) redundant instruction only with same-source
   adjacent selected explanation, a shared content token, an AV-observed
   continuous demonstration spanning the gap and no recorded restart/fumble
   description. Exact content-token subsets still remain redundant. This is
   conservative inclusion, not proof that all semantic paraphrases are unique.
3. Add optional trailing_recording_word_count to the same V2 model call.
   Require an explicit SELECT proposal at >=.97, matching aligned word stream,
   valid non-overlapping words, a short suffix, numeric/negation preservation,
   and independent primary-floor audio silence. Record every refusal. Boundary
   refresh respects accepted source-bound exclusions; Freeze remains intact.
   No extra provider call or increased dollar ceiling is introduced. Payload
   and response size can grow; existing budget failures remain visible.
4. Disable the old short-post-CTA heuristic in V2. CTA + silence is not by itself
   production-error evidence. Legacy V1 behavior is unchanged.

## Evidence limits

The fixture is a reduced copy of v13 words, AV, decisions, final clips and
measured events. Tests are COMPONENT replays, not a full historical pipeline
replay with every intermediate input. The demonstration test uses recorded
decisions and words through real V2 resolution/recovery/Freeze with the
post-Freeze boundary callback set to identity; it is not a rendered video.

The recorded opening and demonstration omissions are recovered by those tests.
The historical run has no explicit tail proposal, so faithful replay MUST NOT
claim the tail fixed. A separate synthetic proposal using its real aligned
words demonstrates safe execution and protection from boundary re-expansion.
Actual model adherence and final video quality need live qualification and
Watch + Listen review. No cross-video/generalization or release claim is made.

## Documentation exception

CUTSELL_DECISIONS.md already contains invalid UTF-8 on this branch. GitHub
fetch also fails decoding it. It was preserved unchanged; this file and the
CURRENT_STATE entry are the durable record for this cycle, not a rewrite of
canonical doctrine.

## Verification checkpoint

- Directed engine, provider, V2 replay and negative-case tests: 83 passed
  before the final three invalid-alignment controls were added.
- Broader boundary/selection regression: 421 passed at that same checkpoint.
- Independent QA reproduced two unsafe proposals; both were addressed before
  publication. Follow-up review found no additional concrete blocker and
  requested the negative matrix, now present in test_cutsell_v2_recording_tail.py.
- compileall and git diff --check passed. These are offline engineering checks,
  not editorial approval, rendered quality or complete repository regression.
- Final broader boundary/selection regression after the three additional
  invalid-alignment controls: 424 passed in 5.58 seconds.

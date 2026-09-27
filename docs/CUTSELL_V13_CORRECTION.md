# Yaskira 09 v13 correction — offline evidence, not approval

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

# CutSell Forensic: BestTake Wrong-Winner Class — Papillary + Pimples (D-104)

**Status: OFFLINE INVESTIGATION ONLY. No `cutsell_worker/*.py` file was
modified to produce this document. No RAW was launched, no paid provider
was called.**

**Branch:** `feature/runpod-pod-on-demand`. **HEAD:** `dabdd28`
(`dabdd287317c3f4ffae8bac22540f492a3136145`), clean tree, unchanged
throughout this investigation.

**Source RAW:** `34077889576` (job log decoded to
`scratchpad/d103_raw_job_log.txt`), the same run analyzed in the D-103
real-media qualification. All clip texts and diagnostic fields quoted
below are taken directly from that log. Cut.ai and Human Gold were used
ONLY to identify which realization each reference kept, per the strict
QA-reference rule — neither was fed into any reconstruction below, and no
production code was touched.

---

## 1. Method

For each family, the real clip texts and Hybrid/Gemini decision records
were extracted verbatim from the RAW's own diagnostics, then run through
the ACTUAL, unmodified production functions (`classify_claim`,
`resolve_critical_coverage_dominance`, `any_pair_contradicts`,
`_is_retrospective_condition_realization`,
`_members_missing_required_condition_realization`,
`_is_incomplete_content_subset` / `_exclude_incomplete_subset_losers`) in
a throwaway local Python session — this is inspection/reconstruction, not
a code change. This gives DIRECT proof of what the general ladder would
have concluded had the single-winner fast path not intervened, rather
than inference from the ladder's physical-region output alone.

---

## 2. PAPILLARY family (`tg_e4439c0e12e9004479`)

### 2.1 Reconstructed evidence chain

| Stage | `clip_582ea96f...` (reference-chosen) | `clip_c4c94acb...` (model "winner") |
|---|---|---|
| Text | "Síntomas que tuve. Según yo, era sintomática, pero sí hubo indicios ahora mirándose atrás." | "Síntomas que no me parecían sospechosos, pero que ahora que lo analizo, sí eran sospechosos." |
| Retry-family membership | Same family (`tg_e4439c0e12e9004479`), 2 members, lexical-grouping only (per D-099, `take_grouping.py` never sees `whole_video_context`) | same |
| `complete_idea` | Not printed in this diagnostic dump (field not surfaced to CI JSON) — not observably `False` anywhere in the log | same limitation |
| Confirmed local-performance evidence (Hybrid decision record, chunk 2) | `local_failure_corroborated: true`, `local_failure_reasons: ["dense_physical_reset:6", "visual_fumble:0.72"]`, `semantic_delete_recommended: true`, `delete_basis: "semantic_failed_plus_local_performance"` | `local_failure_corroborated: false`, `local_failure_reasons: []`, `semantic_delete_recommended: false`, `delete_basis: "kept_fail_open"` |
| Hybrid/Gemini label (governing chunk) | `"failed"`, confidence 0.85 | `"winner"`, confidence 0.95 |
| `classify_claim` (real production call) | `('ACTION_EVENT', 'SUPPORTING', 'general_statement')` | `('NEGATION', 'CRITICAL', 'negation_present')` |
| `_is_retrospective_condition_realization` (D-103) | `False` (no tracked belief marker — "según yo" untracked) | `True` (`parecían` + `ahora`) |
| `resolve_critical_coverage_dominance` (both members, reconstructed) | loses | **wins as dominant** (`clip_c4c94acb...`) |
| `any_pair_contradicts` | `False` | |
| `_semantic_best_take` (live, D-103 code) | discarded | selected, reason `"single_semantic_winner"` |
| Both references | **chose this realization** | rejected |
| Physical Level-1 (rendered MP4) | `missing_delivery`, 7.25s, `take_choice_against_both_references`, BestTakeResolver | kept |

### 2.2 Decisive finding

**Bypassing the fast path would NOT have fixed this family.** Both
members were fed through the real `resolve_critical_coverage_dominance`
(the exact Step 3/4 the fast path skips) and it independently elects
`clip_c4c94acb...` as dominant — the SAME wrong answer — because
`classify_claim` scores its negated "no me parecían... sí eran
sospechosos" as a `CRITICAL`/`NEGATION` claim, while the reference-chosen
clip's "Según yo, era sintomática... indicios ahora mirándose atrás" is
scored `SUPPORTING`/`ACTION_EVENT`. The general ladder's own dominance
check is driven by the identical claim classifier the fast path's veto
already knew was blind here (this is the exact limitation documented in
D-103's own module comment, now confirmed to also defeat Steps 3/4, not
just the veto).

D-103's own Step 2.5 exclusion, run against the real text, returns
`missing = {clip_582ea96f...}` — i.e. if this family HAD entered the
`len(winners) != 1` branch, Step 2.5 would have actively EXCLUDED the
reference-correct realization (the pattern-matching role is inverted: the
model's wrong winner is the one that surface-matches the retrospective
pattern, not the correct sibling). This is worse than a no-op — it is a
latent second failure mode of D-103's mechanism, only invisible in this
run because the fast path never let the general ladder execute at all.

### 2.3 Primary Question 1 classification (papillary)

**Semantic sufficiency / required meaning not represented** — specifically
by the shared claim classifier, not by "evidence ignored." The delivery
evidence (confirmed visual reset+fumble on the reference-correct clip) is
real and correctly disfavors it on a pure cleanliness axis; the miss is
that no existing content-representation layer (claim classification,
D-103's marker pattern, `critical_coverage_sets`) recognizes this
realization as carrying the unique required content, regardless of
which resolution path (fast path or general ladder) is used.

---

## 3. PIMPLES family (`tg_edb72c9305a16337b5`)

### 3.1 Reconstructed evidence chain

| Stage | `clip_ecf64fd...` (ref-chosen) | `clip_d0291f...` (ref-chosen) | `clip_adcd41770...` (model "winner") |
|---|---|---|---|
| Text | "También me salían espinillas, era como un rush, una alergia." | "Otro síntoma era que me salían espinillas como si fuera una alergia de esta parte aquí, detrás de la oreja y en el cuello. Me salía por temporadas." | "También me salían espinillas en esta parte de aquí, detrás de la oreja y todo el cuello, que yo pensaba que era alergia, pero era como espinillas de personas con problemas hormonales." |
| Confirmed local-performance evidence | `local_failure_corroborated: true`, `["dense_physical_reset:7", "visual_fumble:0.85"]` | `local_failure_corroborated: true`, `["dense_physical_reset:6", "visual_fumble:0.71"]` | `local_failure_corroborated: true`, `["dense_physical_reset:7"]` (no fumble score reported) |
| Hybrid label (governing) | `"alternate"`, 0.6–0.9 (unstable across chunks — see 3.3) | `"alternate"`, 0.7–0.8 | `"winner"`, 0.9 |
| `classify_claim` (real call) | `('ACTION_EVENT', 'SUPPORTING', 'general_statement')` | same | same |
| `_is_retrospective_condition_realization` | `False` | `False` | `False` |
| `resolve_critical_coverage_dominance` (all 3, reconstructed) | **`None`** (no CRITICAL claim exists in the family at all — the check never engages) | | |
| `any_pair_contradicts` (all pairs) | `False` | | |
| `_is_incomplete_content_subset` (all directions) | `False` | `False` | `False` |
| Both references | **chose both of these** (as two separate beats) | | rejected entirely |
| Physical Level-1 | `missing_delivery`, 5.75s | `missing_delivery`, 8.0s | kept, `false_keep`, 11.7s, all `take_choice_against_both_references` |

### 3.2 Decisive finding

**None of the existing content-representation checks distinguish these
three candidates at all.** `classify_claim` scores all three as plain
`SUPPORTING`/`ACTION_EVENT` — there is no `CRITICAL` claim anywhere in
this family, so `resolve_critical_coverage_dominance` returns `None`
immediately (`dominance_critical_claims` is empty) regardless of which
path runs it. No contradiction. No D-103 retrospective pattern. No
literal-subset relationship in either direction (D-101 Fix B). This is
NOT a "required meaning invisible to the classifier" case like
papillary — there is no meaning-classification signal available to be
blind to. The only evidence that varies at all across the three is the
confirmed local-performance corroboration: `adcd41770` (the wrongly kept
one) is the only member WITHOUT an explicit `visual_fumble` score, while
both reference-preferred clips carry one (0.85, 0.71).

### 3.3 Additional finding: label instability

Unlike papillary (one governing chunk), the pimples family's Hybrid
labels genuinely disagree across the diagnostic's own chunks:
`family_window_labels` (chunk 3 alone) gives `clip_ecf64fd...` `"alternate"`
0.6, while `global_merge_labels` (the label actually consumed) gives it
`"winner"` 0.9 — the same clip received a materially different label
depending on which window/merge pass produced it. This mirrors the
run-to-run arbiter inconsistency already recorded as an open escalation
in D-097.11 — it is evidence of a SEPARATE, already-known issue (label
stability across Hybrid windows), not something this task is scoped to
fix.

### 3.4 Primary Question 1 classification (pimples)

Cannot be classified as "required meaning not represented" — there is no
meaning-representation gap here at all; every content-based check is
silent. The two candidates that classify_claim/coverage/contradiction/
subset checks are ALL indifferent to. Given no numeric DeliveryScorer
score or raw `MediaSignals` value is surfaced in this diagnostic dump
(Section 4.3), the specific mechanism cannot be proven further offline.
The two remaining, evidence-consistent possibilities are:

- **"model label treated as excessive authority"** — the fast path never
  consults `local_selected_clip_id` (the DeliveryScorer-informed rank,
  which — architecturally — DOES weight `visual_fumble` at `-0.12` per
  `take_judge.score_take`) at all when exactly one "winner" label exists.
  Given `adcd41770` is the only member without a reported fumble score,
  it is plausible DeliveryScorer's own rank already agrees with
  `adcd41770` too, in which case bypassing it changes nothing; this is
  UNPROVEN either way from this diagnostic dump.
- **Family-membership granularity** — the two reference-preferred clips
  differ from each other (short "rush/alergia" vs. longer "detrás de la
  oreja... por temporadas") in a way consistent with two DISTINCT
  sequential beats a human editor kept as complementary content, not as
  competing retries of one idea; if so, the correct fix belongs to
  `IdeaClusterer`/`RetryFamilyFormation` (would this family have been
  split at all) or `CompositeResolver` (should more than one member ever
  survive), not to `BestTakeResolver`'s single-winner authority. This is
  also UNPROVEN from available evidence — it would need the two
  reference edits' own segment adjacency, not available in this
  diagnostic dump.

---

## 4. Primary Question 2 — evidence-signal classification (both families)

| Signal | Papillary | Pimples |
|---|---|---|
| Completeness (`complete_idea`) | NOT AVAILABLE (field not surfaced in this diagnostic dump for either candidate) | NOT AVAILABLE |
| Retry relationship (grouping) | AVAILABLE AND USED (lexical grouping placed both in one family) | AVAILABLE AND USED |
| DeliveryScorer numeric score / `local_selected_clip_id` | AVAILABLE BUT NOT SURFACED in this dump (architecturally computed, never printed to CI diagnostics; also never consulted because the fast path short-circuited before reaching it) | same |
| `MediaSignals` (face/eye contact/motion/fumble/expression/gesture/distraction) raw values | NOT SURFACED (only the derived `local_failure_reasons`/`local_failure_corroborated` summary is printed, not the raw per-field `MediaSignals` record) | same |
| Confirmed local-performance corroboration (`local_failure_corroborated`/`dense_physical_reset`/`visual_fumble`) | AVAILABLE AND USED — by Hybrid's own delete-recommendation logic, correctly identifying the reference-chosen clip as the noisier one | AVAILABLE AND USED — present on all 3 members, does not discriminate cleanly (Section 3.2) |
| Measured dead air | NOT queried in this pass (not adjacent to either family's boundaries in the printed regions) | same |
| Confirmed `wrong_take`/`retry_setup` visual event | NOT queried in this pass; per D-099, `take_grouping.py` cannot see it regardless | same |
| Lexical restart evidence | AVAILABLE BUT DEFAULT/UNMEASURED for this pair specifically (neither `same_opening_restart` nor `_safe_short_prefix_retry`/D-097.12 fired — these are near-paraphrases, not prefix retries) | same |
| Semantic contradiction/polarity | AVAILABLE AND USED — `any_pair_contradicts` correctly returns `False` (this is not a contradiction case) | AVAILABLE AND USED — `False` |
| Semantic sufficiency / unique required proposition | AVAILABLE BUT MISCLASSIFIED — `classify_claim` and `_is_retrospective_condition_realization` both evaluate the WRONG side as carrying the unique/critical content (Section 2.2) | AVAILABLE — correctly finds nothing (there is genuinely no CRITICAL claim in this family) |
| Temporal/narrative role | NOT AVAILABLE from this diagnostic dump | NOT AVAILABLE |

---

## 5. Primary Question 3 — is the single-winner fast path too strong?

**Yes, structurally — but that alone does not explain either failure.**
When exactly one member carries a `"winner"` label ≥ threshold, the fast
path (`pipeline.py:473-479`) returns immediately and skips: attempt
completeness (Step 2), D-103's required-realization exclusion (Step
2.5), `resolve_critical_coverage_dominance` (Steps 3/4),
`any_pair_contradicts`+coverage-asymmetry (Step 5), and — critically —
the DeliveryScorer-informed `local_selected_clip_id` fallback itself
(Steps 6-9's tie-break target).

However, Section 2.2's direct reconstruction proves that for
**papillary**, routing through the bypassed Steps 3/4 would have made NO
difference (same wrong answer, same classifier) — so "the fast path
skips evidence the general ladder would use" is TRUE architecturally but
FALSE as a sufficient explanation: the general ladder's own content
checks share the same blind spot. For **pimples**, the bypassed content
checks (Steps 2/2.5/3/4/5) are uniformly silent regardless of path — so
the fast path's bypass is potentially consequential ONLY insofar as it
also skips the DeliveryScorer-informed `local_selected_clip_id` fallback,
which is the one piece of evidence that was never observed in this
diagnostic dump and so cannot be confirmed to differ from the model's
own choice.

**Smallest general gating rule, if the DeliveryScorer disagreement for
pimples were confirmed:** force fallthrough of the single-winner fast
path whenever the model's chosen winner is NOT also the DeliveryScorer's
`local_selected_clip_id` (i.e. treat the winner label as a nomination
that must still beat the RankedTake fallback it is currently never
compared against) — but this specific gate is unproven and, per Section
2.2, would not have touched papillary at all, since nothing suggests
DeliveryScorer and the model's winner disagree there.

---

## 6. D-103 marker-approach verdict

**BRITTLE / FAMILY-SPECIFIC.** D-103's own module comment already
predicted this; Section 2.1's reconstruction confirms it operationally:
the marker pattern engaged with the WRONG candidate in the one real case
tested (the model's incorrect winner surface-matches "belief +
retrospective," the reference-correct clip does not), and — new finding
in this task — even a hypothetical marker fix would not by itself close
papillary, because `resolve_critical_coverage_dominance`'s classifier
independently reaches the same wrong dominant candidate. No marker
expansion is recommended or implemented here.

---

## 7. Would the same proposed evidence/gate cover both families?

**No — not proven for any single candidate gate.** Papillary's failure
is a content-classification blind spot that ALSO defeats the general
ladder's own dominance check; a structural "let the fast path fall
through more often" fix does not touch it. Pimples' failure shows NO
content-based signal at all distinguishing the candidates; only an
unobserved (in this diagnostic dump) DeliveryScorer disagreement or a
family-membership/granularity issue remain as live, unproven hypotheses.
These are two different root-cause SHAPES, not one class with two
symptoms.

---

## 8. Decision table

| FAMILY | WRONG WINNER | BETTER SIBLING(S) | MODEL LABELS | DELIVERY/MEDIA EVIDENCE | COMPLETENESS | MEANING/SUFFICIENCY | FIRST WRONG AUTHORITY | EVIDENCE IGNORED | GENERAL FIX SHAPE |
|---|---|---|---|---|---|---|---|---|---|
| Papillary | `clip_c4c94acb...` | `clip_582ea96f...` | winner 0.95 vs failed 0.85 | Sibling has confirmed reset+fumble (genuinely noisier); winner is clean — evidence correctly disfavors the sibling on cleanliness | Not observable | `classify_claim` scores winner CRITICAL/NEGATION, sibling SUPPORTING — WRONG side favored; confirmed to survive even the general ladder's own dominance check | BestTakeResolver (`_semantic_best_take` fast path AND, if reached, Steps 3/4) | Nothing usable was "ignored" — the representation itself misclassifies which side carries required content | NONE proven general; would require a claim-classification/semantic-sufficiency fix, explicitly out of this task's scope |
| Pimples | `clip_adcd41770...` | `clip_ecf64fd...` + `clip_d0291f...` | winner 0.9 vs alternate 0.6-0.9 (label unstable across chunks) | All 3 carry confirmed reset evidence; winner alone lacks a reported fumble score | Not observable | No CRITICAL claim anywhere in the family — every content check is silent for all 3 | BestTakeResolver fast path (unproven whether general ladder/DeliveryScorer would differ) | Possibly `local_selected_clip_id`/DeliveryScorer rank (never consulted by the fast path) — UNPROVEN, not surfaced in diagnostics | NONE proven; two live, separate, unproven hypotheses (DeliveryScorer bypass vs. family-granularity) |

---

## 9. Recommendation

**No common general rule is proven by both families. Per the task's own
recommendation rule, this means: do not implement a shared fix, and
treat the two root causes as separate.**

- Papillary needs a semantic-sufficiency/claim-representation fix (NOT a
  BestTake gating rule) — out of this task's scope, and itself risky
  (touches `semantic_claims.py`'s classifier, used by every other
  CRITICAL_COVERAGE_DOMINANCE caller in the codebase).
- Pimples needs ONE more piece of evidence before any fix can be
  proposed at all: whether `local_selected_clip_id` (DeliveryScorer's own
  rank, informed by real `MediaSignals`) already agrees with the model's
  "winner" label for this exact family. That value is not printed to the
  CI-safe diagnostics today. The MINIMUM next investigation (not
  authorized by this document) is to add this one field to the existing
  printed diagnostics — a pure observability addition, not a selection
  change — so a future RAW can answer this without a code change to any
  selection authority.

---

## 10. Confirmations

- **NO CODE CHANGED.** `pipeline.py`, prompts, markers, grouping,
  scoring, and Resolver are all byte-identical to `dabdd28`. This
  document is the only file this investigation produced.
- **NO RAW / PROVIDER / INFRA.** No Video00 RAW was launched, no paid
  provider was called, no AttemptReconstructor or D-099 Gap #2/#3 work
  was begun.

# Forensic: Hereditary-Cancer / Papillary-Diagnosis Cluster (D-101)

**Status: FORENSIC INVESTIGATION ONLY. No engine behavior was changed to
produce this document.**
**Date:** 2026-09-06. **Head at time of writing:** `808317b`
(`feature/runpod-pod-on-demand`), unchanged by this task.

Source evidence: persisted GitHub Actions job logs already fetched in this
session for the two RAWs compared in the D-100 real-media qualification —
**AFTER** = run `34062384187` (`video00-modal-34062384187-1`, head `808317b`,
D-100/D-097.12 present) and **BEFORE** = run `34048444463`
(`video00-modal-34048444463-1`, head `3c8ec4a`, pre-D-097.12/D-100). No RAW
was re-run, no ASR was re-run, no provider was called to produce this
document.

---

## 1. Cluster attempts / realizations

All texts below are byte-identical between BEFORE and AFTER (only the
`clip_*` ids differ — confirming ASR/segmentation content itself is stable
across these two runs for this cluster):

| Role | Text | AFTER clip id | BEFORE clip id |
|---|---|---|---|
| Hook | "tenía cáncer de tiroides y no lo sabía." | `clip_1d7db01e03883c6df3fc` | `clip_026cbbf4ed3db2260bf1` |
| Sonography lead-in | "Nunca se nos ocurrió hacer un chequeo..." | `clip_ab106867e22483b60bfa` / `clip_a134980c9ae59ea36024` | (same-shape pair, restart-evidence merged both runs) |
| Nodule/biopsy referral | "En la sonografía de tiroides apareció un nódulo sospechoso de 3 centímetros que se mandó a biopsia." | `clip_49454354b80990eef604` | equivalent |
| **Biopsy-confirmed diagnosis (part 1)** | "La biopsia confirmó que era un cáncer papilar de tiroides." | `clip_6d2369995806839d8d0e` | `clip_e087dda01ddfbd6292c5` |
| **Diagnosis continuation (part 2, required fused with part 1)** | "Síntomas que tuve según yo era sintomática pero si hubo indicios ahora mirándose atrás." | `clip_291ab9c0b4058cded4ab` | `clip_f80044afc6bfa095dbeb` |
| Competing "other symptoms" clip | "Síntomas que no me parecían sospechosos pero que ahora que lo analizo si eran sospechosos." | `clip_5f582a14bb23820e8211` | `clip_6d5b3471c16e8df90f67` |
| **Family-context (full, required)** | "Esta es mi experiencia. Soy la única en mi familia que tiene este tipo de cáncer. Por eso no creo y está comprobado científicamente que los cánceres son hereditarios. Más bien solo un 5-10% son de carácter hereditario. Mayormente son nuestras elecciones de vida. Así que cuídate." | `clip_c85136385805aed517d5` | `clip_a56462ca5c3553398c09` |
| Family-context (short prefix) | "Soy la única en mi familia que tiene este tipo de cáncer." | `clip_28e81abafe6a4e3c06a2` | `clip_a838c5c16e6051c84d31` |
| "Cancers are hereditary" fragment | "cánceres son hereditarios." | `clip_7f0d45c541f930303448` | `clip_20f00073fa74d89eb1ce` |
| Contradicting/aside clip | "Soy la primera en mi familia con este tipo de cáncer. Nadie en mi familia tiene un carcinoma papilar en la tiroides ni sufre de la tiroides." | `clip_539e755805f7255a7182` | `clip_b939462e3cd7c03e8e74` |

Required exact texts (`benchmarks/video00_regression_qa.json`):
- `biopsy_nodule_preserved` — nodule/biopsy referral sentence (PASSED both runs).
- `papillary_cancer_preserved` — **the two-sentence compound**: "La biopsia
  confirmó que era un cáncer papilar de tiroides. Síntomas que tuve según yo
  era sintomática pero si hubo indicios ahora mirándose atrás." (one
  `required_exact` unit spanning parts 1+2 above).
- `family_context_preserved` — the full family-context passage above.

Attempt reconstruction never fuses parts 1 and 2 into one attempt on
either run (`clip_6d2369995806839d8d0e`/`clip_e087dda01ddfbd6292c5` and
`clip_291ab9c0b4058cded4ab`/`clip_f80044afc6bfa095dbeb` are always separate
clips) — this by itself is not fatal (the required check tolerates
adjacent clips being both kept and adjacent in the render, per its own
`identity_rule`), but it does mean the two-sentence unit's survival
depends entirely on BOTH halves independently surviving Selection.

---

## 2. Papillary diagnosis: FIRST LOSS POINT

**Location:** `cutsell_worker/pipeline.py:238-247`,
`_semantic_best_take`'s `single_semantic_winner` fast path.

**Mechanism (identical on BOTH runs — pre-existing, reproducible):**
The Hybrid session-cleanup layer (`session_id` prefix `hc_...`, provider
`google`, model `gemini-3.5-flash-lite`) classifies the two-member retry
family {part-2 continuation, competing "other symptoms" clip} via
`family_window_labels`:

- AFTER (`group_id: tg_1e997db33bf157da33`): `clip_291ab9c0b4058cded4ab`
  ("Síntomas que tuve...", the REQUIRED continuation) → **"alternate", 0.85**;
  `clip_5f582a14bb23820e8211` ("Síntomas que no me parecían...", unrelated)
  → **"winner", 0.95**.
- BEFORE (`group_id: tg_c47c5cf87eae9c9950`): `clip_f80044afc6bfa095dbeb`
  (same required continuation) → **"failed"**; `clip_6d5b3471c16e8df90f67`
  (same unrelated clip) → **"winner", 0.95**.

Both runs: exactly ONE member is labeled "winner" at ≥0.85 confidence →
`_semantic_best_take` takes the `len(winners) == 1` branch and returns that
winner **unconditionally** (`"single_semantic_winner"`, pipeline.py:243-247)
— this branch never calls `resolve_critical_coverage_dominance`, never
checks `complete_idea`, never checks `any_pair_contradicts`, unlike the
`len(winners) != 1` branch a few lines below it (steps 1-5, lines 277-320).
The quality ladder independently confirms this is the wrong choice on both
runs (`authority: BestTakeResolver`, `refinement:
take_choice_against_both_references`, `"...while both references chose this
realization"` for the discarded continuation, and `"...is a realization
both references rejected"` for the kept "winner").

**First authoritative loss point:** `_semantic_best_take`'s
`single_semantic_winner` fast path, at the moment it trusts a lone
Gemini "winner" label with no claim/coverage cross-check. Everything
upstream (AttemptReconstructor, grouping into this 2-member family,
DeliveryScorer's own `ranked` scores) is not implicated — the family is
correctly formed on both runs; the wrong choice happens entirely inside
this one function's fast path.

**Stable, not new**: identical clip roles, identical "winner"/non-winner
labels' practical effect, identical ladder attribution on BOTH runs. Not
caused by D-097.12 or D-100.

---

## 3. Family context: FIRST LOSS POINT

**Location:** same function, the fallback ladder's final step
(`"delivery_tie_break_among_survivors"`, pipeline.py:322-331).

**Mechanism:** `family_window_labels` for `{clip_c85136385805aed517d5}`
(full family-context text) vs `{clip_28e81abafe6a4e3c06a2}` (short prefix):

- BEFORE (`group_id: tg_09db69f883b7a987cf`): full-text clip →
  **"winner", 0.9** → decisive single winner → `"single_semantic_winner"`
  → **correctly kept**.
- AFTER (`group_id: tg_59798c0f4f110f52c5`): full-text clip →
  **"failed", 0.85**; short-prefix clip → **"alternate", 0.7** → **no
  winner at ≥0.85** → falls through steps 1-5 (none of them decisive
  for this pair — neither is `semantic_delete_recommended`, neither has
  `complete_idea is False`, and `resolve_critical_coverage_dominance`
  apparently finds no dominant candidate between a full passage and its
  own strict prefix) → step 6 delivery-score tie-break picks the SHORT
  clip → **wrong realization kept, full passage discarded**.

Ladder confirms: `authority: BestTakeResolver`,
`refinement: take_choice_against_both_references`, `"candidate
clip_c85136385805aed517d5 lost family tg_59798c0f4f110f52c5 to
clip_28e81abafe6a4e3c06a2 while both references chose this realization"`.

**First authoritative loss point:** the SAME `_semantic_best_take`
function, but its opposite failure mode — here the semantic layer's own
per-window classification for this exact clip is **not stable between
calls on the identical text** (0.9 winner → 0.85 failed), and once it
stops being decisive, the delivery tie-break has no mechanism to prefer
the objectively more-complete realization (the full passage strictly
contains the short one's content plus the hereditary-context reasoning)
over its own truncated prefix.

---

## 4. FIRST BEFORE→AFTER DIVERGENCE

The grouping/family membership for **every** family in this cluster is
IDENTICAL between BEFORE and AFTER (same two-member pairs, same
`group_id`-equivalent shape, same restart-evidence/lexical merges). The
first and only divergence is the **Gemini `family_window_labels` output**
for retry-family `tg_09798...`/`tg_59798c0f4f110f52c5` (family-context
pair): `("winner", 0.9)` before → `("failed", 0.85)` / `("alternate",
0.7)` after, for byte-identical input text on both sides. This is an
LLM-classification-level divergence, not a code-path divergence — no
deterministic function between ASR and the Hybrid call differs in a way
that would explain it (D-097.12/D-100 do not touch this family: its
members are joined by ordinary lexical prefix overlap, unrelated to
either change).

---

## 5. Claim / meaning coverage trace

| Claim/atom | Importance | Where created | Where propagated | Where "considered covered" | Where lost |
|---|---|---|---|---|---|
| Biopsy-confirmed papillary diagnosis (part 1) | CRITICAL (required_exact) | ASR segment `clip_6d2369995806839d8d0e`/`clip_e087dda...` | Kept as its own attempt/clip through grouping, Freeze, render | N/A — this half always survives | Never lost by itself; the required check needs part 2 as well |
| Diagnosis continuation (part 2, "Síntomas que tuve... hubo indicios...") | Should be CRITICAL for `papillary_cancer_preserved` to pass, but **is not distinguished from the competing "other symptoms" clip anywhere in the traced pipeline** — no claim/atom mechanism was found treating part 2 as tied to part 1's diagnosis; it is handled purely as a delivery-ranking contest between two topically-similar "symptoms" sentences | ASR segment | Retry-family `tg_...`, `family_window_labels` | Gemini's per-window "winner" label (WRONGLY, on the unrelated clip) | `_semantic_best_take`'s `single_semantic_winner` fast path, both runs |
| Hereditary/family-cancer context (full passage) | CRITICAL (required_exact) | ASR segment `clip_c85136385805aed517d5`/`clip_a56462ca...` | Retry-family with its own short prefix | AFTER: Gemini "failed" label; delivery tie-break | `_semantic_best_take`'s delivery-tie-break step, AFTER run only |

**Verified answer to the example question** ("does 'carcinoma papilar' in
one realization cause the system to consider the full diagnosis/family-
history proposition safely covered when it is not?"): **No evidence of
this specific mechanism was found.** The contradicting/aside clip
(`clip_539e755805f7255a7182`, "Nadie en mi familia tiene un carcinoma
papilar...") never merges with or substitutes for either required passage
in either run's `restart_evidence_merges`/arbiter-merge lists — it stays
its own separate clip throughout. The actual loss mechanism in both cases
is the delivery/semantic-label ranking choosing the WRONG member of a
correctly-formed 2-member family, not a false claim-coverage
"substitution." No `claim_coverage_best_take`-attributed decision
(`critical_coverage_dominance`) was observed for either family in the
traced diagnostics — for the papillary pair it is never reached
(single-winner fast path bypasses it); for the family-context pair it is
reached but returns no dominant candidate, consistent with the atom/claim
model not registering "contains the hereditary-context reasoning vs.
doesn't" as a CRITICAL distinguishing fact between the two candidates.

---

## 6. Authority collision check

| Authority | Papillary pair | Family-context pair |
|---|---|---|
| AttemptReconstructor | Correct (kept as 2 separate legitimate clips both times) | Correct |
| Grouping (IdeaClusterer) | Correct (2-member family formed both runs) | Correct (2-member family formed both runs) |
| Hybrid semantic labeling (Gemini) | **WRONG, both runs** ("winner" on unrelated clip) | **WRONG on AFTER only** (label flipped from BEFORE) |
| `_semantic_best_take` (BestTakeResolver) | Trusts the wrong label unconditionally (fast path, no safety check) | Trusts fallback tie-break, no completeness-aware tie-break |
| ClaimCoverage | Never consulted (bypassed) | Consulted, found no dominant candidate (did not catch it) |
| StoryValidator | Does not flag either loss (neither check fires a contradiction/coverage-loss signal in the traced diagnostics) | Same |
| Freeze | PASS both runs (does not see this as a defect) | PASS both runs |

**FIRST CORRECT DECISION:** AttemptReconstructor + Grouping (both pairs,
both runs) — the right two candidates are correctly identified as one
retry-family contest.
**FIRST WRONG DECISION:** the Hybrid/Gemini per-window semantic label
(wrong content on the papillary pair; unstable/wrong on the family-context
pair, AFTER only).
**LATER OVERRIDE / MISSED SAFETY NET:** `_semantic_best_take` in
`pipeline.py` — for the papillary pair, its single-winner fast path
propagates the wrong label with zero cross-check; for the family-context
pair, its fallback tie-break has no signal that would have caught "one
candidate is a strict truncation of the other" as the wrong kind of tie to
resolve by raw delivery score. ClaimCoverage and StoryValidator, the two
authorities positioned to catch exactly this kind of loss, are either
bypassed entirely (papillary) or consulted but not decisive
(family-context) — this is the D-096 "later authority reverses/misses a
correct upstream decision, with no safety net downstream" pattern.

---

## 7. Root causes

**ROOT CAUSE #1 (papillary continuation, pre-existing on both runs):**
`_semantic_best_take`'s `single_semantic_winner` fast path
(`pipeline.py:243-247`) accepts a single Gemini "winner" label
unconditionally, with none of the safety checks (delete-recommended
evidence, completeness, CRITICAL_COVERAGE_DOMINANCE, contradiction) the
function's own `len(winners) != 1` branch already has. Gemini itself
misjudges this specific pair on both observed runs.

**ROOT CAUSE #2 (family context, new divergence on the AFTER run):**
The same function's `delivery_tie_break_among_survivors` fallback has no
mechanism to prefer a more-complete realization over its own strict
prefix once the semantic label stops being decisive; the immediate
trigger this run was the Gemini per-window label for this specific
clip flipping from a decisive "winner" (0.9) to a non-decisive
"failed"/"alternate" pair — genuine LLM run-to-run classification
variance on identical input text.

---

## 8. Whether D-097.12 contributed

**No.** D-097.12 (`incomplete_attempt_completed_by_retry`) never touches
either family in this cluster — both are ordinary lexical/prefix-overlap
groupings, unchanged in shape and membership on both runs. D-097.12's own
merge in this RAW fired on the unrelated stomach-family pair
(`clip_033021f84502b852e088`/`clip_c35264cdcfdceb82d2ca`).

## 9. Whether D-100 contributed

**No.** D-100 (`multimodal_corroborated_retry`) never activated anywhere
on this RAW (see the D-100 qualification report) and, even if it had,
only affects whether two takes MERGE into one retry family before the
arbiter — it has no code path that touches `_semantic_best_take`'s
winner-selection logic once a family already exists.

## 10. Whether ASR / arbiter variance contributed

**Yes, for Root Cause #2 specifically.** The ASR/segmentation content
itself is stable (byte-identical clip texts on both runs). The variance is
narrower and more specific: the Hybrid Gemini per-window classification
label for one specific clip pair changed between calls on identical text.
Root Cause #1 is NOT attributable to variance — it reproduced identically
on both runs.

---

## 11. Minimum safe fix location (NOT implemented by this task)

Two independent, narrow candidates, named for a future authorized cycle:

1. For Root Cause #1: extend `_semantic_best_take`'s `single_semantic_winner`
   branch (pipeline.py:243-247) to run the SAME lightweight safety checks
   (D-081 delete-recommended, completeness, CRITICAL_COVERAGE_DOMINANCE,
   contradiction) the multi-candidate branch already has, before trusting
   a lone "winner" label — reusing existing functions, not new authority.
2. For Root Cause #2: give `delivery_tie_break_among_survivors` (or a step
   before it) a cheap, deterministic "is one survivor a strict content
   subset/prefix of the other" check, preferring the more-complete
   realization when claim-coverage itself returns a tie — again reusing
   existing text/content-overlap primitives already in this codebase
   (e.g. `take_grouping._restart_content`-style token containment).

## 12. Expected effect

Fixing #1 would very likely restore `papillary_cancer_preserved` (it
fails identically and deterministically for the reason found here).
Fixing #2 would likely restore `family_context_preserved` on runs where
Gemini's label is non-decisive, though since the label itself is a
run-to-run variable, a single future RAW could not fully prove the fix
without observing the same non-decisive-label condition recur.

## 13. Risks / regressions to protect

- Any change to `_semantic_best_take` must preserve every existing
  D-081/D-082/D-097.B/D-097.8 behavior this function's docstring already
  documents (all cited in Section 2/3 above) — this is a heavily-layered,
  historically fragile function; a narrow, additive check is essential,
  not a rewrite.
- Must not weaken the fail-open "WHEN UNCERTAIN, KEEP" doctrine — any new
  gate should only ADD a safety check before trusting a label, never
  introduce a new deletion path.
- Must not reintroduce a stale completeness/coverage heuristic that
  contradicts D-063's own CRITICAL_COVERAGE_DOMINANCE safety gates
  (never override a real contradiction, never prefer a proven-incomplete
  candidate).

---

## 14. Confirmations

- **NO ENGINE BEHAVIOR CHANGED.** Only this forensic document and its
  decision-log entry were added; no `cutsell_worker/*.py` file was
  modified.
- **NO RAW / PROVIDER / S3 / INFRA WORK.** No RAW was launched, no ASR was
  re-run, no provider was called, no infrastructure was touched. All
  evidence above comes from the two GitHub Actions job logs already
  fetched and decoded earlier in this session for the D-100 qualification
  report.

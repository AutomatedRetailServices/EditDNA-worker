# D-127 — Bounded Multimodal Fallback Arbiter: Forensic + Design

**Status: FORENSIC + DESIGN ONLY. No fallback implemented. No provider call made.
No RAW launched. No engine behavior, score, ranking, winner, grouping, or
Boundary logic changed by this document or the task that produced it.**

Companion to `docs/CUTSELL_BESTTAKE_CASE_B_FORENSIC_D121.md` (D-121) and
`docs/CUTSELL_DECISIONS.md` D-111/D-122/D-123/D-124/D-126. Read those first —
this document assumes their findings and does not re-derive them.

---

## 0. Why this task exists

D-126 (RAW 34172575066) proved D-123's structured, disagreement-gated CASE B
mechanism works correctly on real footage: one real bypass fired correctly
(`tg_ef754f8f610ab360df`), one real meaning-insufficiency block fired
correctly (`tg_7e13303b51e7b07a5f`), and every trivial-agreement family was
correctly left untouched. D-123 is closed for the RAW→Cut.ai milestone.

But D-126 also exposed, in the SAME run, a shape D-123 is not designed to
catch: the pimples family (`tg_dfa8f59296237ae030`) where the semantic
winner and DeliveryScorer's own top pick **agreed** — and D-123 therefore
correctly never engaged — yet the QA reference (`pimples_bad_monolith_
absent`) still failed, and the agreed-upon winner's own D-122 CASE B
evidence (9 delivery events, 0.6s) was **worse** than the non-selected
meaning-sufficient alternative's (7 events, 0.467s). Two independent
structured signals agreed on a candidate that a third, already-computed
structured signal argues against. This is the general shape D-111 named as
"MEDIUM confidence / real evidence conflict" territory for a bounded
multimodal arbiter — this document works out what that would actually mean,
without building it.

---

## 1. Current confidence / conflict signal audit

| Signal | REAL? | Observable today? | Currently used? | Calibrated? | Can indicate conflict? | Safe as fallback trigger? | Why / why not |
|---|---|---|---|---|---|---|---|
| Semantic winner label + confidence (Hybrid/Gemini `family_window_labels`) | Yes | Yes (`take_judge_groups[].semantic_candidates`) | Yes — drives the `single_semantic_winner` fast path | Confidence is a raw LLM score, never independently validated against outcome | Yes — a low-margin winner (e.g. 0.85 vs a 0.80 "alternate") is weaker evidence than 0.97 vs 0.60 | **Structural, not yet safe alone** — the raw number has no proven calibration; using it as a numeric cutoff would be inventing a threshold (forbidden here) | Real and observable, but "how low is too low" is a calibration question, not a structural fact |
| Semantic labels across family members (label diversity: single winner vs 0/2+ winners vs mixed) | Yes | Yes | Yes (`_semantic_best_take`'s own branch selection) | N/A — categorical, not a score | Yes — 0 or 2+ "winner" labels already means the ladder itself treats the family as non-decisive | **Yes, structurally** — this is exactly what the EXISTING ladder already gates on; no new signal needed |
| DeliveryScorer ranking (`ranked`, `take_judge.rank_takes`) | Yes | Yes | Yes (D-082 tie-break, D-123's `deliveryscore_top_candidate`) | Weighted sum of ~12 terms (`score_take`), internally consistent but never independently validated against Human Gold at the per-decision level | Yes — a disagreement between this ranking's top pick and the semantic winner is D-123's own trigger | Yes, already used safely by D-123 | Already the core of D-123; nothing new to add here |
| DeliveryScorer score GAP (numeric margin between top two) | Yes (derivable from `ranked`) | Yes, but not currently surfaced in any diagnostics field | No | No | Could indicate a close call | **Requires calibration** — "how close is close" is a threshold question this task explicitly forbids inventing |
| D-122 CASE B event asymmetry (`delivery_event_count`, `count_by_kind`, `duration_by_kind`) | Yes | Yes (D-125 summary) | Yes — D-123's own conflict-basis comparison, but ONLY when semantic/DeliveryScorer already disagree | Deliberately uncalibrated (D-123 uses strict `>`, never a magnitude cutoff) | **Yes — this is exactly the D-126 pimples finding**: asymmetry can exist even when the two upstream signals agree | **Structurally safe as a trigger SOURCE** (existence of asymmetry is a fact); using its magnitude to auto-decide a winner would need calibration | This is Trigger Class B's evidence source (Section 5) |
| D-097 cleanliness evidence (`delivery_cleanliness_evidence`, `_RESET_KINDS`/`_BREAK_KINDS`, `_CLEANLINESS_EDGE_MARGIN_SEC=0.35`) | Yes | Yes | Yes — folded into `MediaSignals.visual_fumble`/etc., which feeds `score_take` | Yes — has its own fixed confidence floors (`>=0.88` reset, `>=0.76` break) | Partially — it is ALREADY inside the DeliveryScorer number, so it cannot be a SECOND, independent vote (see Section 3) | Not independently — already counted once | Confirmed overlapping with CASE B (D-121/D-122's own documented double-counting risk) |
| MediaSignals (aggregate visual/audio scalars) | Yes | Yes | Yes — `score_take`'s direct inputs | Provider-scored (real, partial) or default (0.5/synthetic) per D-121's REAL/DEFAULT table | Same caveat as above — already inside DeliveryScorer | Not independently | Same double-counting concern |
| Meaning sufficiency (`_meaning_sufficient_member_ids`: D-081 delete-recommended, `complete_idea`, D-103 required-condition-realization) | Yes | Yes (`meaning_sufficient_candidates`) | Yes — D-123's own gate condition | N/A — categorical/boolean, deterministic | Yes — a candidate outside this set is not eligible for anything, fallback included | Yes, as a HARD FILTER (never a trigger by itself, but a mandatory precondition) | Already proven safe by D-123's own 23 tests |
| Critical coverage dominance (`resolve_critical_coverage_dominance`, D-063/D-065/D-066) | Yes | Yes (`final_reason`) | Yes — the ladder step that resolved D-126's own confirmed bypass | Deterministic (claim-set superset check, no numeric threshold) | Yes — when dominance finds NO single winner among survivors, that is itself evidence of unresolved conflict | Yes, structurally — "dominance found no winner" is a fact, not a score | Already a real ladder outcome (`no dominant_id`) |
| Deterministic overrides (`deterministic_best_take_authority`, `CLEAR_WINNER_MINIMUM_GAP=0.30`) | Yes | Yes (`winner_path=="DETERMINISTIC_OVERRIDE"`) | Yes | Yes — fixed 0.30 gap, an EXISTING calibrated threshold (not invented here) | Yes — this authority only fires on a large, clear gap; ABSENCE of an override on a family that otherwise looks contested is itself informative | Indirectly — "no override fired" is a fact, not a new number | Reuses an existing calibrated constant, never re-derives it |
| Semantic idea-equivalence confidence (`semantic_idea_equivalence.py`, D-058/D-085 grouping-safety) | Yes | Yes (`distinct_idea_grouping_safety` diagnostics) | Yes — upstream of family formation entirely | Confidence bands exist (`_BRIDGE_FLOOR`-style constants, per D-085) but this module is TEXT-ONLY (`IdeaEquivalencePair`, no clip_id/timestamp) | Yes, but at the GROUPING layer, not the BestTake layer — governs whether a family forms at all | N/A as a BestTake fallback trigger — this signal decides membership, which fallback must never touch (Section 11) | Confirms grouping stays fully upstream and separate |
| Retry-family confidence/evidence (restart detection: `same_opening_abandoned_start`, `incomplete_attempt_completed_by_retry`, deterministic confidence 1.0 vs arbiter-confirmed pairs) | Yes | Yes (`semantic_idea_equivalence` merge diagnostics) | Yes | Deterministic merges are confidence 1.0 by construction; arbiter-confirmed merges carry the arbiter's own confidence | Yes, at the grouping layer | N/A as a BestTake trigger (same reason as above) | — |
| Current winner path (`winner_path`, `winner_path_before/after`) | Yes | Yes (D-125) | Yes — D-123's own observability | N/A — categorical | Yes — `OTHER_EXISTING_PATH` after a bypass, or a winner-label displaced without a bypass flag, are both directly observable facts | Yes, structurally — this IS the fact a trigger reads, not a new signal | Core to Trigger Classes A/D |
| Whether performance was consulted (`performance_consulted_before_winner`) | Yes | Yes | Yes | N/A — boolean | Yes — `false` on a family with real disagreement (like D-126's `tg_ef754f8f610ab360df` BEFORE the bypass) is exactly what D-123 already gates | Yes | Already D-123's own precondition |
| Whether multiple authorities disagree (semantic vs DeliveryScorer vs critical-coverage-dominance vs deterministic-override) | Yes (derivable by cross-referencing the fields above) | Yes | Partially — D-123 checks semantic-vs-DeliveryScorer only | No | Yes — a THREE-way disagreement (all different picks) is a stronger signal than a two-way one, and is not currently distinguished from a two-way case | Structurally observable, but combining ">=2 disagreeing authorities" into a trigger CLASS (not a count threshold) is safe; picking a numeric "N authorities" cutoff would not be | See Trigger Class G |
| Whether one authority wins despite conflicting evidence (the D-126 pimples shape itself) | Yes | Yes, retrospectively (cross-reference CASE B evidence against the agreed winner) | **No — this is exactly the gap** | N/A | Yes — this is the central finding of this document | Structurally observable; not yet computed or surfaced anywhere | This is Trigger Class B, not yet built |

---

## 2. Agreement can still be low-confidence

**"Semantic winner == DeliveryScorer winner" does NOT always mean HIGH
confidence.** D-126's own pimples family is the direct counter-example:

- semantic label: "winner" 0.95 for `clip_51e6a8e265a375059ea9`
- DeliveryScorer top: the SAME clip
- D-122 CASE B evidence for that SAME clip: 9 delivery events (0.6s) —
  **more** than the non-selected sibling `clip_c61869580931be9ba985`'s 7
  events (0.467s)
- Human Gold regression QA: `pimples_bad_monolith_absent` still fails —
  the QA reference considers the agreed-upon winner the WRONG take.

Two signals "agreeing" is not two independent confirmations here — it is
one signal (Hybrid/Gemini's semantic label) and a second signal
(DeliveryScorer) that, per Section 3 below, is PARTLY DERIVED FROM THE SAME
underlying local-performance events CASE B itself reads. Agreement between
a label and a score built partly from the same raw signal the label may
also be implicitly sensitive to is weaker evidence than agreement between
two genuinely independent measurements. **The general lesson: "the two
existing gates didn't disagree" is necessary but not sufficient evidence
of correctness — it only proves the two gates didn't disagree with each
other, not that they are both right.**

---

## 3. Correlated-evidence audit

Per D-121 Section 9 (restated and extended here for the fallback design
question specifically):

| Pair | Overlap | Independent votes? |
|---|---|---|
| Hybrid/Gemini semantic nomination vs. DeliveryScorer | Semantic label reads the TRANSCRIPT (and, when Hybrid runs with vision, possibly frames); DeliveryScorer (`score_take`) reads `MediaSignals` (provider-scored visual/audio quality) plus deterministic text-shape penalties (`_handling_failure_penalty`, fragment detection). Text overlaps: both can respond to disfluency/hesitation visible in the transcript. | **Partially correlated**, not independent. A transcript-level flaw (e.g. a stumble word) can influence both a semantic re-read AND the deterministic fragment penalty. |
| DeliveryScorer vs. D-097 cleanliness evidence | D-097 cleanliness (`delivery_cleanliness_evidence`) is folded directly into `MediaSignals.visual_fumble`/`gesture_naturalness`/`distraction_risk`/`expression_naturalness` via `local_performance.apply_local_performance_to_takes`, which `score_take` then reads. | **NOT independent — same signal, already summed once.** This is D-121's already-confirmed double-counting risk. |
| DeliveryScorer vs. D-122 CASE B evidence | CASE B (`case_b_performance_evidence.py`) re-projects the EXACT SAME four `local_performance` event kinds D-097/MediaSignals already consumed, over a DIFFERENT window (D-115's DELIVERY-zone-only classification, vs. D-097's own `_CLEANLINESS_EDGE_MARGIN_SEC=0.35`-adjusted interior window). | **Same raw events, two different window derivations.** `MEDIASIGNALS_PROVENANCE`/`d097_would_be_counted` in `case_b_performance_evidence.py` already document this explicitly per-event. |
| MediaSignals vs. D-097 cleanliness | Same relationship as above — D-097 IS one of the two things that populates certain MediaSignals fields for these four event kinds. | Same signal. |
| Critical coverage dominance vs. everything else | Reads ONLY claim-coverage sets (`semantic_claims.py`/`claim_coverage_best_take.py`) — text-content-based, not performance-based. | **Genuinely independent** of DeliveryScorer/CASE B/MediaSignals — a real second vote when it fires. |
| Semantic label vs. Critical coverage dominance | Semantic label is a single LLM judgment over the whole family; dominance is a deterministic claim-set comparison. | **Genuinely independent** in mechanism, though both ultimately read the same transcript text. |

**Conclusion for fallback design:** DeliveryScorer, D-097 cleanliness, and
D-122 CASE B evidence are NOT three independent votes — they are three
views (one summed score, one now-superseded intermediate, one re-projected
positional detail) of the SAME four `local_performance` event streams.
"Semantic label agrees with DeliveryScorer" is really "one LLM judgment
agrees with one performance-derived score" — exactly two votes, one of
which (DeliveryScorer) is itself an amalgam. A future arbiter must not
treat DeliveryScorer-and-CASE-B agreeing with each other as a THIRD
confirmation — they are the same evidence measured twice. **Critical
coverage dominance is the one genuinely independent existing structured
signal.** This materially bounds how much "multiple authorities agree"
can be trusted as evidence of confidence.

---

## 4. Fallback trigger classes (general, not numeric)

| Class | Description | General? | Evidence available today? | Needs new provider? | Over-trigger risk | Under-trigger risk | Cut.ai-milestone status |
|---|---|---|---|---|---|---|---|
| **A** | Semantic winner != DeliveryScorer winner AND structured system (D-123 + the general ladder) remains unresolved after running | Yes | Yes — this is D-123's OWN scope; by the time D-123's ladder finishes, this class either resolved (dominance/tie-break decided) or degenerated into `unresolved_unique_fact_asymmetry`/`unresolved_contradiction` (both ALREADY safe, ALREADY preserve rather than guess) | No | Low — D-123 already proved this fires correctly and rarely (1/8 families in D-126) | Low — the two "unresolved" ladder outcomes already fail safe (never a bad automatic answer) | **Already active via D-123; not a new fallback trigger.** The only open question is whether the two "unresolved" terminal states above should ESCALATE to fallback instead of silently keeping `local_selected_clip_id` — see Section 21 Phase 3 candidate. |
| **B** | Semantic winner == DeliveryScorer winner, but D-122 CASE B evidence for a DIFFERENT meaning-sufficient family member shows a real, strict asymmetry favoring it over the agreed winner | Yes — this is the general form of the D-126 pimples finding, phrased with no pimples-specific text | Yes, fully — `case_b_evidence` is already computed for every family regardless of agreement (D-125 confirms `families_with_case_b_evidence_count` covers ALL 8 families in D-126, not just the disagreement ones) | No | **Real, needs attention**: this trigger would have fired on the pimples family in D-126, which is the case it's designed for — but has NOT been validated against negative controls yet (Section 20) | Real, if left inactive: this is the exact gap D-126 exposed | **Requires calibration to decide "strict asymmetry" magnitude for a 2-member family** (the D-123 CORE RULE's own strict-`>`-no-threshold precedent is directly reusable structurally, but this is a genuinely new comparison shape D-123 never made — proposing to reuse D-123's own comparator function, not invent a new number) — candidate for Phase 1 shadow/Phase 2 eval, not Phase 3 activation without eval proof |
| **C** | DeliveryScorer/semantic winner has stronger completeness but visibly worse performance, while a meaning-sufficient alternative exists | Partially — this reads as a SUBSET of B (Class B already covers "the agreed winner is worse on CASE B than a sufficient alternative"); the "stronger completeness" qualifier adds a claim-coverage dimension not yet cross-referenced against CASE B anywhere | Partial — completeness (`complete_idea`) and coverage (critical-coverage sets) both exist, but no code currently cross-tabulates "wins on completeness AND loses on CASE B" | No | Unknown — never observed or tested | Unknown | **Defer** — this is Class B plus an unproven additional condition; adding it now would be inventing a compound rule not yet evidenced by any real run |
| **D** | Winner path changes repeatedly across runs/evals for the same family SHAPE | Yes, as a category | Yes, but only in eval memory (Section 14) — a single production run has no access to "other runs" | No | N/A for production (see below) | N/A for production | **Eval/calibration signal only — never a production-time trigger** (Section 14 formalizes this) |
| **E** | Semantic-equivalence confidence high (family formed cleanly) but the AUTHORITY outputs (label vs. dominance vs. delivery) still conflict | Yes | Partially — equivalence confidence exists at grouping time (Section 1), but is not currently threaded into `take_judge_groups` rows for cross-reference at BestTake time | No | Unknown | Unknown | **Defer — requires new wiring (equivalence confidence -> BestTake row) before it is even observable**, which is itself a scope decision, not a trigger decision |
| **F** | BestTake vs. Boundary ownership ambiguous (a candidate's own defect sits ambiguously between DELIVERY interior and an EXIT/ENTRY edge) | Yes | Partially — D-115's `classify_event_zone`'s explicit `straddle` flag (`starts_before_delivery`/`ends_after_delivery`) already records this per event; D-116 owns ENTRY/EXIT, BestTake/CASE B owns DELIVERY-or-straddling | No | Unknown — no real run has yet shown this specific ambiguity causing a wrong outcome | Unknown | **Defer** — D-116/D-121's own doctrine already routes straddling events to the DELIVERY side (never ambiguous by construction, per D-115's classification rule: any overlap = DELIVERY); no evidence this needs a NEW trigger beyond what CASE B (part of Class B) already sees |
| **G** | Structured evidence internally contradictory (3+ authorities each prefer a different candidate, or dominance actively finds a contradiction) | Yes | Yes — `unresolved_contradiction`/`unresolved_unique_fact_asymmetry` `final_reason` values already name this exact state | No | Low — these are RARE terminal ladder states that already fail safe (preserve `local_selected_clip_id`, never a bad automatic answer) | Low, since the current fallback behavior (preserve, don't guess) is already the correct safe default D-111's LOW-confidence tier calls for | **Already safely handled by existing WHEN-UNCERTAIN-KEEP doctrine; escalating to a bounded arbiter INSTEAD of silently preserving is a legitimate future Phase 3 candidate, not urgent** |

**Only Class B is authorized to advance past this document** (Phase 1
shadow evaluation, Section 21) — it is the one class with (1) a general,
non-pimples-specific formulation, (2) evidence available today with zero
new wiring, (3) a real, already-observed positive instance (D-126), and
(4) a bounded, well-understood comparator (D-123's own CASE B aggregate
comparison, structurally reused, not re-derived). Classes A/G are already
handled safely by existing machinery. Classes C/D/E/F are deferred pending
either more evidence or additional wiring this task does not authorize.

---

## 5. No threshold invention — structural vs. calibrated

**Structural triggers (usable today, no calibration needed):**
- Class A's own precondition (semantic label non-decisive OR vetoed) — already boolean/categorical.
- Class B's EXISTENCE check ("does a real, strict CASE B asymmetry exist favoring a different meaning-sufficient member than the agreed winner?") — reuses D-123's own strict-`>` comparator, never a magnitude cutoff.
- Class G's terminal ladder states (`unresolved_contradiction`, `unresolved_unique_fact_asymmetry`) — already named, categorical.
- Meaning sufficiency as a hard filter — already boolean.

**Calibrated triggers (would require measurement before use, NOT decided here):**
- Semantic winner confidence cutoff (how low is "uncertain enough").
- DeliveryScorer score-gap threshold (how close is "a close call").
- CASE B event-count/duration MAGNITUDE threshold beyond mere existence of asymmetry (e.g. "asymmetry must exceed N events" — D-123 itself never does this, and this document does not propose it either).
- Any cross-run winner-instability frequency cutoff (Class D, explicitly eval-only per Section 14).

No numeric value for any of the above is proposed, estimated, or implied
anywhere in this document.

---

## 6. Fallback input contract (minimum bounded)

Per this task's own required shape, and cross-checked against what the
codebase can already assemble without new computation:

```
MultimodalArbiterRequest:
  family_id: str                          # take_judge_groups[].group_id
  finalists: tuple[FinalistInput, ...]     # 2-3 members, MEANING-SUFFICIENT ONLY
    FinalistInput:
      clip_id: str
      source_asset_id: str
      source_span: (start: float, end: float)   # already on CandidateTake
      words: tuple[Word, ...]                    # already-aligned transcript
      complete_idea: bool | None
      case_b_evidence: CaseBPerformanceEvidence   # already computed, D-122
      delivery_cleanliness: dict                  # already computed, D-097 (delivery_cleanliness_evidence)
  proposition_context: str                 # the family's shared idea/topic text only -- never full-video summary unless proven necessary
  meaning_sufficiency: dict[clip_id, bool]  # already computed, D-123
  deliveryscore_summary: dict[clip_id, float]  # existing RankedTake.score values ONLY, never re-derived
  boundary_editability: dict[clip_id, EditabilityNote] | None  # OPTIONAL, only for the GOOD_TAKE_TRIM_* outputs (Section 8) -- straddle flags from D-115, never raw timestamps to edit
```

This is a strict SUBSET of what `case_b_performance_evidence.py` and
`_case_b_fast_path_conflict` already assemble per family today — no new
computation, only a new (unbuilt) packaging step. **Full RAW is never
sent** — only the finalists' own source spans (already bounded to
single-take duration, seconds not minutes).

---

## 7. Multimodal requirement — what's actually possible today

The directive requires the arbiter to genuinely SEE + HEAR the finalists,
not run another text-only semantic call. Auditing what exists:

- **SEE (frames): REAL, PROVEN, ALREADY IN PRODUCTION.** `frame_sampling.py`'s
  `sample_take_frames`/`adaptive_frame_count` extract real JPEG frames via
  ffmpeg from the actual source media at real timestamps (`FrameSample`);
  `visual_analysis.py`'s `VisualProvider` protocol and `visual_openai.py`'s
  `OpenAIVisualProvider` already send these as base64 `image_url` inputs to
  a vision-capable model (gpt-4o-mini), batched (`batch_size=6`) for cost
  control, currently for Watch+Listen's per-clip independent visual scoring
  (`VisualObservation`). **This exact frame-extraction + vision-provider
  pipeline could be reused for a multimodal arbiter's visual input with no
  new infrastructure** — the NEW work is the request/response SHAPE (a
  comparative 2-3-way verdict, not independent per-clip scores) and the
  scope (BestTake-triggered, not Watch+Listen-triggered).
- **HEAR (audio): NOT CURRENTLY REAL in this sense.** No module in this
  repository sends raw audio bytes/waveform to any provider. "Hearing" is
  currently represented ENTIRELY through ASR transcript text (`Word`
  timing) and deterministic signal detection (`audio_silence.py`'s
  ffmpeg `silencedetect`, never a model call). A genuinely audio-aware
  arbiter (tone, vocal hesitation quality, prosody) would require NEW
  provider integration work this document does not authorize and has not
  designed. **This must be stated honestly, not assumed solved by
  extension of the existing transcript pipeline.**
- **TRANSCRIPT: REAL, already the primary signal throughout the pipeline
  (`CandidateTake.words`/`.text`).**

**Conclusion:** a Phase 1 arbiter reusing the existing frame + transcript
pipeline is technically buildable today without new provider integration
work beyond request/response shaping. A genuinely audio-perceptive
arbiter is future integration work, not available now.

---

## 8. Fallback output vocabulary

| Outcome | Meaning | Cut.ai-milestone appropriate? |
|---|---|---|
| `BEST_TAKE_A` / `BEST_TAKE_B` / `BEST_TAKE_C` | One named finalist is clearly better | Yes — the core useful outcome |
| `EQUIVALENT` | Finalists are genuinely interchangeable | Yes — must map to "preserve existing structured pick," never a coin flip |
| `KEEP_BOTH_COMPLEMENTARY` | Finalists carry distinct, non-redundant content | **No, not for Cut.ai milestone** — this is a membership/composite decision (D-019 explicitly puts SWAP/multi-realization inventory OUT OF SCOPE for Clean Cut V1; a fallback arbiter must not reintroduce it) |
| `GOOD_TAKE_TRIM_ENTRY` / `GOOD_TAKE_TRIM_EXIT` | The chosen take is good but has a removable ENTRY/EXIT defect | Yes, AS ADVISORY EVIDENCE ONLY (Section 9/12) — the arbiter names the observation, Boundary remains the sole trim authority and sole timestamp editor |
| `UNCERTAIN` | Arbiter itself cannot resolve it | Yes — MUST remain a first-class, expected, non-error outcome (Section 15) |

`KEEP_BOTH_COMPLEMENTARY` is explicitly excluded from the Cut.ai-milestone
vocabulary per current scope doctrine (D-019/D-020) — including it would
smuggle SWAP-era multi-realization inventory back into an active-path
authority, which is out of scope until the Product Owner explicitly
reintroduces SWAP.

**The arbiter must never invent speech or manufacture a composite** — every
allowed outcome either names an EXISTING finalist verbatim or defers.

---

## 9. Authority contract

**Doctrine: fallback may resolve ONLY a pre-certified ambiguity among
already-legitimate, already-meaning-sufficient finalists.** It may not:
introduce a new candidate; merge propositions; override meaning
insufficiency; override polarity/diagnosis/number safety (the existing
D-063/D-066/D-101/D-103 safety layers); override an exact deterministic
rejection; modify grouping; modify source; invent speech.

**Recommendation: (B) advisory evidence consumed by an EXISTING resolver —
never a standalone final authority.** Rationale:

1. Precedent: EVERY existing semantic authority in this codebase that
   touches BestTake is advisory-consumed-by-the-ladder, never a bare
   terminal decision-maker on its own: the semantic label itself only
   short-circuits via `single_semantic_winner`, which the general ladder
   (dominance, tie-break) can still override on veto; `deterministic_
   best_take_authority` is a LATER, separate authority layered on top,
   not a replacement. A brand-new arbiter output becoming an unconditional
   terminal authority would be the FIRST such mechanism in the codebase —
   a bigger architectural step than this task authorizes considering.
2. D-091/D-095 doctrine ("CODE FIXED != TESTS PASS != ... != HUMAN GOLD
   PARITY") and D-111's own framing ("never over a stronger deterministic
   answer") both argue for the newest, least-validated signal entering as
   evidence a resolver weighs, not as an unchallengeable verdict.
3. Practically: routing the arbiter's output through the EXISTING general
   ladder (as one more input `_case_b_fast_path_conflict`-style comparator
   could consult, or as a new, separately-gated step placed analogously)
   means every existing safety veto (D-101/D-103, meaning sufficiency,
   contradiction detection) automatically still applies to its output —
   building it as a standalone terminal authority would require
   re-implementing all of those checks a second time or risk bypassing
   them.

---

## 10. Fallback vs. D-123 order (confirmed from code)

Directive's proposed order is **confirmed correct** against the actual
code path in `pipeline.py`/`take_judge.py`:

```
legitimate competitors (IdeaClusterer/grouping — upstream, D-121 §2)
  -> meaning sufficiency (_meaning_sufficient_member_ids, D-081/D-103)
  -> semantic / DeliveryScorer / CASE B structured reasoning
     (semantic label decisiveness check, then D-123's
      _case_b_fast_path_conflict gate on the single_semantic_winner branch)
  -> D-123 gate (bypass or preserve fast path)
  -> existing BestTake resolution (the general ladder: D-081 exclusion,
     D-103 exclusion, D-063/D-065/D-066 critical_coverage_dominance,
     D-101 unique-fact/contradiction safety, D-082 delivery tie-break)
  -> ONLY IF a Class-B-shaped conflict remains (agreement occurred but
     CASE B still disagrees) OR the ladder terminates in an unresolved
     state (Class A/G): bounded fallback (NOT YET BUILT)
```

D-123 remains the cheaper, already-proven, zero-provider-cost first-line
gate; the bounded arbiter would only ever engage on the SMALL residual set
of families where the fully-structured pipeline above still leaves a real,
evidenced conflict unresolved (in D-126: at most 1 additional family out of
8, since 6 of 8 needed no fallback consideration at all under Class B's own
condition).

---

## 11. Fallback vs. grouping

**Fallback must never repair a missing competitor set.** Confirmed: D-121
Section 2 already established competitor sets form ONCE, upstream, via
`IdeaClusterer`/`take_grouping_provider.reconcile_semantic_idea_equivalence`
/`split_incohesive_retry_groups` — DeliveryScorer/BestTake (and therefore
any future fallback layered at the SAME point) never add members. D-124's
own pimples fragmentation (two separate families, `tg_e51a80f62131206cf2`
and `tg_2351202cafbababad4`) is a directly-observed real instance: no
family-level arbiter, bounded or not, could have "fixed" that split,
because there was no single 3-member family for it to be handed — the
split happened entirely upstream, in `distinct_idea_grouping_safety`'s own
rejected-edge/content-divergence-blocked decisions. **Recorded explicitly:
grouping remains fully upstream and out of any fallback's reach, by
construction.**

---

## 12. Fallback vs. Boundary

D-116's doctrine preserved exactly: ENTRY/EXIT-only defects remain
Boundary's exclusive territory; a DELIVERY-overlap (or straddling) defect
is BestTake/fallback's to consider (D-115's `classify_event_zone`: any
overlap classifies DELIVERY). A fallback MAY return `GOOD_TAKE_TRIM_ENTRY`/
`GOOD_TAKE_TRIM_EXIT` as an OBSERVATION — naming that a good take has a
removable edge defect — but this is advisory input to Boundary, never a
direct edit: **fallback never writes a timestamp.** Boundary
(`boundary_engine_pass.py`) remains the sole physical-trim authority,
exactly as D-097.C/E's "physical ownership contract" already establishes
for every other upstream signal.

---

## 13. Pimples cross-run forensic (eval case only — not a pimples-only trigger)

| Run | Family shape | Semantic winner | DeliveryScorer top | CASE B evidence | Final winner path | QA outcome | Would Trigger B have applied? |
|---|---|---|---|---|---|---|---|
| D-113 (34143240479) | Never grouped (upstream grouping variance, category A per D-121) | N/A — no family existed | N/A | N/A | N/A — singleton(s), no contest | Level-1 = 0s (coincidentally clean — no family to get wrong) | **No** — no family ever formed; Section 11 applies (Trigger B needs a family to evaluate) |
| D-116-first / D-118 (34150026795 / 34160335330) | Formed: `tg_72535925d777047e70`, 2 members | `clip_6cc1223155bdc96db3b8` ("winner" 0.95, `dense_physical_reset:7`) | Not recorded in pre-D-122 diagnostics (field didn't exist) | Pre-D-122: only D-097-era local-failure-reason strings recorded, not D-122's structured `case_b_evidence` | `single_semantic_winner` (pre-D-123; no gate existed) | Level-1 ≈17.4-19.7s (regression traced to this exact bypass, per D-121) | **Cannot be retroactively confirmed** — the OLD diagnostics shape lacks `deliveryscore_top_candidate`/structured CASE B fields needed to evaluate Class B's own condition; the LOSING alternate is recorded as carrying MORE distinct local-failure reasons (`dense_physical_reset:7` + `visual_fumble:0.85`) than the winner (`dense_physical_reset:7` alone) — if anything this suggests the semantic label picked the CLEANER one by these old signals, so this specific historical instance may not even BE a Class-B shape; genuinely uncertain from available evidence |
| D-120 (34162868778) | Formed correctly | Not individually retrieved (D-120's own report focused on the aggregate D-116 verdict, not per-family pimples detail) | — | — | — | 17/18 checks pass; only `sonography_good_before_diagnosis` fails (pimples checks ALL pass) | **No** — nothing to trigger on; this is a clean run |
| D-124 (34169540283) | **Fragmented into 2 families**: `tg_e51a80f62131206cf2` (0 winners, both "failed") + `tg_2351202cafbababad4` (1 winner) | `clip_5b2a548be7f2e10d0ff2` (0.92) for the 2nd family | Same clip (no disagreement recorded in that family's own row) | Not retrievable (pre-D-125; tail-window gap) | `single_semantic_winner`, unbypassed (D-123 was active but found no conflict, per the family's own final reason) | 3 of 5 pimples checks fail (`pimples_micro_2_present`, `pimples_bad_monolith_absent`, `pimples_micro_order`) | **Cannot fully evaluate** — CASE B evidence for this family is not retrievable from that run's own logs (pre-D-125 tail-window gap, D-124's own documented limitation); Section 11 already explains the FRAGMENTATION itself (a separate, upstream problem) as the likely dominant cause this run, independent of whatever Trigger B would have found within either fragment |
| D-126 (34172575066) | Formed correctly, 1 family, 2 members | `clip_51e6a8e265a375059ea9` (0.95) | Same clip (agreement) | Winner: 9 events/0.6s; alternative: 7 events/0.467s — **winner is WORSE** | `single_semantic_winner`, unbypassed (correctly — D-123's own scope excludes agreement cases) | 4 of 5 pimples checks pass; only `pimples_bad_monolith_absent` fails | **YES — this is the one clean, fully-evidenced, retrievable instance where Trigger B's own condition (agreement + real CASE B asymmetry favoring the non-chosen meaning-sufficient alternative) is directly and completely observable.** This is the run Trigger B is derived FROM, not designed around after the fact for pimples specifically — the condition itself (Section 4, Class B) makes no reference to pimples, sonography, or any Video00-specific content. |

**Only D-126 permits a full, evidence-complete answer** to "would Trigger B
have applied here" — every earlier run is missing some combination of the
structured fields Trigger B's own condition requires (a direct consequence
of D-122/D-123/D-125 not existing yet at the time those runs executed).
This is itself informative: **a general trigger can only be evaluated
retroactively as far back as the diagnostics that back it existed** — no
attempt is made here to "prove" Trigger B on D-113/D-116/D-118/D-124 by
guessing at missing fields.

---

## 14. Run-to-run variance — role

**Formalized distinction, per this task's own framing:**

- **(A) Production-time trigger: NO.** A single production run has no
  access to "how this family resolved in other runs" — each RAW is one
  independent invocation of the pipeline against one video; there is no
  cross-run memory available to `pipeline.py` at decision time, and
  building one (a persistent per-family-shape outcome ledger consulted
  live) would be new, unauthorized infrastructure, not a bounded fallback
  trigger.
- **(B) Eval/calibration evidence only: YES.** The cross-run pimples table
  above (Section 13) is exactly this use: it reveals that this family
  SHAPE has been unstable across six real runs (never-formed, formed-and-
  bypassed-pre-D-123, formed-and-fragmented, formed-cleanly-but-agreement-
  hid-a-real-conflict) even though the underlying source video never
  changes — that instability is real signal about WHERE the pipeline is
  weak, usable when designing the offline eval suite (Section 20) and when
  deciding which trigger classes deserve Phase 1 attention (Section 4),
  but it must never be read back into a live decision for a NEW video's
  first-ever run, which has no such history to consult.

---

## 15. Confidence / abstention policy

`UNCERTAIN` must remain a valid, expected, non-degraded arbiter outcome —
never papered over with a manufactured pick. Per D-111's own LOW-confidence
tier ("abstain / preserve safely / review") and the codebase's own
WHEN-UNCERTAIN-KEEP doctrine (CLAUDE.md Editorial rules), the correct
conservative behavior on `UNCERTAIN` (or any failure mode, Section 18) is:
**fall through to whatever the structured system already decided before
the arbiter was consulted** (the pre-fallback `selected_clip_id`/`final_
reason` this document's own input contract, Section 6, already carries)
— never a coin flip, never a default to "first finalist," never a silent
composite. This is the SAME fail-open contract `semantic_idea_equivalence.
py`'s own `safe_check_idea_equivalence` already implements for a different
arbiter (Section 17) — not a new pattern, a reused one.

---

## 16. Cost / latency boundary (recommendation only — no numbers invented)

Design constraints, all structural (no dollar/second/count values
proposed):

- Invoked ONLY on families that survive to a genuine Class-B-shaped (or
  future-authorized) conflict AFTER the full existing ladder — D-126
  shows this is a small minority (at most 1-2 of 8 families per video on
  the one real run with complete evidence).
- Bounded to 2-3 finalists per invocation (never a whole-family N-way
  call, never whole-video).
- Bounded to each finalist's own already-short clip duration (seconds,
  matching D-122's own scoping — never minutes, never full source video).
- A per-video invocation ceiling is the right SHAPE of guard (analogous to
  `HybridGatePolicy.max_candidates_per_request`/`SemanticEquivalenceGate
  Policy.max_pairs_per_request`, Section 17's own precedent) but the
  actual number must be set from measured real-run trigger frequency
  (Section 13's table is the start of that measurement, not its
  conclusion) — **recommend measuring trigger-class-B frequency across a
  batch of eval-suite runs (Section 20) before fixing any per-video cap.**
- Caching/reuse: candidate pairs already evaluated for the SAME family
  shape within one run should never be re-sent (structural, not a number).
- **What must be measured before setting any concrete limit:** (1) actual
  Class-B trigger frequency per video across the eval suite, (2) actual
  frame-extraction + vision-provider latency/cost per finalist pair using
  the EXISTING `visual_openai.py` pipeline as a proxy, (3) whether a
  2-frame-batch-per-finalist-pair request size (mirroring `OpenAIVisual
  Provider.batch_size=6`'s own precedent, scaled down for a 2-3-way
  comparison) is sufficient or needs adjustment.

---

## 17. Provider abstraction recommendation

**Recommend a distinct, bounded interface — do NOT extend `Semantic
EquivalenceArbiter`.** Rationale: `semantic_idea_equivalence.py`'s
contract is explicitly TEXT-ONLY BY DESIGN (`IdeaEquivalencePair`
carries no clip_id/timestamp/video identity — its own docstring calls
this out as the safety property preventing it from becoming a
Video00-specific guard). A multimodal BestTake arbiter needs the OPPOSITE
shape: it must carry clip identity, source spans, and (per Section 7)
actual frame/media references — extending the equivalence contract would
either break its own designed-in safety property or bolt on an unrelated
payload shape to a module whose entire contract is "text pairs only."

**Recommended NEW interface, mirroring the SAME proven pattern (`Semantic
EquivalenceArbiter` / `hybrid_editorial.py`'s `EditorialJudge` / `Visual
Provider`) — request/response/gate-policy/safe-call, not a new pattern:**

```
MultimodalBestTakeArbiter(Protocol):
    def arbitrate(self, request: MultimodalArbiterRequest) -> MultimodalArbiterResult: ...

MultimodalArbiterResult:
    family_id: str
    outcome: str            # one of Section 8's bounded vocabulary
    confidence: float       # 0..1, validated range (mirrors D-058's own confidence validation)
    reason: str             # concise, general, never clip-specific hardcoding
    evidence_considered: dict   # which of the input contract's fields it actually used
    provider: str
    model: str
    requested: bool
    available: bool
```

A `MultimodalArbiterGatePolicy` (max finalists, max estimated tokens/
frames, matching `SemanticEquivalenceGatePolicy`'s existing shape) and a
`safe_arbitrate` fail-open wrapper (matching `safe_check_idea_equivalence`/
`safe_visual_analyze`'s existing pattern exactly) complete the contract —
**none of this is implemented; this is the shape a future authorized task
would build.**

---

## 18. Failure-mode policy

| Failure | Policy |
|---|---|
| Provider timeout | Fail open — treat as `available=False`, fall through to pre-fallback structured decision (Section 15) |
| Provider error | Same |
| Invalid response (schema mismatch, out-of-range confidence) | Reject via validation (mirrors `validate_idea_equivalence_result`'s own raise-then-catch pattern), fail open |
| Unsupported media (e.g. corrupt frame extraction) | Fail open — never retry with degraded input silently |
| Low confidence (arbiter itself uncertain) | `UNCERTAIN` outcome (Section 15) — explicit, not a failure, but treated identically to failure for winner-selection purposes: fall through |
| Candidate missing (a named finalist no longer resolvable) | Fail open — this indicates an upstream contract violation (Section 6's input contract should prevent it) worth logging as an ANOMALY, never silently substituted |
| Meaning-safety mismatch (arbiter's own reasoning appears to contradict meaning sufficiency/polarity) | Reject the arbiter output entirely, defer to structured system — the existing D-101/D-103/D-063 safety layers are NEVER overridable by fallback (Section 9) |
| Cost ceiling reached | Fail open for the remainder of that video's run — never partial-apply mid-video |

**Fallback failure always degrades to exactly what the structured engine
already decided before consulting it** — never a worse outcome than not
having a fallback at all.

---

## 19. Required observability (future — not implemented)

A future implementation must log, per invocation: `family_id`, `trigger_
class`, `trigger_evidence` (the specific fields that qualified it), 
`candidate_ids`, `meaning_sufficient_candidates`, `structured_winner`
(pre-fallback), `structured_confidence_or_conflict_state`, `fallback_
invoked` (yes/no), `provider`, `model`, `arbiter_output`, `arbiter_
confidence`, `final_owning_authority` (structured vs. fallback-informed),
`winner_changed` (bool, vs. pre-fallback), `latency_ms`/`cost_estimate`
when available, `abstention_reason` when `UNCERTAIN`. This mirrors D-122/
D-123's own observability precedent (`winner_path_before/after`, `bypass_
reason`) — a NEW fallback layer should carry the SAME "before/after +
reason" shape, extended with a D-125-style tail-safe compact summary from
day one rather than retrofitted later (learn from the D-119→D-125 lesson
directly).

---

## 20. Offline eval plan (required before any activation)

Before ANY production activation, an eval suite must include (per the
directive's own list, cross-referenced against what this repo already has
or would need):

- **Pimples** — the D-126 positive instance (Section 13); already has
  real, complete evidence.
- **Papillary equivalent-realization case** — already tracked in the
  Human Gold regression QA's own `papillary_symptom_realization_parity`
  check (currently non-gating); a natural existing negative-control
  candidate (D-126 showed NO conflict here, i.e. arbiter must not
  "fix" a case that isn't broken).
- **Stomach retry** — named in D-097.11's own "escalation A" (arbiter
  run-to-run inconsistency, three runs three verdicts) — a DIFFERENT kind
  of instability (upstream semantic-arbiter variance, not a BestTake
  conflict) worth including as a negative control that a BestTake
  fallback must NOT attempt to resolve (it is not this arbiter's problem).
- **Complementary-content family** — a case where two members carry
  genuinely distinct information (D-019's KEEP/DISCARD-only doctrine
  means this must resolve to picking the single best one, never
  `KEEP_BOTH_COMPLEMENTARY`, per Section 8).
- **Polarity/negation safety** — D-066's own adversarial suite already
  has fixtures; reuse rather than re-invent.
- **Legitimate clean retry** — any of D-126's 3 trivial-agreement
  negative controls (Section 4 Class A) — arbiter must never be invoked.
- **Semantic+DeliveryScorer agreement but bad performance** — D-126's own
  pimples case, the canonical Class B positive.
- **Semantic/DeliveryScorer disagreement** — D-126's own confirmed D-123
  bypass (`tg_ef754f8f610ab360df`) — arbiter must NOT be needed here
  either (D-123 already resolved it correctly) — a control proving the
  fallback doesn't get invoked when the cheaper gate already worked.
- **Ambiguous/tied performance** — D-126's own tied-evidence negative
  control shape (Section 4/D-123's own offline tests already cover this
  at the code level; needs a REAL-media instance for this eval suite).
- **Boundary-only EXIT defect** — D-116's own CASE A visual-edge-trim
  fixtures are the template; must prove fallback is never invoked for a
  pure ENTRY/EXIT case (Section 12).

**The arbiter must prove it improves the identified conflict cases (like
pimples) without changing outcome on any negative control** before Phase
3 activation (Section 21).

---

## 21. Implementation phasing — recommended and appropriate

The directive's staged plan is **appropriate and is the recommendation**:

- **Phase 1 — advisory shadow fallback.** Build the interface (Section
  17), wire ONLY Class B's trigger condition (Section 4/5), call the
  arbiter, log everything (Section 19), but NEVER let its output change
  `selected_clip_id`. Zero production-behavior risk.
- **Phase 2 — offline comparison.** Run Phase 1's shadow logging against
  the eval suite (Section 20) plus Cut.ai/Human Gold references; measure
  whether the arbiter's shadow verdict would have IMPROVED the pimples-
  shaped positive case and left every negative control unchanged.
- **Phase 3 — bounded authoritative activation on Class B ONLY**, and
  only after Phase 2 evidence supports it — per Section 9, wired as
  advisory evidence INTO the existing ladder, never a standalone
  terminal authority.
- **Phase 4 — expand triggers (A's unresolved-terminal-state escalation,
  G's contradiction escalation, or others) ONLY with equivalent evidence**,
  never speculatively.

**None of these phases are implemented, scheduled, or authorized by this
document.**

---

## 22. Minimum first implementation scope (for when authorized — not now)

If and when the Product Owner authorizes Phase 1: (1) the `Multimodal
BestTakeArbiter` protocol + gate policy + `safe_arbitrate` wrapper
(Section 17), as a NEW module, zero changes to existing files' behavior;
(2) Class B's trigger condition as a pure, read-only, diagnostics-only
function (mirroring `_case_b_fast_path_conflict`'s own additive precedent
exactly — never wired into `_semantic_best_take`'s actual `return`
statements in Phase 1); (3) the request-assembly function implementing
Section 6's input contract; (4) a shadow-logging diagnostics row
(mirroring D-122's `case_b_evidence` row shape) added ADDITIVELY to
`take_judge_groups`, never consulted by any decision; (5) targeted tests
proving zero behavior change, exactly like D-122's own 41-test precedent.
**Not scoped further here — this is a pointer for the NEXT authorized
task, not a commitment made by this one.**

---

## Risks / regressions if built carelessly

- Treating DeliveryScorer-and-CASE-B agreement as independent confirmation
  (Section 3) would systematically UNDER-trigger on real conflicts, since
  they are correlated, not independent.
- Adding `KEEP_BOTH_COMPLEMENTARY` to the output vocabulary would silently
  reopen the SWAP/multi-realization question D-019 explicitly closed.
- Letting the arbiter become a terminal authority (Section 9 option A)
  would require re-implementing every existing safety veto a second time
  or risk silently bypassing D-101/D-103/D-063's proven protections.
- Reading cross-run variance (Section 14) into a live per-video decision
  would be architecturally impossible without new persistent
  infrastructure this document does not authorize, and would blur the
  eval/production boundary D-091's own doctrine depends on.
- Sending full RAW media instead of the bounded per-finalist span
  (Section 6) would reintroduce the exact "whole-video reasoner" pattern
  D-111 explicitly forbids fallback from becoming.

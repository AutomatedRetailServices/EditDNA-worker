# CutSell Editorial Resolution & Human Escalation Contract

Canonical doctrine document. Documentation only -- no engine behavior is
implemented, changed, or implied by this file. It registers the *concepts*
future engine work must be evaluated against, so that resolver escalation
behavior (when CutSell decides for itself vs. when it would ever ask a human)
has one durable definition instead of being re-invented per directive.

This document sits alongside `docs/CUTSELL_COMMERCIAL_ENGINEERING_OPERATING_MODEL.md`
(D-062) in the read order (see `AGENTS.md` / `CLAUDE.md`). Where the operating
model defines *who* may implement/certify, this document defines *what* the
editorial resolver is doctrinally supposed to do when evidence conflicts, and
*when* (if ever) that becomes a human's decision instead of CutSell's.

## 1. Automatic Editor Doctrine

CutSell is the editorial decision-maker. The product's value is an automatic,
finished, correct edit -- not a set of options for a human to adjudicate.

Human choice is a last resort, not a convenience valve for resolver
uncertainty. A resolver that is unsure is not evidence that a human should
decide; it is evidence the resolver needs a better rule, more evidence, or a
documented "when uncertain, keep/prefer-complete" default. Every escalation
to a human must be justified by the *nature of the ambiguity itself*
(irreducible, evidence-symmetric, no dominance signal available -- see
Section 6) and never by resolver conservatism, missing test coverage, or
convenience.

This doctrine does not change Selection Freeze, KEEP/DISCARD semantics, or
any existing gate. It states the standard those gates are already implicitly
operating under, so future resolver work has an explicit doctrine to be
checked against instead of ad hoc judgment calls per directive.

## 2. Semantic Dominance Before Performance

When two realizations of the same idea are compared, **critical semantic
content coverage must be checked and settled before delivery/performance
quality is allowed to decide the outcome.**

Example shape (generic, not tied to any specific video or clip):

- Realization A contains every critical fact/claim that realization B
  contains, **plus** at least one additional critical fact/claim B does not
  contain.
- B has marginally better delivery (cleaner audio, better pacing, no
  stumble) but is missing that additional fact.

Under Semantic Dominance Before Performance, A wins. Delivery quality never
overrides a content-coverage gap on a critical fact. Delivery quality is only
a legitimate tie-breaker **after** critical-claim coverage is equal between
candidates (see `CONTEXTUAL_DELIVERY_FIT`, Section 5).

This is the general principle behind D-062.1's specific finding (Section 8):
the resolver fell into `REVIEW_REQUIRED` on conflicting semantic-winner
labels without first checking whether one candidate was a strict superset of
the other's critical content.

## 3. CRITICAL_COVERAGE_DOMINANCE

`CRITICAL_COVERAGE_DOMINANCE` is the conceptual check that operationalizes
Section 2: given two (or more) candidate realizations of the same idea,
determine whether one candidate's critical-claim set is a strict superset of
every other candidate's critical-claim set.

- If exactly one candidate strictly dominates (superset, with at least one
  additional critical claim, and no critical claim it lacks that another
  candidate has), that candidate wins outright. Delivery/performance
  evidence is not consulted for this decision -- it only remains relevant if
  no dominance relationship exists.
- If no candidate strictly dominates (each has at least one critical claim
  the other lacks -- a genuine trade-off, not a superset relationship), this
  check does not resolve the conflict, and later layers apply (see the
  Automatic Resolution Hierarchy, Section 4, and Contradictory Retries
  handling per D-020, which already governs the case where the differing
  claims actively contradict rather than merely differ in coverage).
- Dominance is about **critical** claims specifically (the same notion
  `claim_coverage`/`lost_critical_claims`, D-038, already scopes to) --
  incidental phrasing differences, filler, or non-critical color do not
  enter into the dominance comparison.

`CRITICAL_COVERAGE_DOMINANCE` is a concept to be checked, not a new module,
score, or hard-coded weight. How it is computed (which existing
claim-coverage machinery it reuses) is an implementation question for
whichever future directive authorizes the D-062.1 fix -- not decided here.

## 4. Automatic Resolution Hierarchy (conceptual precedence, not hard-coded weights)

The following is a **conceptual precedence order**, not a scoring formula,
not a set of numeric weights, and not an implementation spec. It exists so
that when a future resolver decision must choose which signal governs, there
is one documented order of appeal instead of a per-directive judgment call.
Higher layers cannot be silently overridden by lower layers -- a lower-layer
signal (e.g. delivery quality) must never be allowed to reverse a decision a
higher layer already settled (e.g. critical coverage dominance, or a
contradiction that must block Freeze under D-020).

1. Explicit contradiction detection (D-020) -- differing number/negation/
   causal-direction retries are never composited or silently resolved; they
   block Selection Freeze for human review regardless of any other signal.
2. Idea coverage completeness (D-021/AttemptReconstructor+IdeaClusterer) --
   an intended idea must not silently vanish from the winning edit.
3. Complete-delivery-dominates-incomplete/abandoned-retry (existing
   preserved invariant) -- an abandoned or incomplete retry of the same idea
   never outranks a complete delivery of that idea.
4. **CRITICAL_COVERAGE_DOMINANCE** (Section 3) -- strict superset of
   critical claims wins outright when such a relationship exists.
5. Semantic arbiter confirmation/paraphrase-equivalence evidence (D-061) --
   claim-equivalence and idea-equivalence confirmations resolve ambiguous
   coverage bands and paraphrase credit, once no higher layer already
   settled the outcome.
6. Human-performance-error awareness (existing AGENTS.md/CLAUDE.md rule) --
   a technically-complete transcript can still lose to a candidate without a
   human performance error, but only once layers 1-5 are already satisfied
   for both candidates.
7. `CONTEXTUAL_DELIVERY_FIT` (Section 5) -- delivery/performance quality,
   consulted only once no higher layer resolves the comparison.
8. Story/personality preservation (existing AGENTS.md invariant) --
   authentic humor, mannerisms, reactions are not stripped merely because a
   more sterile edit looks cleaner, when nothing above already decided.
9. Boundary-only physical timing (existing invariant) -- never repairs a
   semantic membership mistake; only applies after semantic membership is
   already settled by layers 1-8.
10. `WHEN UNCERTAIN, KEEP` (existing default) -- the fallback bias when
    layers 1-9 leave a genuine tie on inclusion.
11. `REVIEW_REQUIRED_SEMANTIC` escalation (Section 6) -- reached only when
    layers 1-10 leave irreducible, evidence-symmetric ambiguity with no
    dominance signal.
12. `AUTO_RESOLVED_LOW_MARGIN` classification (Section 6) -- a resolution
    that layers 1-10 did produce, but by a narrow margin, recorded as such
    for observability even though it is not escalated.
13. `HUMAN_CHOICE_ELIGIBLE` classification (Section 6) -- the rare case
    where layers 1-12 genuinely cannot produce or justify a single winner,
    and the ambiguity is not itself a contradiction requiring D-020's
    mandatory block.
14. Selection Freeze -- applies once the above layers have produced a
    settled semantic membership (or a deliberate block per layer 1 or 11).
15. `CONTEXTUAL_DELIVERY_FIT`-informed Boundary/render-time physical
    adjustments -- strictly physical, never semantic (see layer 9).
16. Renderer -- executes the frozen, resolved plan; performs no editorial
    judgment of its own.

This hierarchy documents precedence relationships between existing and
future concepts. It does not assign numeric weights, does not specify a
scoring function, and does not itself require or authorize any code change.

## 5. CONTEXTUAL_DELIVERY_FIT

`CONTEXTUAL_DELIVERY_FIT` is the conceptual name for delivery/performance
quality evaluation **once no higher-precedence semantic signal (contradiction,
coverage completeness, dominance, arbiter confirmation) has already decided
the outcome.** It is not a replacement for or reordering of those higher
layers -- it is what legitimately remains to compare when candidates are
semantically equivalent in critical-claim coverage.

This is consistent with the existing Best Take doctrine already in
`AGENTS.md` ("Best Take ranks valid alternatives using multimodal/contextual
evidence. It is not a deletion authority.") -- `CONTEXTUAL_DELIVERY_FIT` is
that same ranking concept, named explicitly so the hierarchy in Section 4 can
refer to it as a specific, bounded layer rather than an undifferentiated
"quality score."

## 6. Escalation classes

Three mutually exclusive classes describe how a resolved (or unresolved)
comparison should be recorded. These are **semantics for future diagnostics
fields**, not implemented states -- see Section 13 (Future Engine States).

### `REVIEW_REQUIRED_SEMANTIC`

Used only when the Automatic Resolution Hierarchy (Section 4) genuinely
cannot produce a winner **and** the ambiguity is not a D-020 contradiction
(which has its own mandatory block, independent of this class). This is the
class for irreducible, evidence-symmetric ambiguity: no dominance signal
(Section 3), no arbiter confirmation, no coverage-completeness distinction,
and no legitimate default from `WHEN UNCERTAIN, KEEP` because inclusion
itself is not the question -- *which* candidate is.

`REVIEW_REQUIRED_SEMANTIC` freezes are not automatically human-facing (see
Section 7) -- they are an internal diagnostic/blocking state. Whether a given
`REVIEW_REQUIRED_SEMANTIC` case should ever become `HUMAN_CHOICE_ELIGIBLE`
is a separate, later judgment (Section 6.3), not an automatic consequence.

### `AUTO_RESOLVED_LOW_MARGIN`

Used when the hierarchy *did* produce a single winner, but the margin
between candidates was narrow (e.g. a dominance relationship existed but was
based on a single low-salience claim, or an arbiter confirmation carried
borderline confidence). This is not an escalation -- Freeze proceeds
normally with the resolved winner -- but the narrow margin is recorded for
observability, QA sampling, and `HUMAN_DECISION_RATE`-adjacent product
metrics (Section 10). Recording low margin is how the product learns where
its hierarchy is thin without turning every close call into a blocker.

### `HUMAN_CHOICE_ELIGIBLE`

Reserved for the narrow subset of `REVIEW_REQUIRED_SEMANTIC` cases where,
after exhausting the Automatic Resolution Hierarchy, a genuine, symmetric,
non-contradictory choice remains that a human could meaningfully and quickly
adjudicate (e.g. two complete, non-contradictory, equally-covered
realizations that differ only in a subjective style/tone preference no
signal in the hierarchy can settle). This is the **only** class that may
ever surface as a user-facing task, and only under the Human Choice Contract
(Section 7) and Future UX contract (Section 8) constraints below.

`HUMAN_CHOICE_ELIGIBLE` must never be assigned merely because
`REVIEW_REQUIRED_SEMANTIC` was reached -- it requires an additional,
affirmative determination that no further automatic signal is coming and the
remaining choice is genuinely presentable to a non-technical user without
resolver internals leaking through (Section 8).

## 7. Human Choice Contract

Uncertainty alone never creates a user-facing task.

A resolver being unsure, a low-confidence arbiter call, a narrow margin
(`AUTO_RESOLVED_LOW_MARGIN`), or even a full `REVIEW_REQUIRED_SEMANTIC`
internal block does **not**, by itself, mean the user is shown anything.
Surfacing a choice to the user requires the affirmative `HUMAN_CHOICE_ELIGIBLE`
determination (Section 6.3) -- deliberately a high bar, so that resolver
immaturity or thin test coverage never becomes the user's problem to solve.
The default posture for any unresolved ambiguity remains internal (block
Freeze per D-020, or apply `WHEN UNCERTAIN, KEEP`), never a user prompt.

## 8. Future UX contract (contract only -- not implemented)

If and when `HUMAN_CHOICE_ELIGIBLE` cases are surfaced to users, the
presentation contract (to be implemented later, under its own directive) is:

- A compact A/B comparison of exactly the two (or few) remaining candidate
  realizations.
- The user can play each candidate and choose "A" or "B" (or equivalent).
- The presentation must **NOT** expose confidence scores, semantic ids,
  clip ids, resolver terminology (`REVIEW_REQUIRED_SEMANTIC`,
  `CRITICAL_COVERAGE_DOMINANCE`, arbiter names, etc.), or any other
  internal diagnostic language. The user sees two finished options and picks
  one, nothing more.
- This is analogous in *shape* to SWAP's alternate-take comparison UI
  concept, but is a **distinct** product layer: SWAP (D-019, out of scope
  for Clean Cut V1) is a user-driven alternate-take browsing/inventory
  feature; `HUMAN_CHOICE_ELIGIBLE` is a rare, resolver-driven escalation for
  a specific unresolved comparison the automatic editor could not settle
  itself. Do not conflate the two, and do not use this section as
  authorization to build, revisit, or reintroduce SWAP.

No UI is implemented by this document. This section is a forward-looking
contract only.

## 9. SWAP vs. HUMAN_CHOICE

To make the distinction in Section 8 unambiguous:

| | SWAP (D-019, out of scope) | HUMAN_CHOICE_ELIGIBLE (this doc) |
|---|---|---|
| Trigger | User-initiated browsing of alternate takes | Resolver-initiated, only after exhausting Section 4's hierarchy |
| Scope | Any take the user wants to inspect/swap in | Only the specific unresolved comparison the resolver could not settle |
| Frequency | Available broadly, whenever the user wants it | Rare, reserved for genuine irreducible ties |
| Membership model | Alternate-take inventory alongside the winning timeline | Still KEEP/DISCARD only -- no alternate-take inventory is created or preserved |
| Status | Explicitly out of scope for Clean Cut V1 until the user reintroduces it | A doctrinal contract only; not implemented |

Nothing in this document reintroduces SWAP or authorizes engineering time
against it, per D-019.

## 10. HUMAN_DECISION_RATE (product metric, no hard threshold)

`HUMAN_DECISION_RATE` is defined as: the proportion of processed
videos/ideas for which a `HUMAN_CHOICE_ELIGIBLE` task was surfaced to the
user, out of all processed videos/ideas.

This is a product health metric, not a gate: a rising `HUMAN_DECISION_RATE`
signals the Automatic Resolution Hierarchy is failing to resolve cases it
should be resolving automatically (per the Automatic Editor Doctrine,
Section 1, human choice is a last resort) and should prompt investigation
into which hierarchy layer is under-specified. No hard numeric threshold is
set by this document -- that is a future product decision, to be made once
real `HUMAN_CHOICE_ELIGIBLE` volume exists to calibrate against.

## 11. QA_ENGINE contract update

`RUN QA_ENGINE` (per D-062's operating model) must, from this checkpoint
forward, additionally challenge every `HUMAN_CHOICE_ELIGIBLE` result against
a checklist before accepting it as correctly classified:

1. Was `CRITICAL_COVERAGE_DOMINANCE` (Section 3) actually checked and ruled
   out (no strict superset relationship) before escalating?
2. Was D-020 contradiction detection checked and ruled out (the ambiguity is
   a genuine coverage/style tie, not a number/negation/causal-direction
   contradiction, which must block rather than escalate to human choice)?
3. Was semantic-arbiter/paraphrase-equivalence evidence (D-061) already
   consulted and found insufficient to resolve the tie?
4. Does the remaining difference between candidates actually meet the
   Future UX bar (Section 8) -- presentable as a compact A/B choice without
   leaking resolver internals?
5. Is this truly `HUMAN_CHOICE_ELIGIBLE`, or does it better fit
   `AUTO_RESOLVED_LOW_MARGIN` (a winner does exist, just by a narrow margin)
   or `REVIEW_REQUIRED_SEMANTIC` (should stay an internal block, not reach
   the user)?

A `HUMAN_CHOICE_ELIGIBLE` result that fails any checklist item is
mis-classified and must be corrected (to `AUTO_RESOLVED_LOW_MARGIN`,
`REVIEW_REQUIRED_SEMANTIC`, or a D-020 contradiction block) before it is
accepted as a QA_ENGINE pass. This checklist is a contract for whichever
future directive implements these classifications -- it does not itself run
against any code today, since no such classification exists yet (Section
13).

## 12. Relationship to existing CleanCutBench / editorial test suites

This document does not add new test requirements to CleanCutBench today. It
establishes the doctrine that a *future* implementation of
`CRITICAL_COVERAGE_DOMINANCE` and the escalation classes must be validated
against, so that when that implementation directive is authorized, its test
matrix and QA_ENGINE review have an existing contract to be checked against
rather than being invented from scratch under time pressure.

## 13. Future Engine States (semantics only -- not implemented)

The following are **documented semantics for future diagnostics/state
values**, not implemented enum members, not new code paths, and not
authorization to implement them:

- `AUTO_RESOLVED` -- the hierarchy (Section 4) produced a single winner with
  a comfortable margin; no observability flag needed beyond normal
  diagnostics.
- `AUTO_RESOLVED_LOW_MARGIN` -- per Section 6.2.
- `REVIEW_REQUIRED_SEMANTIC` -- per Section 6.1.
- `HUMAN_CHOICE_ELIGIBLE` -- per Section 6.3.

No code today emits any of these four values. Their introduction, together
with the `CRITICAL_COVERAGE_DOMINANCE` check itself, is future engineering
work requiring its own directive, its own targeted tests, and its own
`RUN QA_ENGINE` pass (per Section 11's checklist) before any paid canary --
consistent with D-062's operating model and this session's established
gating pattern.

## 14. D-062.1 as the canonical example of NOT-human-choice-eligible

D-062.1 (see `docs/CUTSELL_DECISIONS.md`) traced a real Freeze blocker
(`tg_c7c1ae9f22e6c10986`, two clips where one strictly dominated the other's
critical content plus carried the shared reflective statement, while the
other had marginally better delivery). The resolver reached `REVIEW_REQUIRED`
on conflicting semantic-winner labels without first checking
critical-claim-coverage dominance.

Under this contract, that case is **not** `HUMAN_CHOICE_ELIGIBLE`. It is a
resolver defect: `CRITICAL_COVERAGE_DOMINANCE` (Section 3) should have been
checked before the conflicting-semantic-winner-labels branch was allowed to
fire, and had it been checked, it would have produced a single dominant
winner automatically (Section 2, Semantic Dominance Before Performance) --
no ambiguity, no escalation, no human task of any kind. This is a strict,
mechanical failure to apply layer 4 of the hierarchy (Section 4) before
falling through to a lower-confidence branch, not a genuine tie between two
otherwise-equivalent candidates.

D-062.1 is registered here purely as the **generic pattern** this contract
exists to prevent (a coverage-dominance case being escalated instead of
auto-resolved). Do not encode the Video00 text, clip ids, or idea id from
D-062.1 into production logic -- the example exists only to document the
general contract; the actual fix (implementing the
`CRITICAL_COVERAGE_DOMINANCE` check itself) is deferred to its own future
directive, per Section 13.

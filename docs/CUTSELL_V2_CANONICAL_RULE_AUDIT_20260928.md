# CutSell V2 — audit of inherited rules

Date: 2026-09-28  
Scope: Editorial Engine V2 on Video00 and Yaskira 01–10.  The canonical
project is evidence, not automatic authority for the new engine.

## Decision standard

A rule is retained only when it prevents an observed failure without causing
a larger Human Gold error.  A rule is reworked when its intent is valid but
its current proxy or layer is wrong.  A rule is retired from V2 when it acts as
editorial authority without evidence of net value.  Human Gold and rendered
Watch + Listen remain QA-only oracles; they never enter production prompts or
runtime decisions.

## Retain

| Rule/protection | Evidence and V2 disposition |
|---|---|
| Deterministic source identity, transcript/word provenance and no fabricated speech | Required to reproduce decisions and prevents cross-take sentence invention. Retain. |
| Numbers, negations, names, causal claims and qualifications require explicit preservation checks | A misplaced/lost “no” can invert meaning; this is content integrity, not stylistic conservatism. Retain, but it cannot automatically preserve an entire inferior retry. |
| Selection Freeze and semantic hash | Cycle 2 of Video08 violated the frozen semantic-token contract after Selection; the invariant correctly refused an unsafe render. The hash evidence alone does not prove whether membership, word coverage or order changed. Retain the guard and reproduce the exact mutation before changing the responsible operation. |
| Provider identity, budget, retry and failure observability | Needed for reproducibility, cost control and honest failure states. Retain. |
| Authentication, tenant isolation, secret handling, private media and destructive/deployment approvals | Commercial security boundaries independent of editorial strategy. Retain until a dedicated security review proves a replacement; never remove to improve an edit. |
| Render/decode/codec and delivery QC | Necessary but not evidence of editorial quality. Retain as technical gates. |
| Actual rendered MP4 Watch + Listen acceptance | Prior CI success did not predict Video00 editorial success. Retain as the proof gate after offline tests. |

## Rework

| Inherited rule | Observed defect | V2 replacement |
|---|---|---|
| `WHEN UNCERTAIN, KEEP` | Useful against silent loss, but as a universal final authority it over-preserved Video08's earlier patchwork after the complete later take was recovered. | Apply only to unresolved material facts or genuinely independent content. Uncertainty about which equivalent retry wins must trigger whole-attempt comparison, not keep every retry. |
| Unique-information safeguard based on lexical tokens | Styling/examples and ordinary rewording looked “unique”; Video08 kept redundant fragments. | Judge material semantic/visual contribution relative to the winning complete attempt. Token novelty is diagnostic evidence only. |
| Retry grouping by one `take_group_id` | The prompt saw fragments individually; additionally the legacy model aliases take group, retry family and semantic idea. | Keep source-scoped take grouping as provisional delivery evidence. Build explicit interval/gap summaries. Infer delivery attempt, retry family and semantic idea as separate concepts. |
| Best Take on local families | Can choose a locally clean fragment while missing a later take covering the union of several fragments. | Compare complete reconstructed attempts and the union of patchwork fragments across the whole source before candidate-level actions. |
| Complete-delivery dominance | Correct intent, but “complete” was inferred at fragment level and could suppress complementary composites. | A complete attempt competes only with alternatives for the same intended delivery. Complementary pieces remain valid when their union adds material coverage or necessary continuation. |
| Critical-coverage dominance | It can overvalue any extracted claim and choose a worse monolith; earlier evidence showed the resolver maximizing coverage mechanically. | Limit dominance to verified material claims within genuine competing attempts; require contradiction and completion checks first. Incidental wording is not critical coverage. |
| Post-Freeze boundary cleanup | Cycle 2 produced different frozen/final semantic hashes; available evidence does not yet isolate the exact mutator. | Instrument and reproduce the transition. Any operation that changes words, membership or spoken order belongs before Freeze. Post-Freeze Boundary may alter only physical edges while preserving the frozen spoken stream. |
| Human-performance and visual evidence | Correct principle but broad AV regions can miss short resets/suffixes. | Use exact candidate/take-aligned Watch + Listen evidence; absence of a detected flaw is not proof of a clean take. |

## Approved retirement target for V2 authority

The audit concludes that the following must be removed as decision authorities.
Some still execute in the current runtime, including lexical unique-token
protections; therefore this is the migration target, not a claim that retirement
is already complete.  Until each call path is changed and regression-tested,
it remains an open implementation item and must be reported as such:

- treating `take_group_id`, `retry_family_id` and `semantic_idea_id` as aliases;
- token-count or vocabulary novelty thresholds as proof that a retry contains
  unique audience value;
- legacy current bucket, local family or Hybrid vote as privileged truth;
- fixed funnel order or target-duration pressure as a reason to delete valid
  speech;
- “longer is better” or “shorter is cleaner” as standalone take selection;
- preserving all alternatives merely because the system is uncertain;
- technical CI/workflow success as a claim of editorial improvement.

## Video08 root cause and current change

Video08's Human Gold is the continuous `2:00–2:27` delivery.  The baseline
selected earlier fragments and missed it.  Cycle 1 recovered the late take but
kept the fragment patchwork because candidate-level novelty protection treated
rewording as unique.  Cycle 2 still fragmented the late attempt; a later stage
then violated the frozen semantic-token contract, and Freeze correctly blocked
rendering.  The recorded hash mismatch does not by itself identify whether the
change was membership, word coverage or order, so exact causation remains an
instrumented reproduction task.

V2 now receives an explicit `take_groups` view containing source identity,
original candidate indices, every interval, measured gaps, bucket state and
text excerpts.  This view is explicitly evidence-only: it cannot merge
candidates, change boundaries, prove continuity, or substitute for retry/idea
inference.  Missing IDs remain singletons and equal IDs from different sources
never merge.

## Exit criteria

This change is accepted only after:

1. positive and adversarial tests pass, including missing IDs, same ID across
   sources, long gaps, split phrases and material unique fragments;
2. independent QA finds no new deletion authority or one-video hardcode;
3. Video08 is rendered and compared with Human Gold by Watch + Listen;
4. regressions on 05, 06 and 09 show the fix did not collapse legitimate
   composites or unique content;
5. results are recorded whether they pass or fail.

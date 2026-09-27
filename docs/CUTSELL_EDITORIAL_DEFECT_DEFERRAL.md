# Editorial defect deferral and bounded calibration
Decision date: 2026-09-27, 17:07 America/New_York.
Owner direction: accept minor defects provisionally, record them, and revisit with more examples instead of spending indefinitely on one video.

## Active policy
- Current Yaskira09 correction loop is stopped: zero additional paid runs for this isolated case under this checkpoint.
- Operational limit for subsequent calibration: at most two correction-and-evaluation cycles per defect class in a batch. Each cycle must test a specific hypothesis. Repeated execution without a change or new evidence is not a correction cycle.
- After that limit, retain the best observed output and record the defect. Do not mark a known failure fixed or reset the limit merely by changing version numbers.
- Reopen when independent examples clarify the same failure, or a relevant engine change justifies a bounded regression evaluation.
- More videos provide evidence; the current engine does not automatically learn from executions. General changes must be evaluated against both original cases and independent cases.
- Minor residual debris or redundant CTAs may be provisionally tolerated. Missing substantive demonstrations, changed meaning, and lost unique information remain substantive failures. No commercial release approval is implied.
- Runtime automatic retry/budget enforcement is NOT implemented by this documentation decision.

## Known cases
| Case | Evidence | Disposition |
| --- | --- | --- |
| Yaskira09 opening | V17 run 36348620732 retained incomplete ending “y aparte tus músculos tus mus” | Minor defect provisionally accepted and deferred |
| Yaskira09 closing | Same run retained recording comment “ya se acabó ese” | Minor defect provisionally accepted and deferred |
| Yaskira09 demonstration | Same run omitted the demonstration around source 96–106 seconds | Substantive pending defect; NOT covered by minor-defect acceptance |
| video00 repeated CTA | Owner reports two CTAs instead of the intended one on 2026-09-27; exact artifact/timecodes not reverified this turn | Record as owner-reported minor defect and defer. Earlier claims of a single final CTA are insufficient to close this report. Preserve distinct useful CTA information; no universal “delete any second CTA” rule |

## Engineering checkpoint
Remote implementation: cd4094d652c038322d2028ff4160026459a3660e (V17).
V17 workflow completed, but editorial failures above remain.
Local V18 candidate requires explicit suffix assessment and changes continuous-demonstration preservation. Relevant regression: 475 passed. Independent QA found no blocking defect, with the limitation that AV and selection confidence scores are not calibrated against each other.
V18 is not published or live-qualified at this decision. Do not label its test results as video quality acceptance.
This documentation-only checkpoint does not change workflow triggers or launch another paid qualification.

## Documentation exception
CUTSELL_DECISIONS.md has pre-existing invalid UTF-8; preserve it unchanged. This companion decision is the durable record for this cycle.

# V3 Door 0 — existing evidence inventory

**Date:** 2026-09-29. **Purpose:** read-only inventory for `CUTSELL_EDITORIAL_ENGINE_V3_EXECUTION_CANON.md`. No RAW was reprocessed and no provider call or paid run was started for this inventory. Saved artifacts were extracted/inspected only. Gold comes from `docs/CUTSELL_YASKIRA_01_10_HUMAN_GOLD.md`; none of it was added to production inputs.

## Findings

The ten-video Gold and a historical offline comparison already exist. That 2026-09-27 comparison is a **historical baseline**, not a score for the current V2 revision. Latest saved evidence is uneven: there is recent evidence for 06, 08, 09 and 10, while 01–05 and 07 do not have a complete current-revision packet in the workspace. Therefore Door 0 is **IN_PROGRESS**, not PASS.

| Source | Latest usable saved evidence | Gold result | What the evidence establishes | Still unknown / next artifact needed |
| --- | --- | --- | --- | --- |
| 01 | Historical baseline in `CUTSELL_YASKIRA_BASELINE_VS_GOLD_20260927.md` | 89.5% recall; 12.06 s KEEP lost; 1.00 s DELETE retained | Engine did not preserve all approved source time. | Current-revision candidate/word evidence and decision report to locate whether loss starts before or at selection. |
| 02 | Same historical baseline | 95.0% recall; 4.80 s KEEP lost; 2.33 s DELETE retained | Both false deletion and extra retention occur. | Current-revision trace, then per-error layer attribution. |
| 03 | Same historical baseline | 91.2% recall; 1.72 s of an all-KEEP source lost; 0 s extra | A technically clean output can still remove approved content. | Current-revision trace and exact source intervals. |
| 04 | Same historical baseline | 87.7% recall; 4.66 s KEEP lost; 0.29 s extra | Material approved content is missing. | Current-revision trace. |
| 05 | Historical baseline; approved reference also in `evidence/YASKIRA_05_HUMAN_EDITORIAL_REFERENCE_20260925.md` | 94.9% recall; 1.97 s KEEP lost; 0.92 s extra | Baseline is not the approved later result. | Current-revision run manifest and full decision evidence. |
| 06 | Run `36534356753`, artifact `11018520264` | 22/22 s Gold retained; 0.34 s extra; QC PASS | One strong saved result is a useful regression control. | Stability across revisions is not established; do not call this source permanently solved. |
| 07 | Gold doc plus partial known RAW-time events; historical baseline only | Full interval map unavailable; no valid full-source score | There is known partial editorial evidence, not a complete aligned Gold timeline. | Mechanically align the authoritative `Video00_Human_Gold.mp4` to `Yaskira/07.mp4`; until then mark EVIDENCE_INCOMPLETE. |
| 08 | Run `36546701227`, artifact `11023061700`, revision `37cb7b8d`; prompt SHA `ca3c16aa…` | 25.199/27 s retained; 0 s outside Gold; QC PASS | Final delivery was selected and earlier attempts were absent from the selected intervals in this run. | 1.801 s Gold not intersected: 0.39 s leading margin, 1.321 s internal gap, 0.09 s trailing margin. The internal gap is measured silence in prior source analysis; inspect the relevant saved boundary evidence before calling it an editorial loss. One run does not establish repeatability. |
| 09 | Latest available 09 run `36538958510`, artifact `11019339666`, revision `ae2a4366` | 65.146/73 s retained; 7.854 s KEEP lost; 11.252 s DELETE retained; QC PASS | Selection remains materially wrong. The optional competition review failed because its explanation exceeded parser length; the saved selection used the first-pass comparisons. | Need the full interval/decision trace on one frozen evidence packet to split attempt grouping, semantic comparison and silent visual action. Do not attribute all 19–28 or 97–120 failures to one layer without that trace. |
| 10 | Run `36546701251`, artifact `11023682900`; compact extraction `36553561580` / artifact `11024579682`; revision `37cb7b8d` | 26.8/36 s retained; 9.2 s KEEP lost; 18.6 s DELETE retained; QC PASS | ASR includes the full approved phrase in candidate `8.95–24.77`. The selector labels that candidate `retry_alternate/redundant_retry` and swaps it out. It selects `33.98–41.16` (outside Gold) and `46.75–64.65` (first 8.25 s outside the 55–64 Gold block). The error reaches semantic selection despite the phrase being present in the input. | AV for this run was broad `audience 8–42 s` without a focused delivery judgment. Need determine how attempt comparison uses that evidence and why the candidate’s unique clean subspan cannot survive the selected whole-take decision. |

Gold interval scoring uses the owner’s rounded source-time labels. It is useful for diagnosis but cannot by itself judge whether a short interior gap is meaningful, whether a visual action is part of the story, or whether two recordings sound like separate attempts.

## Cross-video diagnosis supported by these artifacts

1. **A universal ASR failure is not supported.** In Video10 the approved speech is present in the saved ASR candidate, yet the candidate is demoted as a redundant retry. For that measured miss, transcription absence is not the cause.
2. **The selector can commit a semantic error and still pass technical QC.** This is explicit in Video09 and Video10. Render/QC status cannot certify editorial correctness.
3. **Attempt grouping and selection are not robust enough.** Video10’s selected whole-take alternative removes approved content along with failed starts; Video09’s comparison review is invalidated; Video08’s latest saved selection keeps only the final take but has boundary/gap omissions.
4. **Not all ten can yet be diagnosed at the same evidence depth.** 01–05 are historical scores without current traces here. 07 lacks the full RAW-aligned Gold. Any stronger claim about their present failure layer would be speculation.

## Door 0 exit checklist

- [x] Canonical owner Gold found for 01–10.
- [x] Historical 01–10 interval baseline found and explicitly marked historical.
- [x] Recent, measured saved reports inspected for 06, 08, 09 and 10.
- [x] Video10 failure located at semantic selection, with the approved phrase present in ASR input.
- [ ] Current-revision evidence packet for 01–05.
- [ ] Complete RAW-time alignment of Video07 Gold.
- [ ] One normalized, replayable manifest schema containing media hash, provider/model/config versions, full ASR word timeline, AV evidence, attempt candidates, selector input/output, plan, selected intervals and rendered mapping for each of 01–10.

**Gate status: IN_PROGRESS.** The existing evidence is sufficient to reject the claim that all failures are simply missing transcription or bad FFmpeg boundaries. It is not sufficient to close per-stage attribution for all ten. The next implementation decision should use the saved Video10 selector failure and Video09 review failure as concrete counterexamples, while Door 0 fills only the evidence gaps listed above. No new paid RAW run is required to write this inventory.

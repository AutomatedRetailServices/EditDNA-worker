# V3 Door 0 — existing evidence inventory

**Date:** 2026-09-29. **Purpose:** read-only inventory for `CUTSELL_EDITORIAL_ENGINE_V3_EXECUTION_CANON.md`. No RAW was reprocessed and no provider call or paid run was started for this inventory. Existing GitHub Actions artifacts were extracted and inspected only. Gold comes from `docs/CUTSELL_YASKIRA_01_10_HUMAN_GOLD.md`; none of it was added to production inputs.

## Evidence packets

- Historical V2 multi-video diagnostics: runs `36502632708` (02–06, 09) and `36502756185` (01, 05, 07, 10), extracted into Actions artifact `11026136307` (`v2-door0-saved-decision-evidence`). The runs are historical V2 results, not current V3 qualification. Source SHA was absent in this packet.
- Later saved diagnostics: 06 run `36534356753`; 08 run `36546701227`; 09 run `36538958510`; 10 run `36546701251`, compacted from existing artifact by run `36553561580`. These help explain variation but do not replace a normalized, same-revision ten-video replay manifest.
- The 2026-09-27 offline comparison remains a historical baseline only.

## Per-video evidence

| Video | Saved V2 cross-video evidence vs owner Gold | Directly observed failure evidence | Attribution status |
| --- | --- | --- | --- |
| 01 | 107.198/114.561 s Gold retained; 7.363 s KEEP lost; 1.000 s outside Gold retained. | Attempt reconstruction formed a 0–29.18 s unit from five member fragments. The old selector discarded 105.33–114.30 s as `redundant_retry`; the later 2026-09-27 baseline is not this run. | Selection/attempt-unit concern is visible; exact Gold-loss-to-decision mapping needs replay and full boundary comparison. |
| 02 | 90.380/96.262 s retained; 5.882 s lost; 2.503 s outside Gold retained. | Optional competion review completed in `missing_competitions` mode with zero comparisons. | Selection result is imperfect; packet does not establish which stage caused each interval error. |
| 03 | 17.756/19.623 s Gold retained; 1.867 s lost; no extra retention. All three reconstructed units were selected. | No alternative-take competition was eligible. | Boundary/selection mapping not isolated; all-KEEP Gold does not mean any removed source span is editorially invalid. |
| 04 | 34.160/38.000 s retained; 3.840 s lost; 0.440 s outside Gold retained. | Three complete units selected; 0.46 s `failed_delivery` discarded at 26.493–26.953 s. | Small failed segment is directly attributed; remaining Gold loss not isolated. |
| 05 | 37.029/39.000 s retained; 1.971 s lost; 0.540 s outside Gold retained. | 5.17–13.41 s is labeled `swap/redundant_retry`; 26.88–48.48 s also `swap/redundant_retry`; AV evidence includes `mixed` and `recording_only` over several regions. Gold begins at 98 s, so earlier labels alone do not prove a Gold miss. | Selector/attempt classification is active; need inspect exact selected replacement and word boundaries against the approved 8;–137 s reference. |
| 06 | Historical run: 21.904/22 s retained; 0.096 s lost; 26.630 s outside Gold retained. Later saved run: 22/22 s retained, 0.34 s extra, render QC PASS. | In historical run, the 17.85–25.19 s unit is discarded as `redundant_retry`; later run has a better Gold overlap. | Measured revision/run variance. One good later result does not establish stability; investigate extra retention. |
| 07 | No valid full-source Gold score. | Run failed before selection: 72,502 input tokens vs 64,000 max, 37 candidates. | Confirmed selector input-budget failure; obtain full RAW-time alignment of the authoritative Human Gold before scoring. |
| 08 | Latest saved run: 25.199/27 s retained; no out-of-Gold retention; render QC PASS. | Selected intervals contain only the final delivery; 1.801 s Gold not intersected (0.39 s lead, 1.321 s internal gap, 0.09 s tail). Prior audio analysis describes the internal gap as measured silence. | Final take selection succeeded on this run; boundary treatment/repeatability remains open. No further Video08 run is implied. |
| 09 | Latest saved run: 65.146/73 s retained; 7.854 s Gold lost; 11.252 s outside Gold retained; QC PASS. | The historical run failed with `overlapping or empty competition`. Latest saved optional competition review failed because the explanation exceeded parser length; first-pass selection still yielded a render. | Confirmed malformed competition validation and review robustness issues on separate runs; current miss location needs frozen full-trace attribution. |
| 10 | Historical packet: 25.380/36 s retained; 10.620 s lost; 13.413 s outside Gold retained. Later focused run: 26.8/36 retained; 9.2 s lost; 18.6 s outside Gold retained; QC PASS. | Approved speech exists in ASR within 8.95–24.77 s, but selector marked it `discard/redundant_retry`; selected alternatives include 33.98–41.16 s. Attempt grouping joins three fragments into 8.95–24.77 s. | Confirmed semantic selection error after transcription on this span. It does not prove a single cause for every 10 error. |

All overlap values are calculated against the owner’s rounded source-time Gold labels. Gold interval overlap is diagnostic, not a complete human-quality judgment.

## Cross-video findings supported by saved artifacts

1. **The failures are not all in one layer.** Video07 fails before editorial selection because its input exceeds the token budget. Video09 also has invalid competition data/review handling. Videos05 and 10 show semantic labels (`redundant_retry`) affecting take decisions. Video08’s latest miss includes a measured silence gap and margins.
2. **Attempt reconstruction can make a mixed editorial unit.** In the historical traces, Video01 combines five fragments into 0–29.18 s, Video10 combines three into 8.95–24.77 s, and Video05 combines multiple fragments into other long units. Selection then assigns one action/relation to the reconstructed unit. This is evidence that the current representation can hide useful subspans; it is not yet a measured root cause for every video.
3. **ASR absence is not the cause of Video10’s 8.95–24.77 s approved phrase miss.** The phrase is in the saved ASR text, but the decision is to discard the candidate as a redundant retry.
4. **Technical render QC does not certify editorial correctness.** Video09 and Video10 have QC PASS alongside measurable Gold misses and extra retention.
5. **Saved cross-video figures are not apples-to-apples proof of a current revision.** The multi-video packet has no source SHA, and later runs use differing commits/configurations. Preserve the figures as historical diagnostics; do not call them a stable baseline.

## Door 0 exit checklist

- [x] Canonical owner Gold found for 01–10.
- [x] Historical interval baseline found and labeled historical.
- [x] Saved cross-video V2 decisions and Gold overlap extracted for 01, 02, 03, 04, 05, 06, and 10.
- [x] Failure records inspected for 07 and 09.
- [x] Later saved evidence inspected for 06, 08, 09, and 10.
- [ ] Current-revision evidence packet with source hashes and matching configs for 01–10.
- [ ] Complete RAW-time alignment of Video07 Gold.
- [ ] One normalized, replayable manifest schema with media hash, provider/model/config versions, full ASR word timeline, AV evidence, attempt candidates, selector input/output, plan, selected intervals, and render mapping for each source.

**Gate status: IN_PROGRESS.** The diagnostic inventory now covers saved evidence across all ten source labels, but the packets are historical and not revision-normalized. Do not start paid RAW reruns to fill these gaps. The next V3 step is to define the normalized evidence manifest and replay the stored artifacts offline; Video07’s Gold alignment remains a separate prerequisite for a valid score.

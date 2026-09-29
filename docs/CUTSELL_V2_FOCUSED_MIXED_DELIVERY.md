# V2 calibration: focused mixed-delivery evidence

## Failure under investigation

The V33b Video09 replay lost a valid delivery after the whole-source audiovisual
pass labeled 17.5–26.5 s `mixed`. The candidate at 19.218–28.14 s was then
classified `failed_delivery`, and its unique spoken claim was removed. A
previously completed 11.5-second focused observation of source 17.5–29 s
confirmed a clear camera-facing delivery from local 1.1–6.8 s (source
18.6–24.3 s). This confirms that broad region labels can cover a clean delivery
and a nearby reset in the same candidate.

## General correction

When V2 is enabled, the whole-source audiovisual provider can nominate up to
two short follow-up observations per source for confident `mixed` or
`uncertain` regions. Each call uses the existing token preflight and dollar
ledger. The focused result is advisory and cannot select footage by itself.

For a candidate independently labeled `failed_delivery`, the reasoner can
preserve a focused, high-confidence audience delivery only when at least three
aligned words and three previously uncovered content tokens fall inside that
observed span. It trims the candidate to those original ASR word timestamps,
excluding unobserved tail content. No benchmark timestamps or phrases are
embedded in production logic. If aligned words or focused evidence are absent,
the existing decision remains in force.

## Verification

- Targeted offline regression: 110 tests passed across V2 reasoner, AV provider,
  source selection, runtime, and focused continuation suites.
- Full repository collection was attempted but is blocked in this environment
  by missing optional dependencies (`fastapi`, `modal`, `botocore`, `redis`) and
  a collection-time error in `tests/test_semantic_stitch.py`.
- Video08 run `36502227811` completed with HTTP 200, no block reason, 12
  candidates, Selection applied, and render QC PASS. The selected timeline was
  120.13–141.388, 142.709–145.1, and 145.1–147.05 s. It retained 25.549/27 s
  of Gold, retained 0.05 s outside Gold, and omitted 1.451 s inside Gold
  (0.13 s at the opening and a 1.321 s gap). Visual preview inspection found a
  coherent try-on sequence; the earlier `PROHIBITED_CONTENT` block did not
  recur, so its cause remains unconfirmed rather than fixed. This result is an
  improvement in completion, not an exact Gold match or quality certification.

This is a calibration change, not a release-quality certification. Gold remains
an evaluation artifact and is not passed to production selection.

## Same-revision qualification: runs 36502632708 and 36502756185

Both workflows completed. The first produced four renders and failed on 09;
the second produced three renders and failed on 07. Source-time comparisons
below use the owner-declared rounded Gold and are not a judgment of the audio
or the finished story.

| Video | Gold lost (s) | Outside Gold retained (s) | Result |
| --- | ---: | ---: | --- |
| 01 | 4.104 | 1.000 | Render QC PASS |
| 02 | 1.629 | 3.000 | Render QC PASS |
| 03 | 0.003 | 0 | Render QC PASS |
| 04 | 3.840 | 0.440 | Render QC PASS |
| 05 | 1.971 | 0.540 | Render QC PASS |
| 06 | 0 | 26.740 | Render QC PASS; editorial failure |
| 07 | unknown | unknown | Selection preflight 72,502 tokens exceeded 64,000 ceiling |
| 08 | 1.451 | 0.050 | Separate run 36502227811; QC PASS |
| 09 | unknown | unknown | Model supplied an overlapping/empty optional take comparison |
| 10 | 10.620 | 11.420 | Render QC PASS; editorial failure |

In 06, 4.21–15.09 s was rescued despite a `failed_delivery` model label by a
broad `audience` observation (2–39 s); 63.68–79.20 s was selected directly by
the model. In 10, a candidate spanning 8.95–24.77 s was discarded as a
redundant retry, despite overlapping owner Gold. These are distinct selection
issues and need audiovisual review before changing editorial authority.

The next transport correction treats a logically impossible optional
competition as advisory data to omit and audit, preserving a complete set of
candidate decisions. The admission ceiling for a preflighted V2 AV input is
raised from 64k to 96k; the existing per-call and per-session dollar checks
still decide affordability before generation. Both changes require a targeted
live qualification of 07 and 09; passing unit tests alone does not establish
their editorial quality.

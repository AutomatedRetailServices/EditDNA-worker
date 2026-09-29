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
- Paid V2 qualification run `36502227811` was triggered by the branch update;
  its result must be reviewed before claiming video quality improvement.

This is a calibration change, not a release-quality certification. Gold remains
an evaluation artifact and is not passed to production selection.

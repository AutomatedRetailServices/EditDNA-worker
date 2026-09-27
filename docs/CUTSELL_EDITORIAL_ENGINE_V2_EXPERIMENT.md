# CutSell Editorial Engine V2 experiment

Status: implemented behind an OFF-by-default flag; offline qualification in
progress; no production activation or paid 10-video run authorized by this file.

## Purpose

V2 tests the architecture approved for the CutSell Raw-to-Ready editor without
replacing the qualified V1 engine. It makes verified whole-source audiovisual
understanding and one complete whole-video editorial plan the only semantic
selection seam.

## Runtime contract

Enable with `CUTSELL_EDITORIAL_ENGINE_V2=1`. The runtime also requires:

- `CUTSELL_HYBRID_LLM_ENABLED=1`;
- approved Google provider configuration and `GEMINI_API_KEY`;
- `CUTSELL_WATCH_LISTEN_AV_ENABLED=1`;
- explicit positive native-AV budget and conservative pricing values;
- a complete validated decision for every candidate.

Missing evidence, malformed/partial model output, provider failure, an empty
candidate universe or a post-freeze semantic mutation aborts the V2 run. There
is no silent legacy fallback.

## Authority order

`Perceive → Understand → Reconstruct → Clean Cut/Best Take → Compose → Resolve
KEEP/DISCARD → Freeze → Boundary → Post-render review`

Upstream perception and candidate construction may produce hypotheses and local
measurements. The whole-video reasoner resolves the complete universe. SWAP is
folded into DISCARD before freeze. After freeze, Boundary may change physical
timing/splits only; it may not change the ordered spoken token stream, recreate
alternates or change discarded membership.

Post-render review remains a blocking validator. A future controlled-repair
implementation may reopen only the named responsible family and must run a new
resolution/freeze cycle. It may never restore a clip directly.

## A/B acceptance

Run the same ten authorized raw sources through stable V1 and V2 using identical
ASR/media inputs. Record per source:

- source identity and build SHA;
- provider/model and cost;
- complete selection audit and freeze signature;
- rendered MP4 and technical QC;
- fumbles/retries/dead air remaining;
- lost unique or critical information;
- duplicate ideas;
- boundary defects;
- Human Gold overlap where an oracle exists;
- human Watch + Listen postability verdict.

Workflow success alone is not acceptance. No engine promotion occurs until the
rendered outputs are inspected and V2 improves cross-video editorial quality
without source-specific rules or regressions.

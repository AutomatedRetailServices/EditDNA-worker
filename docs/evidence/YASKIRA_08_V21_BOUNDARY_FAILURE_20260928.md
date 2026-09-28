# Yaskira08 V21 — failed before render

Branch commit `2deafc6a`, GitHub Actions run `36422660928`, artifact `10970755616`. Qualification tests passed. The live execution failed the frozen Selection/Boundary semantic hash and did not produce a preview. This is a deliberate safety block, not an accepted edit.

The frozen Selection contained both source 101.57–105.65 and 103.01–116.55, overlapping by 2.64 seconds. The first has distinct speech at 101.57–103.01; the second has distinct speech after 105.65. Boundary split the original overlap, and the resulting token sequence differed from the frozen sequence. The V21 failure artifact contains the selected clips and Boundary diagnostics, but omitted unified reasoner decisions. The selected portion shown in that artifact ended at 116.55; Gold 120–147 was absent from the frozen Selection. This is an additional unresolved editorial regression, not a verified fix to the 95.30 seconds of unwanted V20 material.

V22 attempts to coalesce only identically aligned shared words before Freeze, retaining the unique leading and trailing speech and exposing reasoner diagnostics in any subsequent Boundary failure. A live render and Gold comparison remain mandatory before claiming quality. Do not treat the V21 failure as an improved score.

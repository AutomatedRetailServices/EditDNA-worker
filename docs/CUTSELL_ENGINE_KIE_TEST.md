# Engine model call: Anthropic vs kie.ai, Sonnet 5.5 vs Opus 5.5 (16 Nov 2026)

PO question: can the engine run through kie.ai (a reseller) to save money? PO rule: "duplicate the code, adapt the
copy; if it does not work the original is not touched". The copy is `cutsell_worker/simple_engine_kie/` (branch
`feature/engine-kie`, never deployed); the production engine is unchanged.

What kie.ai does differently (probe, 16 Nov): ignores `output_config.format` (JSON schema), ignores prompt caching
(writes every time, never reads), adds ~1,600 tokens of its own, ~15 s per call. The copy writes the schema into the
question and extracts the JSON (0 re-asks needed in every run).

Gold benchmark (benchmarks/simple_engine_gold/gold_v1.json), decision + reviews + word fix, remembered answers off:

| Run | Videos | Gold agreement | Good parts cut by mistake | Extra kept | Cost | Time |
|---|---|---|---|---|---|---|
| Sonnet 5.5 · Anthropic | 27 | 567/611 (93 %) | 24 | 20 | US$1.28 | 530 s |
| **Opus 5.5 · Anthropic** | 27 | **584/611 (96 %)** | **15** | **12** | US$1.31 | 463 s |
| Opus 5.5 · kie.ai | 27 | 566/611 (93 %) | 21 | 24 | US$2.39 | 2,382 s |
| Sonnet 5.5 · Anthropic | 5 | 46/48 | 2 | 0 | US$0.146 | 66 s |
| Sonnet 5.5 · kie.ai | 5 | 42/48 | 6 | 0 | US$0.082 | 541 s |
| Opus 5.5 · Anthropic | 5 | 46/48 | 0 | 2 | US$0.179 | 65 s |
| Opus 5.5 · kie.ai | 5 | 45/48 | 0 | 3 | US$0.159 | 389 s |

Conclusion: kie.ai is discarded for the engine (worse cuts, 5-8x slower, and on the full set MORE expensive because it
has no cache). Opus 5.5 on Anthropic cuts better than Sonnet 5.5 at ~+3 % cost: proposed to the PO (switch =
`CUTSELL_SIMPLE_ENGINE_MODEL` / `DEFAULT_MODEL`). Runs: GitHub Actions `cutsell-engine-kie-compare.yml`
(43903888684, 43912476772, 43921163281, 43927111979). Total paid for the tests ~US$5.65.

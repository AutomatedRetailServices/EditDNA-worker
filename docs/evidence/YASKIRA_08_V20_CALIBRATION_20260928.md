# Yaskira08 V20 — provisional labels withheld

Commit `6fa803e7`, run `36420143783`, artifact `10969262149`. Changes
relative to V19: V2 Gemini request omits prior `current_bucket` and Hybrid
votes from candidate and take-group rows. Prompt's existing CTA instruction
and production protections otherwise retain their prior behavior. Both
runs use the same Yaskira08 source, but their live ASR/candidate boundaries
can vary; results are a controlled code hypothesis, not proof of causality.

| Measure | V19 | V20 |
|---|---:|---:|
| Gold approved seconds retained | 24.11 | 26.52 |
| Gold seconds lost | 2.89 | 0.48 |
| Unwanted seconds retained | 98.61 | 95.30 |
| Gold recall / precision | 89.30% / 19.65% | 98.22% / 21.77% |
| Technical delivery | BLOCKED (silence) | PENDING HUMAN WATCH + LISTEN |

V20 selected 19.25–116.55 plus 120.39–146.91. The CTA 145.1–146.91
survived, and the previously blocking silence fell outside the new selected
end at 116.55. The core editorial failure remains: 95.30 seconds outside
Gold, mostly earlier attempts. Two earlier candidates whose model action
was `discard/redundant_retry` were put back by
`unique_retry_information_preserved` based on vocabulary. Other earlier
candidates were selected by the model itself. Removing prior buckets is
therefore insufficient to make complete attempts compete correctly.

The next general correction must (a) distinguish a material fact/action from
lexical novelty when safeguarding a discarded retry; and (b) have a model
compare the later full delivery against the *union* of prior fragments while
allowing true unique information and legitimate composites. No existing
Gold timestamp enters production. Re-run V20 unchanged only for an explicit
stability objective; no blind paid retries. No 01–10 release regression yet.

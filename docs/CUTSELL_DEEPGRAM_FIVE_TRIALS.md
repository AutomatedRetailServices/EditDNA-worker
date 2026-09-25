# Video00: five Deepgram full-engine trials

Run 36070640917; tested a5fa3918d69d89d110dac02e740a005706ca27ca.
Manual-only workflow restored at ba46c72a942dea13d07c7bced5c981f4a37bd8cb.
Five sequential independent trials; frozen source/config/build, no retries,
no cross-trial transcript cache. All engine calls and full media decode checks
succeeded; all Modal apps stopped. Aggregate editorial gate failed as designed.

| Trial | Output seconds | Editorial /11 | Words |
|---|---:|---:|---:|
| 1 | 160.467 | 7 | 652 |
| 2 | 150.667 | 10 | 651 |
| 3 | 155.567 | 7 | 651 |
| 4 | 148.834 | 7 | 652 |
| 5 | 155.167 | 7 | 651 |

All five: NEEDS_HUMAN_REVIEW, diagnostic output, not approved delivery.
Abandoned stomach attempt remains in 5/5; percentage restatement and both
closing repetition checks fail in 4/5. No trial passes all eleven criteria.
Five distinct Deepgram requests; one within-job cache hit each; no zero-duration
words. Artifact ZIPs, MP4 parts and reconstructed MP4 SHA-256 values verified.

Compared with existing GPT run 36054192894: mean editorial criteria 7.6 vs 6.4;
selected-source coverage IoU range 91.393–100% vs 72.821–90.996%; maximum
within-provider lexical disagreement 0.3067% vs 0.7764%. Deepgram performs
better on these criteria and is more consistent on this source. These are
not word-accuracy percentages, human audiovisual acceptance or sales metrics.
Same editorial code, different provider/timing/segmentation and whole runtime
snapshots: this experiment cannot isolate a causal ASR quality improvement.
No human listening or reference transcript; no production provider promotion.

The Product Owner considers the prior individual Deepgram output a useful
Clean Cut. Preserve this assessment alongside technical findings. A future
integrated sales layer may improve source-supported hook, benefit progression
and CTA; no funnel change or conversion claim is established by this batch.

Reproduce CPU-only comparison with benchmarks/report_deepgram_vs_gpt_five.py
using recovered Deepgram trial folders and existing GPT trial folders.

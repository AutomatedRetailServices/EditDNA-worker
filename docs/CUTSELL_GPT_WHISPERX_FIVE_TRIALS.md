# GPT + WhisperX: five full-engine Video00 trials

Completed 2026-09-24, 20:20–20:59 UTC. User explicitly authorized five new
trials: “podrias hacerlo otra vez? hacer 5 pruebas con este? sistema?”
Status: **5 full engine executions, 5 diagnostic MP4s, 0 quality-approved
deliverables. Do not promote to production.** No sixth GPU run occurred.

## Reproducibility and execution bounds

- Full batch: https://github.com/AutomatedRetailServices/EditDNA-worker/actions/runs/36054192894
- Exact tested SHA: `99c02183246c3e2caa96c006ad81e7c966ba4a51`.
- Engine code is the `e1d6892` GPT + WhisperX provider with the per-job source
  cache. The batch adds transport/QA only; no editorial module changed.
- Worker package fingerprint, identical in all five:
  `71db0555a03f53c22db47247e40762daaa67e9882a1a38997110de4c425ea59e`.
- Original SHA-256, verified in all five:
  `b37059b1790cf3eb0447bb54595cc99f6668aadc5f65dafa9cdf6181621ef9f5`.
- One frozen private worker-template configuration, same full canonical
  `run_op("focused")` payload and auto-microtrim setting, sequential fresh
  Modal apps/providers, unique benchmark/job identities, L4, retries=0,
  5400-second remote bound. All five logs confirm app completion/stop.
- ASR fingerprint in all five: `asrcfg_0ad52df19a8fccba`. WhisperX 3.8.6,
  Torch/TorchAudio 2.8.0, Numpy 2.2.6, Transformers 4.57.6.
- Both temporary launch triggers were restored to manual-only. No changes
  to `main`, canonical release branch, PR #25, or production defaults.

## Results

| Trial | Words | Selected clips | MP4 seconds | Editorial criteria passed | Engine seconds | Render/QC |
|---|---:|---:|---:|---:|---:|---|
| 1 | 643 | 22 | 156.667 | 6/11 | 456.007 | NEEDS_HUMAN_REVIEW |
| 2 | 644 | 20 | 151.700 | 5/11 | 389.283 | NEEDS_HUMAN_REVIEW |
| 3 | 644 | 20 | 146.734 | 8/11 | 435.918 | NEEDS_HUMAN_REVIEW |
| 4 | 642 | 21 | 149.667 | 5/11 | 460.037 | NEEDS_HUMAN_REVIEW |
| 5 | 643 | 20 | 139.934 | 8/11 | 440.728 | NEEDS_HUMAN_REVIEW |

All five engine responses have `ok=true` and a full result JSON. All five
MP4s are explicitly **diagnostic-invalidated**, with `preview_uri=null`,
`deliverable=false`, Watch+Listen BLOCKED and physical silence findings.
There are 1, 3, 2, 2 and 1 lingering-silence findings respectively, about
1.50–2.81 seconds. They must not be presented as publish-ready edits.
All five files were recovered byte-for-byte, hashes verified, and decoded
successfully with local ffmpeg. Sample frames were inspected; no human
audio listening occurred.

The same three editorial failures occur in **5/5**: abandoned stomach
attempt retained, full gynecologist take omitted, percentage restatement
retained. The failed long pimples take remains in 3/5; the preferred later
take is missing in 2/5. Historical Gold passes 17/18 in all five, illustrating
why that older oracle alone does not establish editorial acceptability.
The live Selection/Boundary contract is verified and matches the reviewed
plan in all five; this is not a post-Freeze mutation.

## Stability and ASR observations

- Zero zero-duration words and zero nonfinite/nonpositive/out-of-bounds
  word intervals in all five. Long and very short intervals remain;
  positive duration alone does not establish phoneme-boundary accuracy.
- Five exact text variants, four lexical variants when case/punctuation
  are ignored, five timed transcripts and five different selections.
- Pairwise lexical token edit difference: **0–0.7764%**. This is disagreement
  between outputs, **not WER or accuracy against a human transcript**.
- Pairwise selected-source interval intersection/union: **72.821–90.996%**.
  Trial 1 and 5 have identical lexical tokens but only 79.587% selected
  interval overlap; punctuation/segmentation, job-derived IDs and editorial
  model behavior are still variable. These trials do not isolate one cause.
- 13 recorded GPT request IDs per run, **65 distinct requests total**;
  cache-hit count is exactly 1 in every run. The redundant same-source
  pre-Freeze pass is reused within each job, with no transcript reuse
  between trials. No billing query or exact dollar-cost claim.
- Word confidence remains a CTC alignment score, not GPT lexical
  confidence. Post-render cleanup retains the existing Medium component;
  blocked candidates do not run that cleanup.

## QA transport incident and recovery

The Ubuntu runner lacked `ffprobe`. Each transport step collected the full
engine JSON, editorial/Gold checks and diagnostic MP4, then hit
`FileNotFoundError` during the additional CPU media probe. Consequently the
original transport summary says `status=failed` and the batch aggregate's
`all_engine_runs_completed=false`; these are harness outcomes, not proof of
failed GPU execution. The full workflow conclusion is FAILURE and is not
relabeled green. Original reports are retained unchanged.

Read-only CPU collector run `36055467262` (SUCCESS) downloaded existing
artifacts and split the exact MP4 bytes into 20 MiB transfer parts, avoiding
the workspace's 32 MiB per-file transfer limit. No editor, GPU or ASR was
invoked. Local probes/decode and source/build/ASR comparisons recovered the
missing verification, recorded separately from the original summaries.
The harness now installs ffmpeg and checks both ffmpeg/ffprobe before any
paid dispatch; eight focused transport tests pass. This post-batch harness
fix was not tested with an extra GPU run and changes no editor behavior.

User deliverables: five explicitly labeled diagnostic MP4s,
`Video00_GPT_WhisperX_5_Pruebas.html` (comparison and embedded original JSON
evidence ZIP), and `Video00_GPT_WhisperX_5_Resultados.json`. Evidence applies
only to Video00; human Watch+Listen and cross-video qualification remain open.

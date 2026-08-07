# V1 controlled + historical benchmark runbook

This runbook closes the real-video validation gate for PR #23. It deliberately
uses the existing private benchmark API and S3 allowlist; it does not add a
second benchmark path or expose credentials.

## Preconditions

- Run the API + worker from `repair/v1-structural-recovery`.
- Configure `BENCHMARK_INTERNAL_API_KEY` on the API.
- Configure the existing Redis, S3 and OpenAI environment values on the worker.
- Keep benchmark output rendering off for the first analysis pass.
- Dataset: `editdna/training/take_judge_dataset.jsonl`.
- Source prefixes are limited to the allowlisted prefixes in `benchmark_s3.py`.

## 1. Controlled session

Submit one session first:

```json
{
  "source_prefixes": [
    "Editdna bloopers videos/",
    "Editdna good videos/"
  ],
  "dataset_key": "editdna/training/take_judge_dataset.jsonl",
  "mode": "old_vs_new",
  "limit": 1,
  "render_outputs": false,
  "use_semantic_v2": true,
  "use_take_judge_v2": true
}
```

POST the body to `/benchmark/run` with the configured `X-Internal-API-Key`.
Poll `/benchmark/jobs/{job_id}` and read `/benchmark/jobs/{job_id}/results`
only after the job reaches `finished`.

The controlled run must be inspected before the full historical run. Confirm:

- no cross-source/take merge;
- no impossible microfragment;
- no adjacent residual duplicate;
- valid repeated takes remain available rather than being deleted;
- slot classification never changes Clean Cut validity;
- Semantic V2 execution/fallback is visible;
- Take Judge execution/fallback is visible;
- selected clips still form a coherent source-time edit.

## 2. Historical benchmark

After the controlled session passes, run the same request with `limit` omitted:

```json
{
  "source_prefixes": [
    "Editdna bloopers videos/",
    "Editdna good videos/"
  ],
  "dataset_key": "editdna/training/take_judge_dataset.jsonl",
  "mode": "old_vs_new",
  "render_outputs": false,
  "use_semantic_v2": true,
  "use_take_judge_v2": true
}
```

The benchmark is resumable. Do not start a duplicate request while an identical
job is queued or running; the API already deduplicates identical active jobs.

## 3. Required evidence for PR #23

Attach or record the final `summary.json` plus targeted review of
`disagreements.jsonl` and `unmatched.jsonl`. The merge gate is not satisfied by
a green unit-test suite alone.

At minimum record:

- total / resolved / unresolved sessions;
- matched and unmatched clip counts;
- old-keep/new-discard and old-discard/new-keep counts;
- semantic V2 usage/fallback counts and execution status;
- Take Judge usage/fallback and execution status counts;
- failed sessions and errors;
- processing time;
- manual findings for cross-source merges, microfragments and residual duplicates.

If a regression is found, fix the smallest owning layer. Do not widen the
English/Spanish fallback or introduce a new commercial ordering doctrine to
make benchmark numbers look better.

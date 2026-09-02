"""D-044 (retry-family / idea-clustering regression audit): pure, read-only
extraction of the pipeline-stage diagnostics needed to forensically trace a
specific semantic outcome (which clip survived, which was discarded, and at
which stage the decision was made) from an already-produced result.json.

Never runs the pipeline, never calls a provider, never touches Modal/RunPod
infrastructure -- this only reads a JSON file already sitting on disk
(downloaded from S3 by the caller) and re-shapes a bounded subset of its
`diagnostics` dict into a smaller, forensic-focused JSON. Built because the
GitHub Actions CI log route is lossy for this purpose: GitHub's own
`::add-mask::` log redaction blanks out any digit sequence that happens to
coincide with a masked secret/config value printed earlier in the same job,
which corrupts exact clip_id/timestamp/count reporting. Reading the file
directly (as this script does, run inside the fetching job before anything
is echoed to the log) never has that problem -- the file itself is never
masked, only console output is.

Fields sourced directly from cutsell_worker.pipeline's own DraftTimeline
diagnostics dict (see pipeline.py's `draft = DraftTimeline(..., diagnostics=
{...})` literal) -- this script does not invent new diagnostic keys, it only
selects and re-shapes existing ones:
  - attempt_reconstruction: AttemptReconstructor's own output
  - take_grouping_reason / take_group_members: IdeaClusterer's grouping
  - semantic_idea_equivalence: the bounded SemanticArbiter tier's decisions
  - take_judge_groups: DeliveryScorer's RankedTake scores per retry family
  - clean_cut_decisions: per-candidate keep/discard decisions + reasons
  - canonical_edit_plan.ideas: CanonicalEditPlan's own idea/winner/discard map
  - final_story_coherence_validation: StoryValidator's findings
  - selected / discarded: the final clip lists with clip_id + text
"""
from __future__ import annotations

import json
import sys


def extract(result_path: str, keywords: list[str] | None = None) -> dict:
    with open(result_path, "r", encoding="utf-8") as fh:
        result = json.load(fh)

    diagnostics = result.get("diagnostics") or {}
    keywords = [k.lower() for k in (keywords or [])]

    def _matches(text: str) -> bool:
        if not keywords:
            return True
        low = text.lower()
        return any(k in low for k in keywords)

    def _filter_list(items, text_keys: tuple[str, ...]) -> list:
        if not keywords:
            return list(items)
        out = []
        for item in items:
            if not isinstance(item, dict):
                out.append(item)
                continue
            blob = " ".join(str(item.get(k, "")) for k in text_keys)
            if _matches(blob):
                out.append(item)
        return out

    forensic = {
        "schema_version": "cutsell.video00.d044_forensic_extract.v1",
        "benchmark_id": result.get("benchmark_id"),
        "selected_count": result.get("selected_count"),
        "source_duration_sec": result.get("source_duration_sec"),
        "selected": [
            {"clip_id": c.get("clip_id"), "text": c.get("text")}
            for c in (result.get("selected") or [])
        ],
        "discarded": [
            {"clip_id": c.get("clip_id"), "text": c.get("text")}
            for c in (result.get("discarded") or [])
        ],
        "attempt_reconstruction": diagnostics.get("attempt_reconstruction"),
        "take_grouping_status": diagnostics.get("take_grouping_status"),
        "take_grouping_reason": diagnostics.get("take_grouping_reason"),
        "take_group_count": diagnostics.get("take_group_count"),
        "alternate_group_count": diagnostics.get("alternate_group_count"),
        "take_group_members": diagnostics.get("take_group_members"),
        "semantic_idea_equivalence": diagnostics.get("semantic_idea_equivalence"),
        "take_judge_status_counts": diagnostics.get("take_judge_status_counts"),
        "take_judge_groups_filtered": _filter_list(
            diagnostics.get("take_judge_groups") or [], ("clip_id", "text", "reason", "group_id", "semantic_key")
        ),
        "clean_cut_decisions_filtered": _filter_list(
            diagnostics.get("clean_cut_decisions") or [], ("clip_id", "reason")
        ),
        "hybrid_editorial_chunks_filtered": _filter_list(
            diagnostics.get("hybrid_editorial_chunks") or [], ("clip_id", "text", "reason")
        ),
        "claim_coverage_best_take": diagnostics.get("claim_coverage_best_take"),
        "final_selection_retry_arbiter": diagnostics.get("final_selection_retry_arbiter"),
        "canonical_edit_plan_ideas": (diagnostics.get("canonical_edit_plan") or {}).get("ideas"),
        "final_story_coherence_validation": diagnostics.get("final_story_coherence_validation"),
        "post_selection_complementary_family_stabilizer": diagnostics.get("post_selection_complementary_family_stabilizer"),
        "post_selection_composite_handoff_trim": diagnostics.get("post_selection_composite_handoff_trim"),
    }
    return forensic


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: video00_d044_forensic_extract.py RESULT_JSON [keyword ...]", file=sys.stderr)
        return 2
    result_path = sys.argv[1]
    keywords = sys.argv[2:] or None
    forensic = extract(result_path, keywords)
    print(json.dumps(forensic, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())

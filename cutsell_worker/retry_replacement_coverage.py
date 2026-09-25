"""Check a proposed retry replacement before any removal; never delete here.

Uses the existing claim/contradiction contracts. Coverage is directional:
every claim from the abandoned attempt must survive in the replacement.
No inference that a later take is better, and no union of several retakes.
"""
from .semantic_claims import extract_claims, claim_is_covered
from .final_sibling_grouping import _numbers
from .contradiction_signal import any_pair_contradicts
from .complete_retry_identity_guard import prior_replacement_rejections


def replacement_semantics(windows):
    """Positive evidence across ALL returned windows, not a last-row winner.

    This only makes a peer available for comparison. Identity, completeness,
    chronology, coverage and the owning authority's thresholds still apply.
    Missing roles remain compatible with historical providers, but an explicit
    mixed/failure/uncertain decision in any window cannot be ignored.
    """
    by_id = {}
    for window in windows:
        for row in window.get('decisions', ()):
            by_id.setdefault(row['clip_id'], []).append(row)
    result = {}
    for cid, rows in by_id.items():
        if any(r.get('content_role') in {'mixed', 'recording_only'}
               or r.get('label') not in {'winner', 'keep', 'alternate'} for r in rows):
            continue
        result[cid] = ('winner' if any(r['label'] == 'winner' for r in rows) else 'keep',
                       min(float(r.get('confidence', 0)) for r in rows))
    return result


def replacement_coverage(failed, winner, session_diagnostics=()):
    result = {
        "candidate_clip_id": failed.clip_id,
        "proposed_replacement_id": winner.clip_id,
        "coverage_verified": False,
        "uncovered_claims": [],
    }
    reason = None
    session_diagnostics = tuple(session_diagnostics)
    if prior_replacement_rejections(session_diagnostics).get(failed.clip_id) == winner.clip_id:
        reason = "prior_replacement_rejection_respected"
    elif winner.source_asset_id != failed.source_asset_id or winner.start < failed.end:
        reason = "source_or_chronology_mismatch"
    elif not winner.complete_idea:
        reason = "replacement_incomplete"
    else:
        rows = [r for window in session_diagnostics for r in window.get("decisions", ())
                if r.get("clip_id") == winner.clip_id]
        if any(r.get("content_role") in {"mixed", "recording_only"} for r in rows):
            reason = "replacement_contains_recording_process"
        elif any(r.get("label") in {"failed", "bts"} for r in rows):
            reason = "replacement_has_conflicting_failure_evidence"
        elif not _numbers(failed.text).issubset(_numbers(winner.text)):
            reason = "replacement_missing_numeric_fact"
        elif any_pair_contradicts([failed.text, winner.text]):
            reason = "replacement_contradicts_attempt"
        else:
            claims = extract_claims(failed.clip_id, failed.text, split_clauses=False)
            result["uncovered_claims"] = [c.text for c in claims if not claim_is_covered(c, winner.text)]
            if not claims:
                reason = "insufficient_claim_evidence"
            elif result["uncovered_claims"]:
                reason = "replacement_does_not_cover_attempt"
    result["coverage_verified"] = reason is None
    result["reason"] = reason or "attempt_claims_covered_by_complete_replacement"
    return result


def review_retry_pool(takes, windows):
    """Review the collected window evidence before downstream resolution.

    This inventory grants no deletion authority. It exposes cross-window
    peers and blocked comparisons, using the existing retry relation test.
    """
    from .hybrid_retry_winner_authority import _same_retry_attempt
    rows = {}
    for window in windows:
        for row in window.get("decisions", ()):
            rows.setdefault(row.get("clip_id"), []).append(row)
    reviews = []
    for failed in takes:
        if not any(r.get("label") == "failed" and r.get("confidence", 0) >= .8
                   for r in rows.get(failed.clip_id, ())):
            continue
        comparisons = []
        for peer in takes:
            if peer.source_asset_id != failed.source_asset_id or not 0 <= peer.start - failed.end <= 24:
                continue
            if not any(r.get("label") in {"winner", "keep", "alternate"} and r.get("confidence", 0) >= .68
                       for r in rows.get(peer.clip_id, ())):
                continue
            same, relation = _same_retry_attempt(failed, peer)
            if not same:
                continue
            comparisons.append({**replacement_coverage(failed, peer, windows), "relation_evidence": relation})
        reviews.append({"candidate_clip_id": failed.clip_id, "comparisons": comparisons,
                        "status": "compared" if comparisons else "no_supported_nearby_replacement",
                        "authority": "comparison_only"})
    return reviews

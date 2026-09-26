"""Check a proposed retry replacement before any removal; never delete here.

Uses the existing claim/contradiction contracts. Coverage is directional:
every claim from the abandoned attempt must survive in the replacement.
No inference that a later take is better, and no union of several retakes.
"""
from .semantic_claims import extract_claims, claim_is_covered
from .final_sibling_grouping import _numbers
from .contradiction_signal import any_pair_contradicts
from .complete_retry_identity_guard import prior_replacement_rejections

_NUMBER_WORDS = {
    "zero": "0", "cero": "0", "one": "1", "uno": "1", "una": "1", "un": "1",
    "two": "2", "dos": "2", "three": "3", "tres": "3", "four": "4", "cuatro": "4",
    "five": "5", "cinco": "5", "six": "6", "seis": "6", "seven": "7", "siete": "7",
    "eight": "8", "ocho": "8", "nine": "9", "nueve": "9", "ten": "10", "diez": "10",
    "eleven": "11", "once": "11", "twelve": "12", "doce": "12", "thirteen": "13",
    "trece": "13", "fourteen": "14", "catorce": "14", "fifteen": "15", "quince": "15",
    "sixteen": "16", "dieciséis": "16", "seventeen": "17", "diecisiete": "17",
    "eighteen": "18", "dieciocho": "18", "nineteen": "19", "diecinueve": "19",
    "twenty": "20", "veinte": "20", "thirty": "30", "treinta": "30",
    "forty": "40", "cuarenta": "40", "fifty": "50", "cincuenta": "50",
    "sixty": "60", "sesenta": "60", "seventy": "70", "setenta": "70",
    "eighty": "80", "ochenta": "80", "ninety": "90", "noventa": "90",
    "hundred": "100", "cien": "100", "ciento": "100", "thousand": "1000", "mil": "1000",
}


def _number_facts(text):
    """Normalize numeric digits and common spoken EN/ES number words.

    Keep the existing digit extractor for percentages/ranges (e.g. 5-10%) and
    add spoken tokens so a retake changing "three" to "two" cannot pass merely
    because neither transcript contains digits.
    """
    import re
    words = {token.casefold() for token in re.findall(r"[a-z0-9áéíóúñü]+", str(text or ""), re.I)}
    return _numbers(text) | {_NUMBER_WORDS[word] for word in words if word in _NUMBER_WORDS}


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
        authoritative_winner = any(
            r.get('label') == 'winner'
            and r.get('content_role') == 'audience'
            and float(r.get('confidence', 0)) >= .95
            for r in rows
        )
        conflicting_failure = any(r.get('label') in {'failed', 'bts'} for r in rows)
        compatible_shadow = all(
            r.get('label') in {'winner', 'keep', 'alternate'}
            or (r.get('label') == 'uncertain' and r.get('content_role') == 'mixed')
            for r in rows
        )
        conflicting_role = any(
            r.get('content_role') in {'mixed', 'recording_only'}
            and not (r.get('label') == 'uncertain' and r.get('content_role') == 'mixed')
            for r in rows
        )
        if conflicting_failure or not compatible_shadow or conflicting_role or (
            not authoritative_winner
            and any(r.get('content_role') in {'mixed', 'recording_only'} for r in rows)
        ):
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
        authoritative_winner = any(
            r.get("label") == "winner"
            and r.get("content_role") == "audience"
            and float(r.get("confidence", 0)) >= .95
            for r in rows
        )
        incompatible_shadow = any(
            r.get("label") not in {"winner", "keep", "alternate"}
            and not (r.get("label") == "uncertain" and r.get("content_role") == "mixed")
            for r in rows
        )
        conflicting_role = any(
            r.get("content_role") in {"mixed", "recording_only"}
            and not (r.get("label") == "uncertain" and r.get("content_role") == "mixed")
            for r in rows
        )
        if incompatible_shadow or (authoritative_winner and conflicting_role):
            reason = "replacement_not_consistently_usable"
        elif conflicting_role:
            reason = "replacement_contains_recording_process"
        elif (not authoritative_winner
                and any(r.get("content_role") in {"mixed", "recording_only"} for r in rows)):
            reason = "replacement_contains_recording_process"
        elif any(r.get("label") in {"failed", "bts"} for r in rows):
            reason = "replacement_has_conflicting_failure_evidence"
        elif not _number_facts(failed.text).issubset(_number_facts(winner.text)):
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
    partitions = {}
    for window in windows:
        partition = window.get("partition_index")
        if type(partition) is int:
            for cid in window.get("member_ids", ()):
                partitions.setdefault(cid, set()).add(partition)
        for row in window.get("decisions", ()):
            rows.setdefault(row.get("clip_id"), []).append(row)
    reviews = []
    for failed in takes:
        if not any(r.get("label") == "failed" and r.get("confidence", 0) >= .8
                   for r in rows.get(failed.clip_id, ())):
            continue
        comparisons = []
        for peer in takes:
            if peer.clip_id == failed.clip_id or peer.source_asset_id != failed.source_asset_id:
                continue
            if not any(r.get("label") in {"winner", "keep", "alternate"} and r.get("confidence", 0) >= .68
                       for r in rows.get(peer.clip_id, ())):
                continue
            left_partition, right_partition = partitions.get(failed.clip_id), partitions.get(peer.clip_id)
            if left_partition and right_partition and (
                    len(left_partition) != 1 or left_partition != right_partition):
                comparisons.append({
                    "candidate_clip_id": failed.clip_id,
                    "proposed_replacement_id": peer.clip_id,
                    "coverage_verified": False,
                    "reason": "creator_session_partition_mismatch",
                    "comparison_status": "blocked_before_relation_test",
                    "authority": "comparison_only",
                })
                continue
            gap = float(peer.start - failed.end)
            if gap < 0:
                comparisons.append({
                    "candidate_clip_id": failed.clip_id,
                    "proposed_replacement_id": peer.clip_id,
                    "coverage_verified": False,
                    "reason": "chronology_order",
                    "gap_sec": round(gap, 3),
                    "failed_start_sec": round(float(failed.start), 3),
                    "failed_end_sec": round(float(failed.end), 3),
                    "peer_start_sec": round(float(peer.start), 3),
                    "peer_end_sec": round(float(peer.end), 3),
                    "peer_complete_idea": bool(peer.complete_idea),
                    "comparison_status": "blocked_before_relation_test",
                    "authority": "comparison_only",
                })
                continue
            # Observe all later candidates in the same creator-session
            # partition.  The relation and claim-coverage checks below remain
            # mandatory; chronology distance is diagnostic, not a veto.
            same, relation = _same_retry_attempt(failed, peer)
            if not same:
                comparisons.append({
                    "candidate_clip_id": failed.clip_id,
                    "proposed_replacement_id": peer.clip_id,
                    "coverage_verified": False,
                    "reason": "relation_test",
                    "relation_evidence": relation,
                    "gap_sec": round(gap, 3),
                    "failed_start_sec": round(float(failed.start), 3),
                    "failed_end_sec": round(float(failed.end), 3),
                    "peer_start_sec": round(float(peer.start), 3),
                    "peer_end_sec": round(float(peer.end), 3),
                    "peer_complete_idea": bool(peer.complete_idea),
                    "comparison_status": "relation_not_established",
                    "authority": "comparison_only",
                })
                continue
            comparisons.append({**replacement_coverage(failed, peer, windows),
                "relation_evidence": relation, "gap_sec": round(gap, 3),
                "failed_start_sec": round(float(failed.start), 3),
                "failed_end_sec": round(float(failed.end), 3),
                "peer_start_sec": round(float(peer.start), 3),
                "peer_end_sec": round(float(peer.end), 3),
                "peer_complete_idea": bool(peer.complete_idea),
                "comparison_status": "coverage_checked", "authority": "comparison_only"})
        status = ("coverage_checked" if any(c.get("comparison_status") == "coverage_checked"
                                             for c in comparisons)
                  else "blocked_or_unrelated" if comparisons
                  else "no_supported_nearby_replacement")
        reviews.append({"candidate_clip_id": failed.clip_id, "comparisons": comparisons,
                        # Preserve the v1 consumer value while exposing the
                        # more precise status separately for newer readers.
                        "status": "compared" if any(
                            c.get("comparison_status") in {"coverage_checked", "relation_not_established"}
                            for c in comparisons
                        ) else status,
                        "comparison_status": status,
                        "authority": "comparison_only"})
    return reviews

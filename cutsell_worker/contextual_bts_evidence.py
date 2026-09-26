"""Carry corroborated recording-context evidence to singleton resolution.

This is not a general bad-take veto. Only consistent BTS judgments may use
the already computed dense failure context, and only at the final singleton
authority. Conflicting or uncertain window judgments preserve the candidate.
"""

import math


def contextual_bts_ids(diagnostics):
    rows_by_id = {}
    for window in diagnostics:
        for row in window.get("decisions") or ():
            cid = row.get("clip_id")
            if cid:
                rows_by_id.setdefault(cid, []).append(row)
    supported = set()
    for cid, rows in rows_by_id.items():
        def consistent(row):
            confidence = row.get("confidence")
            return (row.get("label") == "bts"
                    and type(confidence) in (int, float)
                    and math.isfinite(confidence) and .9 <= confidence <= 1
                    and row.get("semantic_delete_recommended") is True)
        if not all(consistent(row) for row in rows):
            continue
        # The dense-cluster fact is the corroboration.  A higher-confidence
        # BTS label may have been assigned the generic
        # ``high_confidence_semantic`` basis earlier in the cleanup ladder;
        # that precedence must not erase the independently-computed cluster
        # evidence before singleton resolution sees it.  Every row still
        # has to agree on BTS >= .90, recommend deletion, and at least one
        # row must carry the dense-cluster observation.
        if any(row.get("dense_semantic_failure_cluster") is True for row in rows):
            supported.add(cid)
    return frozenset(supported)
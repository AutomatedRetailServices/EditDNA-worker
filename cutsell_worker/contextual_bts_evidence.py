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
        if any(row.get("delete_basis") == "semantic_bts_inside_corroborated_failure_cluster"
               and row.get("dense_semantic_failure_cluster") is True for row in rows):
            supported.add(cid)
    return frozenset(supported)

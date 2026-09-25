"""Source-bound proof for wholly recording-process speech, never mixed content.

Model roles are insufficient alone: every window must agree at high confidence,
local failure must corroborate, and exact source/text/bounds must survive. The
same proof is consumed by cleanup and loss validation; no phrase blacklist.
"""
import hashlib
import math


def identity(clip):
    return {'source_asset_id': clip.source_asset_id, 'start': float(clip.start),
            'end': float(clip.end), 'text_sha256': hashlib.sha256(clip.text.encode()).hexdigest()}


def recording_process_proofs(windows):
    by_id = {}
    for window in windows:
        for row in window.get('decisions') or ():
            by_id.setdefault(row['clip_id'], []).append(row)
    proofs = {}
    for cid, rows in by_id.items():
        if not all(r.get('content_role') == 'recording_only'
                   and r.get('label') in {'bts', 'failed'}
                   and type(r.get('confidence')) in (int, float)
                   and math.isfinite(r['confidence']) and .95 <= r['confidence'] <= 1
                   and r.get('local_failure_corroborated') is True
                   and isinstance(r.get('source_identity'), dict) for r in rows):
            continue
        bound = rows[0]['source_identity']
        if not all(r['source_identity'] == bound for r in rows):
            continue
        proofs[cid] = {**bound, 'basis': 'corroborated_recording_only',
                       'confidence': min(r['confidence'] for r in rows), 'window_count': len(rows)}
    return proofs


def proof_for_clip(clip, proofs):
    proof = proofs.get(clip.clip_id)
    if not isinstance(proof, dict) or proof.get('basis') != 'corroborated_recording_only':
        return None
    return proof if all(proof.get(k) == v for k, v in identity(clip).items()) else None


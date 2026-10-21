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


def _recording_confidence(row):
    value = row.get('recording_confidence')
    return row.get('confidence') if value is None else value


def _whole_video_recording_only(row):
    """Independent source-level AV evidence that the whole span is setup/BTS."""
    audiovisual = row.get('audiovisual')
    if not isinstance(audiovisual, dict) or audiovisual.get('status') != 'overlap':
        return False
    bound = row.get('source_identity') or {}
    start, end = bound.get('start'), bound.get('end')
    if type(start) not in (int, float) or type(end) not in (int, float):
        return False
    for observation in audiovisual.get('observations') or ():
        confidence = observation.get('confidence')
        if not (
            observation.get('role') == 'recording_only'
            and type(confidence) in (int, float)
            and math.isfinite(confidence)
            and confidence >= .85
        ):
            continue
        source_start = observation.get('source_start')
        source_end = observation.get('source_end')
        if type(source_start) in (int, float) and type(source_end) in (int, float):
            if source_start <= start and end <= source_end:
                return True
    return False


def recording_process_proofs(windows):
    by_id = {}
    for window in windows:
        for row in window.get('decisions') or ():
            by_id.setdefault(row['clip_id'], []).append(row)
    proofs = {}
    for cid, rows in by_id.items():
        locally_corroborated = all(r.get('content_role') == 'recording_only'
                   and r.get('label') in {'bts', 'failed'}
                   and type(_recording_confidence(r)) in (int, float)
                   and math.isfinite(_recording_confidence(r)) and .95 <= _recording_confidence(r) <= 1
                   and r.get('local_failure_corroborated') is True
                   and isinstance(r.get('source_identity'), dict) for r in rows)
        whole_video_corroborated = all(r.get('content_role') == 'recording_only'
                   and r.get('label') in {'bts', 'failed'}
                   and type(_recording_confidence(r)) in (int, float)
                   and math.isfinite(_recording_confidence(r)) and .85 <= _recording_confidence(r) <= 1
                   and isinstance(r.get('source_identity'), dict)
                   and _whole_video_recording_only(r) for r in rows)
        if not (locally_corroborated or whole_video_corroborated):
            continue
        bound = rows[0]['source_identity']
        if not all(r['source_identity'] == bound for r in rows):
            continue
        proofs[cid] = {**bound, 'basis': ('corroborated_recording_only' if locally_corroborated
                                          else 'whole_video_recording_only'),
                       'confidence': min(_recording_confidence(r) for r in rows), 'window_count': len(rows)}
    return proofs


def proof_for_clip(clip, proofs):
    proof = proofs.get(clip.clip_id)
    if not isinstance(proof, dict) or proof.get('basis') not in {
        'corroborated_recording_only', 'whole_video_recording_only'
    }:
        return None
    return proof if all(proof.get(k) == v for k, v in identity(clip).items()) else None

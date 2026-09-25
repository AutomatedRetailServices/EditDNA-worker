"""Source-bound, contiguous mixed-speech boundary trimming at Clean Cut.

No new provider calls. Never stitch interior speech or inherit a failed label
onto the audience remainder. Missing alignment, disagreement or missing local
corroboration preserves the original candidate.
"""
from dataclasses import replace
import math

from .clean_cut_provider import _candidate_from_words
from .recording_process_evidence import identity


def apply_recording_process_trims(takes, windows, context):
    from .hybrid_session_cleanup import _failed_local_evidence

    rows_by_id = {}
    for window in windows:
        for row in window.get('decisions') or ():
            rows_by_id.setdefault(row['clip_id'], []).append(row)
    kept, discarded, proof_rows, diagnostics = [], [], [], []
    for take in takes:
        rows = rows_by_id.get(take.clip_id, [])
        words = take.words
        counts = {(r.get('recording_prefix_words', 0), r.get('recording_suffix_words', 0))
                  for r in rows}
        if not rows or len(counts) != 1:
            kept.append(take)
            continue
        prefix, suffix = next(iter(counts))
        valid = all(
            r.get('content_role') == 'mixed'
            and r.get('source_identity') == identity(take)
            and type(r.get('recording_confidence')) in (int, float)
            and math.isfinite(r['recording_confidence'])
            and .97 <= r['recording_confidence'] <= 1
            for r in rows
        )
        valid = valid and all(type(n) is int and n >= 0 for n in (prefix, suffix))
        valid = valid and bool(words) and 0 < prefix + suffix < len(words)
        # Only word-aligned text actually visible to the provider may be trimmed.
        valid = valid and ' '.join(' '.join(w.text for w in words).split()) == ' '.join(take.text.split())
        previous_end = take.start
        for word in words:
            valid = valid and all(math.isfinite(t) for t in (word.start, word.end))
            valid = valid and take.start <= word.start < word.end <= take.end and word.start >= previous_end
            previous_end = word.end
        if not valid:
            kept.append(take)
            continue
        end = len(words) - suffix
        audience = _candidate_from_words(take, words[prefix:end])
        if len(audience.words) < 3 or audience.duration_sec < .5:
            kept.append(take)
            continue
        removed = []
        if prefix:
            removed.append(_candidate_from_words(take, words[:prefix]))
        if suffix:
            removed.append(_candidate_from_words(take, words[end:]))
        # Parent-wide visual scores do not corroborate a specific word boundary.
        evidence = [_failed_local_evidence(replace(part, signals=None), context) for part in removed]
        if not all(corroborated for corroborated, reasons in evidence):
            kept.append(take)
            continue
        kept.append(audience)
        discarded.extend(removed)
        confidence = min(r['recording_confidence'] for r in rows)
        for part, (_, reasons) in zip(removed, evidence):
            proof_rows.append(dict(
                clip_id=part.clip_id, source_identity=identity(part),
                label='bts', content_role='recording_only', confidence=confidence,
                recording_confidence=confidence, local_failure_corroborated=True,
                parent_clip_id=take.clip_id, parent_source_identity=identity(take),
                recording_prefix_words=prefix, recording_suffix_words=suffix,
                local_failure_reasons=list(reasons),
            ))
        diagnostics.append(dict(
            clip_id=take.clip_id, action='mixed', confidence=confidence,
            reason='corroborated_recording_boundaries', applied_delete=False, applied_mixed_trim=True,
            keep_start_word_index=prefix, keep_end_word_index=end - 1,
            kept_clip_id=audience.clip_id, discarded_clip_ids=[p.clip_id for p in removed],
            basis='source_bound_recording_boundaries', recording_confidence=confidence,
        ))
    return tuple(kept), tuple(discarded), tuple(proof_rows), tuple(diagnostics)

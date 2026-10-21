"""Source-bound removal of recording edges or inter-sentence preparation.

Keep contiguous source pieces as separate candidates for ordinary grouping.
No cross-take stitching, no AV self-corroboration, no inherited failed label.
"""
from dataclasses import replace
import math
from .clean_cut_provider import _candidate_from_words
from .recording_process_evidence import identity
from .av_candidate_evidence import candidate_observations


def _ranges(row, size):
    prefix,suffix=row.get('recording_prefix_words',0),row.get('recording_suffix_words',0)
    if any(type(n) is not int or n<0 for n in (prefix,suffix)):
        raise ValueError('invalid_word_boundaries')
    proposed=row.get('recording_word_ranges') or ()
    if proposed and (prefix or suffix):
        raise ValueError('ambiguous_boundary_contract')
    ranges=list(proposed) if proposed else ([(0,prefix-1)] if prefix else []) + ([(size-suffix,size-1)] if suffix else [])
    if len(ranges)>4:
        raise ValueError('too_many_recording_ranges')
    result=[];last=-1
    for pair in ranges:
        if not isinstance(pair,(list,tuple)) or len(pair)!=2 or any(type(n) is not int for n in pair):
            raise ValueError('invalid_word_boundaries')
        a,b=pair
        if not 0<=a<=b<size or a<=last:
            raise ValueError('invalid_word_boundaries')
        result.append((a,b));last=b
    return tuple(result)


def apply_recording_process_trims(takes, windows, context):
    from .hybrid_session_cleanup import _failed_local_evidence
    rows_by_id={}
    for window in windows:
        for row in window.get('decisions') or ():
            rows_by_id.setdefault(row['clip_id'],[]).append(row)
    kept,discarded,proof_rows,diagnostics=[],[],[],[]
    for take in takes:
        rows=rows_by_id.get(take.clip_id,[]); words=take.words
        trace=dict(clip_id=take.clip_id,action='keep',applied_delete=False,applied_mixed_trim=False,
                   source_identity=identity(take),audiovisual=candidate_observations(take,context))
        def preserve(reason):
            kept.append(take)
            diagnostics.append({**trace,'reason':reason})
        if not rows:
            preserve('no_editorial_decision');continue
        try:
            proposals=[_ranges(r,len(words)) for r in rows]
        except (ValueError,TypeError) as exc:
            preserve(str(exc));continue
        if any(p!=proposals[0] for p in proposals):
            preserve('conflicting_word_boundaries');continue
        ranges=proposals[0]
        if not ranges:
            preserve('no_recording_word_boundaries');continue
        if not all(r.get('content_role')=='mixed' and r.get('source_identity')==identity(take) for r in rows):
            preserve('role_or_source_identity_not_confirmed');continue
        if not all(type(r.get('recording_confidence')) in (int,float)
                   and math.isfinite(r['recording_confidence']) and .97<=r['recording_confidence']<=1 for r in rows):
            preserve('insufficient_recording_confidence');continue
        valid=bool(words) and ' '.join(' '.join(w.text for w in words).split())==' '.join(take.text.split())
        previous_end=take.start
        for word in words:
            valid=valid and all(math.isfinite(t) for t in (word.start,word.end))
            valid=valid and take.start<=word.start<word.end<=take.end and word.start>=previous_end
            previous_end=word.end
        if not valid:
            preserve('alignment_not_source_bound');continue
        # Interior excision must separate complete sentences; a hesitation or
        # self-correction inside a sentence is not eligible for this authority.
        if any(a>0 and b<len(words)-1 and (
            not words[a-1].text.rstrip().endswith(('.', '?', '!'))
            or not words[b].text.rstrip().endswith(('.', '?', '!'))
        ) for a,b in ranges):
            preserve('interior_sentence_boundary_unproven');continue
        retained=[];cursor=0
        for a,b in ranges:
            if a>cursor:retained.append((cursor,a-1))
            cursor=b+1
        if cursor<len(words):retained.append((cursor,len(words)-1))
        if not retained or any(b-a+1<3 or words[b].end-words[a].start<.5 for a,b in retained):
            preserve('audience_remainder_too_short');continue
        audience=[_candidate_from_words(take,words[a:b+1]) for a,b in retained]
        removed=[_candidate_from_words(take,words[a:b+1]) for a,b in ranges]
        evidence=[_failed_local_evidence(replace(part,signals=None),context) for part in removed]
        if not all(ok for ok,_ in evidence):
            preserve('missing_independent_local_corroboration');continue
        kept.extend(audience);discarded.extend(removed)
        confidence=min(r['recording_confidence'] for r in rows)
        for part,word_range,(_,reasons) in zip(removed,ranges,evidence):
            proof_rows.append(dict(clip_id=part.clip_id,source_identity=identity(part),
                label='bts',content_role='recording_only',confidence=confidence,
                recording_confidence=confidence,local_failure_corroborated=True,
                parent_clip_id=take.clip_id,parent_source_identity=identity(take),
                recording_word_range=list(word_range),local_failure_reasons=list(reasons)))
        diagnostics.append({**trace,'action':'mixed','confidence':confidence,
            'reason':'corroborated_recording_word_ranges','applied_mixed_trim':True,
            'keep_start_word_index':retained[0][0],'keep_end_word_index':retained[-1][1],
            'kept_clip_id':audience[0].clip_id,'kept_clip_ids':[p.clip_id for p in audience],
            'discarded_clip_ids':[p.clip_id for p in removed],
            'recording_word_ranges':[list(r) for r in ranges],
            'basis':'source_bound_recording_boundaries','recording_confidence':confidence})
    return tuple(kept),tuple(discarded),tuple(proof_rows),tuple(diagnostics)

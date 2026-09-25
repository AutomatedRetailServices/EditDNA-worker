"""Advisory source/time/word links; model observations are never local corroboration."""
import json
import math


def candidate_observations(take, context):
    if context is None:
        return {'status':'unavailable','observations':[],'omitted_count':0}
    source=next((s for s in context.sources if s.source_asset_id==take.source_asset_id),None)
    if source is None or not source.audiovisual_evidence:
        return {'status':'unavailable','observations':[],'omitted_count':0}
    try:
        data=json.loads(source.audiovisual_evidence)
        rows=[]
        for index,r in enumerate(data['regions']):
            start,end,confidence=(r[k] for k in ('start','end','confidence'))
            if not all(type(v) in (int,float) and math.isfinite(v) for v in (start,end,confidence)):
                continue
            if not 0<=start<end or not 0<=confidence<=1 or end<=take.start or start>=take.end:
                continue
            words=[i for i,w in enumerate(take.words) if w.start>=start and w.end<=end]
            rows.append({'observation_id':r.get('observation_id',f'av_{index}'),
                         'source_start':start,'source_end':end,'role':r.get('role','uncertain'),
                         'confidence':confidence,'word_range':[words[0],words[-1]] if words else None,
                         'audio':str(r.get('audio_observation',''))[:64],
                         'visual':str(r.get('visual_observation',''))[:64]})
        return {'status':'overlap' if rows else 'no_observation_for_span',
                'source_sha256':data.get('source_sha256'),
                'observations':rows[:3],'omitted_count':max(0,len(rows)-3),
                'authority':'advisory_not_independent_local_evidence'}
    except (ValueError,TypeError,KeyError):
        return {'status':'invalid','observations':[],'omitted_count':0}

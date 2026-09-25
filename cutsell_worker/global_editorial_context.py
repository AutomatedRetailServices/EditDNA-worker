"""Feed source-grounded global hypotheses to the existing editorial decision stage."""
from dataclasses import replace
import json


def with_global_editorial_evidence(context, result):
    if result is None or result.understanding is None:
        return context, {'status':'not_evaluable', 'reason':'no_global_understanding', 'sources':[]}
    if context is None:
        return context, {'status':'not_evaluable', 'reason':'no_source_context', 'sources':[]}
    understanding = result.understanding
    sources, rows = [], []
    for source in context.sources:
        regions = [r for r in understanding.regions if r.source_asset_id == source.source_asset_id]
        # Bounded source-scoped evidence. It is explicitly advisory, never an
        # event/keep/delete flag or an instruction to override valid speech.
        evidence = [{'start':r.source_start, 'end':r.source_end,
                     'process':r.dominant_process_status, 'audience':r.audience_delivery_status,
                     'confidence':r.confidence, 'conflicts':list(r.conflict_flags)} for r in regions[:12]]
        if evidence:
            payload = json.dumps({'kind':'global_editorial_hypotheses',
                                  'rule':'Corroborate with candidate content; never delete from this summary alone.',
                                  'regions':evidence}, separators=(',',':'))
            sources.append(replace(source, editorial_evidence=payload))
        else:
            sources.append(source)
        rows.append({'source_asset_id':source.source_asset_id, 'region_count':len(regions),
                     'provided_region_count':len(evidence), 'omitted_region_count':max(0,len(regions)-12)})
    return replace(context, sources=tuple(sources)), {
        'status':'prepared_for_editorial_classifier' if any(r['provided_region_count'] for r in rows) else 'no_regions',
        'stage':'before_composite_resolution', 'sources':rows,
        'capability_status':result.capability_status,
    }

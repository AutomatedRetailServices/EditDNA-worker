"""Offline review of saved benchmark evidence; never calls providers or renders.

Only replays candidates with exact source/text identity and retained completeness
metadata. Missing evidence is reported, never inferred from the final selection.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cutsell_worker.contracts import CandidateTake
from cutsell_worker.retry_replacement_coverage import review_retry_pool


def audit(data):
    windows=data['diagnostics']['hybrid_editorial_chunks']
    records={r['clip_id']:r for w in windows for r in w.get('decisions',())}
    attempts={r['clip_id']:r for r in data['diagnostics']['attempt_reconstruction']['attempts']}
    saved={r['clip_id']:r for bucket in ('selected','discarded','alternates') for r in data[bucket]}
    words=[(s['source_asset_id'],w) for s in data['timed_asr_replay_evidence']['raw_segments'] for w in s.get('words',())]
    takes=[]; unavailable=[]
    for cid,row in records.items():
        identity=row.get('source_identity') or {}
        if cid not in attempts or not identity:
            unavailable.append({'clip_id':cid,'reason':'missing_original_identity_or_completeness'});continue
        candidates=[saved.get(cid,{}).get('text','')]
        candidates.append(' '.join(w['text'] for source,w in words if source==identity['source_asset_id']
            and w['start']>=identity['start']-.001 and w['end']<=identity['end']+.001))
        text=next((t for t in candidates if hashlib.sha256(t.encode()).hexdigest()==identity['text_sha256']),None)
        if text is None:
            unavailable.append({'clip_id':cid,'reason':'original_text_hash_not_recovered'});continue
        takes.append(CandidateTake(cid,identity['source_asset_id'],0,identity['start'],identity['end'],text,
                                   complete_idea=attempts[cid]['complete_idea']))
    reviews=review_retry_pool(takes,windows)
    texts={t.clip_id:t.text for t in takes}
    for review in reviews:
        review['candidate_text']=texts[review['candidate_clip_id']]
        for pair in review['comparisons']:
            pair['replacement_text']=texts[pair['proposed_replacement_id']]
    return {'schema_version':'cutsell.retry_coverage_audit.v1','source_sha256':data.get('source_media_sha256'),
            'classified_count':len(records),'recovered_count':len(takes),'unavailable':unavailable,
            'reviews':reviews,'rendered':False,'counterfactual_selection':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('result');parser.add_argument('output')
    args=parser.parse_args()
    Path(args.output).write_text(json.dumps(audit(json.loads(Path(args.result).read_text())),indent=2,ensure_ascii=False))

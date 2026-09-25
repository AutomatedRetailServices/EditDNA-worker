import os,json
from pathlib import Path
import requests,boto3
r=requests.get('https://rest.runpod.io/v1/templates',headers={'Authorization':'Bearer '+os.environ['RUNPOD_API_KEY']},timeout=60);r.raise_for_status()
env=next(x['env'] for x in r.json() if x['name']=='EditDNA-Worker-2')
c=boto3.client('s3',aws_access_key_id=env['AWS_ACCESS_KEY_ID'],aws_secret_access_key=env['AWS_SECRET_ACCESS_KEY'],region_name=env.get('AWS_REGION','us-east-1'))
p=Path('december-recovery');p.mkdir()
for name in ['result.json','summary.json']:
 c.download_file(env['S3_BUCKET'],'cutsell/legacy-comparison/36079579008/'+name,str(p/name))
r=json.loads((p/'result.json').read_text());print(json.dumps({'clip_count':len(r['clips']),'composer':r['composer'],'llm_used':r['llm_used']}))

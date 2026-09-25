"""CPU orchestration of exactly one authorized historical comparison."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from benchmarks.run_video00_gpt_whisperx_stability import s3_client

def main():
    assert os.environ['GITHUB_RUN_ATTEMPT']=='1','Paid reruns forbidden'
    out=ROOT/'august-artifacts';out.mkdir(exist_ok=True)
    r=requests.get('https://rest.runpod.io/v1/templates',headers={'Authorization':'Bearer '+os.environ['RUNPOD_API_KEY']},timeout=60);r.raise_for_status()
    matches=[t for t in r.json() if t['name']=='EditDNA-Worker-2'];assert len(matches)==1
    original={str(k):str(v) for k,v in matches[0]['env'].items()}
    assert original.get('OPENAI_API_KEY') and not original['OPENAI_API_KEY'].startswith('sk-admin-')
    # Credentials only from current template. Historical behavior uses explicit
    # reconstructed flags and code defaults, never modern editorial overlays.
    names=['OPENAI_API_KEY','AWS_ACCESS_KEY_ID','AWS_SECRET_ACCESS_KEY','AWS_SESSION_TOKEN','AWS_REGION','AWS_DEFAULT_REGION','S3_BUCKET']
    env={k:original[k] for k in names if k in original}
    flags={'WHISPER_MODEL_NAME':'medium','WHISPER_DEVICE':'cuda','ASR_ENABLED':'1','EDITDNA_USE_LLM':'1','EDITDNA_LLM_MODEL':'gpt-5.1','VISION_ENABLED':'1','BAD_TAKES_ENABLED':'1','TAKE_JUDGE_ENABLED':'1','TAKE_JUDGE_MODEL':'gpt-4o-mini','BOUNDARY_REFINER_ENABLED':'0','OPENAI_MAX_RETRIES':'0','OPENAI_TIMEOUT_SECONDS':'120'}
    env.update(flags,S3_PREFIX='cutsell/august-comparison/'+os.environ['GITHUB_RUN_ID'],LEGACY_SOURCE_KEY='cutsell/benchmark-inputs/upload-36065470982/5ae3cffb9034cc9aebbe4f645c4f1f34538f9113f05990291215c86b7163b681.mp4',UPLOAD_EXPECTED_SHA='5ae3cffb9034cc9aebbe4f645c4f1f34538f9113f05990291215c86b7163b681')
    for k,v in env.items():
        if k in names:print('::add-mask::'+v)
    private=Path(os.environ['RUNNER_TEMP'])/'august-env.json';private.write_text(json.dumps(env));private.chmod(0o600)
    (out/'configuration.json').write_text(json.dumps({'reconstructed':True,'flags':flags,'historical_commit':'c8aa989','boundary_refiner':'historical default off','source_sha256':env['UPLOAD_EXPECTED_SHA'],'modern_reference_run':36076863910},indent=2))
    child=dict(os.environ,CUTSELL_ENV_JSON_PATH=str(private))
    try:
        with (out/'modal.log').open('w') as log:
            call=subprocess.run(['modal','run','benchmarks/modal_editdna_august.py'],env=child,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
        assert call.returncode==0,'Modal did not return; never retry automatically'
        client=s3_client(env)
        client.download_file(env['S3_BUCKET'],env['S3_PREFIX']+'/summary.json',str(out/'summary.json'))
        s=json.loads((out/'summary.json').read_text())
        client.download_file(env['S3_BUCKET'],env['S3_PREFIX']+'/stages.json',str(out/'stages.json'))
        if s.get('ok'):
            client.download_file(env['S3_BUCKET'],env['S3_PREFIX']+'/result.json',str(out/'result.json'))
            client.download_file(env['S3_BUCKET'],env['S3_PREFIX']+'/edited.mp4',str(out/'EditDNA_Agosto_RECONSTRUIDO.mp4'))
            video=out/'EditDNA_Agosto_RECONSTRUIDO.mp4'
            data=video.read_bytes();assert hashlib.sha256(data).hexdigest()==s['sha256']
            parts=[]
            for i,start in enumerate(range(0,len(data),20*1024*1024)):
                assert i<12
                folder=out/chr(65+i);folder.mkdir()
                name=f'video.part{i}';block=data[start:start+20*1024*1024];(folder/name).write_bytes(block)
                parts.append({'filename':name,'sha256':hashlib.sha256(block).hexdigest()})
            (out/'parts.json').write_text(json.dumps({'sha256':s['sha256'],'parts':parts}))
        print(json.dumps(s))
    finally:private.unlink(missing_ok=True)

if __name__=='__main__':main()

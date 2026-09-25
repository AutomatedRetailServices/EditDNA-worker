"""One isolated historical pipeline invocation; source code is byte-identical.

Transport adaptations: verified local S3 input, unique output prefix, bounded
OpenAI transport with zero retries. The opt-in repaired run substitutes only the approved phrase-merge repair.
"""
import json
import os
from pathlib import Path
import modal

env=json.loads(Path(os.environ['CUTSELL_ENV_JSON_PATH']).read_text()) if os.environ.get('CUTSELL_ENV_JSON_PATH') else {}
app=modal.App('cutsell-authorized-august-comparison')
image=(modal.Image.from_registry('madiator2011/better-pytorch:cuda12.4-torch2.6.0')
       .apt_install('ffmpeg','git','build-essential','python3-dev','pkg-config','libavformat-dev','libavcodec-dev','libavdevice-dev','libavutil-dev','libavfilter-dev','libswscale-dev','libswresample-dev')
       .pip_install('faster-whisper==1.0.0','boto3','requests','openai','openai-clip','pillow','torchvision==0.21.0')
       .add_local_dir('benchmarks/editdna_august_repaired_runtime' if os.environ.get('AUGUST_PHRASE_REPAIR')=='1' else 'benchmarks/editdna_august','/opt/august',copy=True)
       .run_commands("PYTHONPATH=/opt/august python -c \"from worker import pipeline\""))

@app.function(image=image,gpu='L4',timeout=1800,retries=0,secrets=[modal.Secret.from_dict(env)])
def run_legacy():
    import hashlib, logging, subprocess, time, sys, copy
    import boto3, openai
    from functools import partial
    from botocore.exceptions import ClientError
    began=time.monotonic()
    client=boto3.client('s3');bucket=os.environ['S3_BUCKET'];prefix=os.environ['S3_PREFIX']
    summary_key=prefix+'/summary.json'
    try:
        client.put_object(Bucket=bucket,Key=prefix+'/claimed',Body=b'no retry',IfNoneMatch='*')
    except ClientError as e:
        raise RuntimeError('Historical invocation already claimed; no retry') from e
    summary={'historical_commit':'c8aa989','configuration_reconstructed':True,'source_sha256':os.environ['UPLOAD_EXPECTED_SHA']}
    summary['phrase_repair']=os.environ.get('AUGUST_PHRASE_REPAIR')=='1'
    try:
        source='/tmp/verified-source.mov'
        client.download_file(bucket,os.environ['LEGACY_SOURCE_KEY'],source)
        assert hashlib.sha256(Path(source).read_bytes()).hexdigest()==os.environ['UPLOAD_EXPECTED_SHA']
        sys.path.insert(0,'/opt/august')
        from worker import pipeline as m
        assert hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()==os.environ.get('AUGUST_PIPELINE_SHA256','6285b8dae54c94691cc5be8fe0b5f61ff60c6373f6e330f062aa817c6e59fec1')
        traces={}
        def persist_traces():
            client.put_object(Bucket=bucket,Key=prefix+'/stages.json',Body=json.dumps(traces,default=str).encode(),ContentType='application/json')
        def observe(name):
            original=getattr(m,name)
            def wrapped(*args,**kwargs):
                item={}
                if args and isinstance(args[0],list):item['input']=copy.deepcopy(args[0])
                traces[name]=item
                persist_traces()
                try:
                    result=original(*args,**kwargs)
                    item['output']=copy.deepcopy(result)
                    if args and isinstance(args[0],list):item['input_after']=copy.deepcopy(args[0])
                    persist_traces()
                    return result
                except Exception as e:
                    item['error_type']=type(e).__name__;persist_traces();raise
            setattr(m,name,wrapped)
        for name in ['run_asr','sentence_boundary_micro_cuts','merge_incomplete_phrases','enrich_clips_semantic','run_visual_pass','run_take_judge','build_composer']:
            observe(name)
        logging.getLogger('editdna.pipeline').setLevel(logging.WARNING)
        r=m.run_pipeline(session_id='historical-test',local_files=[source],mode='human',use_semantic_v2=True,use_take_judge_v2=True)
        used=r['composer']['used_clip_ids']
        # Historical dataset fallback returns original input; never label it edited.
        summary.update(ok=bool(used),selected_count=len(used),llm_used=r['llm_used'],vision=r['vision'],take_judge_used=r['take_judge_used'],bad_takes_used=r['bad_takes_used'],boundaries_refined=r['boundaries_refined'])
        client.put_object(Bucket=bucket,Key=prefix+'/result.json',Body=json.dumps(r).encode(),ContentType='application/json')
        if used:
            video='/tmp/august-edited.mp4';client.download_file(bucket,prefix+'/historical-test-final.mp4',video);probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_format','-of','json',video]))
            decode=subprocess.run(['ffmpeg','-v','error','-i',video,'-f','null','-'],capture_output=True,text=True)
            assert decode.returncode==0 and not decode.stderr
            summary.update(output_duration_sec=float(probe['format']['duration']),sha256=hashlib.sha256(Path(video).read_bytes()).hexdigest(),decode_ok=True)
            client.upload_file(video,bucket,prefix+'/edited.mp4')
        else:summary['error']='No selected clips; historical original-input fallback is not an edited video'
    except Exception as e:
        summary.update(ok=False,error_type=type(e).__name__,error=str(e)[:1500])
    summary['elapsed_sec']=round(time.monotonic()-began,3)
    client.put_object(Bucket=bucket,Key=summary_key,Body=json.dumps(summary).encode(),ContentType='application/json')
    return summary

@app.local_entrypoint()
def main():
    result=run_legacy.remote()
    Path('august-compact.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))

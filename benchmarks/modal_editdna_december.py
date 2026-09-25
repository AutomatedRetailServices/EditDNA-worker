"""One isolated historical pipeline invocation; source code is byte-identical.

Transport adaptations: verified local S3 input, unique output prefix, bounded
OpenAI transport with zero retries. No editorial function is replaced.
"""
import json
import os
from pathlib import Path
import modal

env=json.loads(Path(os.environ['CUTSELL_ENV_JSON_PATH']).read_text()) if os.environ.get('CUTSELL_ENV_JSON_PATH') else {}
app=modal.App('cutsell-authorized-december-comparison')
image=(modal.Image.from_registry('madiator2011/better-pytorch:cuda12.4-torch2.6.0')
       .apt_install('ffmpeg','git','build-essential','python3-dev','pkg-config','libavformat-dev','libavcodec-dev','libavdevice-dev','libavutil-dev','libavfilter-dev','libswscale-dev','libswresample-dev')
       .pip_install('faster-whisper==1.0.0','boto3','requests','openai','openai-clip')
       .add_local_file('benchmarks/editdna_december_pipeline.py','/opt/december_pipeline.py',copy=True))

@app.function(image=image,gpu='L4',timeout=1800,retries=0,secrets=[modal.Secret.from_dict(env)])
def run_legacy():
    import hashlib, importlib.util, logging, shutil, subprocess, time
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
    summary={'historical_commit':'9ffbdc1','configuration_reconstructed':True,'source_sha256':os.environ['UPLOAD_EXPECTED_SHA']}
    try:
        source='/tmp/verified-source.mov'
        client.download_file(bucket,os.environ['LEGACY_SOURCE_KEY'],source)
        assert hashlib.sha256(Path(source).read_bytes()).hexdigest()==os.environ['UPLOAD_EXPECTED_SHA']
        assert hashlib.sha256(Path('/opt/december_pipeline.py').read_bytes()).hexdigest()=='5d7ee91ddad80e4fcc1bb5a06cda22a8ce07658ff848e7ad0e3efb5db3c5736d'
        # Transport bound only; historical prompts, models and parsing unchanged.
        openai.OpenAI=partial(openai.OpenAI,max_retries=0,timeout=120)
        spec=importlib.util.spec_from_file_location('december_pipeline','/opt/december_pipeline.py')
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        m.download_to_local=lambda url,dest:shutil.copyfile(source,dest)
        logging.getLogger('editdna.pipeline').setLevel(logging.WARNING)
        r=m.run_pipeline(session_id='historical-test',files=['verified-source'],mode='human')
        used=r['composer']['used_clip_ids']
        # Historical dataset fallback returns original input; never label it edited.
        summary.update(ok=bool(used),selected_count=len(used),llm_used=r['llm_used'],vision=r['vision'],take_judge_used=r['take_judge_used'],bad_takes_used=r['bad_takes_used'],boundaries_refined=r['boundaries_refined'])
        client.put_object(Bucket=bucket,Key=prefix+'/result.json',Body=json.dumps(r).encode(),ContentType='application/json')
        if used:
            video=r['output_video_local'];probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_format','-of','json',video]))
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
    Path('december-compact.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))

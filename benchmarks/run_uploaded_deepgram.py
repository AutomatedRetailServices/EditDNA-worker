"""One authorized CPU-only Deepgram transcription; no edit authority or retries."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from benchmarks.run_video00_gpt_whisperx_stability import s3_client

SOURCE_KEY = 'cutsell/benchmark-inputs/upload-36065470982/5ae3cffb9034cc9aebbe4f645c4f1f34538f9113f05990291215c86b7163b681.mp4'
SOURCE_SHA = '5ae3cffb9034cc9aebbe4f645c4f1f34538f9113f05990291215c86b7163b681'
OUT = ROOT / 'deepgram-uploaded-artifacts'


def main():
    OUT.mkdir(exist_ok=True)
    if os.environ.get('GITHUB_RUN_ATTEMPT') != '1':
        raise RuntimeError('No paid reruns')
    r = requests.get('https://rest.runpod.io/v1/templates', headers={'Authorization': 'Bearer ' + os.environ['RUNPOD_API_KEY']}, timeout=(15,60))
    r.raise_for_status()
    matches = [t for t in r.json() if t.get('name') == 'EditDNA-Worker-2']
    if len(matches) != 1:
        raise RuntimeError('Canonical template absent or ambiguous')
    env = matches[0]['env']
    key = str(env.get('DEEPGRAM_API_KEY') or '').strip()
    preflight = {'deepgram_key_present': bool(key), 'source_sha256': SOURCE_SHA, 'authorized_requests': 1, 'edit_authority': False}
    (OUT/'preflight.json').write_text(json.dumps(preflight,indent=2))
    if not key:
        raise RuntimeError('DEEPGRAM_API_KEY missing from EditDNA-Worker-2; no provider call made')
    print('Deepgram credential present; value remains private')
    with tempfile.TemporaryDirectory(prefix='deepgram-uploaded-') as tmp:
        source, audio = Path(tmp)/'source.mov', Path(tmp)/'audio.mp3'
        s3_client(env).download_file(env['S3_BUCKET'], SOURCE_KEY, str(source))
        if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA:
            raise RuntimeError('Source SHA mismatch; no provider call made')
        subprocess.run(['ffmpeg','-v','error','-y','-i',str(source),'-vn','-ac','1','-ar','16000','-b:a','64k',str(audio)],check=True,capture_output=True,timeout=120)
        with (OUT/'request.claimed').open('x') as f:
            f.write('One authorized request; never automatically retry')
        began=time.monotonic()
        with audio.open('rb') as f:
            response=requests.post('https://api.deepgram.com/v1/listen',params={'model':'nova-3','language':'multi'},headers={'Authorization':'Token '+key,'Content-Type':'audio/mpeg'},data=f,timeout=(20,300))
        elapsed=round(time.monotonic()-began,3)
        try:
            payload=response.json()
        except ValueError:
            payload={'error':'Non-JSON response'}
        (OUT/'response.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2))
        if response.status_code != 200:
            raise RuntimeError(f'Deepgram HTTP {response.status_code}; see response artifact')
        alt=payload['results']['channels'][0]['alternatives'][0]
        words=alt.get('words',[])
        summary={'provider':'deepgram','model':'nova-3','language':'multi','source_sha256':SOURCE_SHA,'elapsed_sec':elapsed,'word_count':len(words),'words_before_60':sum(w['start']<60 for w in words),'transcript':alt.get('transcript'),'first_word_start':words[0]['start'] if words else None,'last_word_end':words[-1]['end'] if words else None,'edit_authority':False,'request_id':payload.get('metadata',{}).get('request_id')}
        (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
        print(json.dumps({k:v for k,v in summary.items() if k!='transcript'}))


if __name__ == '__main__':
    main()

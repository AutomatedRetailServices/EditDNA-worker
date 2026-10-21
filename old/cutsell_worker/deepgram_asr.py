"""Opt-in Deepgram word evidence for the existing RAW editor; no fallback."""
from dataclasses import dataclass, field
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time

import requests
from .contracts import TranscriptSegment, Word

PROVIDER = 'deepgram-nova-3-multi'


@dataclass(frozen=True)
class DeepgramFingerprint:
    language_hint: str | None = None

    def fingerprint(self):
        spec = {'provider': PROVIDER, 'audio': 'mp3-mono-16k-64k',
                'punctuate': True, 'segmentation': 'punctuation-gap650ms-v1',
                'language': self.language_hint or 'multi', 'cache': 'per-job-source-sha256'}
        return 'asrcfg_' + hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()[:16]


def checked_segments(payload, source_asset_id, duration):
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('Invalid source duration')
    raw = payload['results']['channels'][0]['alternatives'][0]
    words = []
    previous_start = 0.0
    for item in raw.get('words', []):
        start, end, confidence = float(item['start']), float(item['end']), float(item['confidence'])
        text = str(item.get('punctuated_word') or item['word']).strip()
        if not text or len(text.split()) != 1 or not all(map(math.isfinite,(start,end,confidence))):
            raise ValueError('Invalid Deepgram word evidence')
        if not 0 <= start < end <= duration + 0.1 or start < previous_start - 0.002 or not 0 <= confidence <= 1:
            raise ValueError('Invalid Deepgram timing or confidence')
        words.append(Word(text,start,end,confidence))
        previous_start = start
    if not words:
        raise ValueError('Deepgram returned no timed words')
    # Check all provider tokens are represented; punctuation may differ.
    import re
    normalize = lambda text: re.findall(r"\w+",text.casefold())
    if normalize(raw.get('transcript','')) != normalize(' '.join(w.text for w in words)):
        raise ValueError('Deepgram transcript/word coverage mismatch')
    segments, group = [], []
    for w in words:
        if group and (w.start-group[-1].end >= 0.65 or group[-1].text.endswith(('.', '?', '!'))):
            segments.append(TranscriptSegment(source_asset_id,group[0].start,max(x.end for x in group),' '.join(x.text for x in group),tuple(group)))
            group=[]
        group.append(w)
    if group:
        segments.append(TranscriptSegment(source_asset_id,group[0].start,max(x.end for x in group),' '.join(x.text for x in group),tuple(group)))
    return tuple(segments)


@dataclass
class DeepgramASR:
    model_name: str = PROVIDER
    last_audit: dict = field(default_factory=dict, init=False)
    _source_cache: dict = field(default_factory=dict, init=False, repr=False)

    def config_fingerprint(self, *, language_hint=None):
        return DeepgramFingerprint(language_hint)

    def transcribe(self, path, *, source_asset_id, language_hint=None):
        if language_hint not in (None,'en','es'):
            raise ValueError('Unsupported Deepgram language hint')
        key=os.environ.get('DEEPGRAM_API_KEY','').strip()
        if not key:
            raise RuntimeError('DEEPGRAM_API_KEY missing')
        digest=hashlib.sha256()
        with open(path,'rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
        sha=digest.hexdigest()
        fingerprint=self.config_fingerprint(language_hint=language_hint).fingerprint()
        cache_key=(sha,source_asset_id,fingerprint)
        if cache_key in self._source_cache:
            result,audit=self._source_cache[cache_key]
            audit['cache_hit_count']+=1
            self.last_audit=copy.deepcopy(audit)
            return result
        began=time.monotonic()
        duration=float(json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','json',path],text=True,timeout=60))['format']['duration'])
        self.last_audit={'provider':PROVIDER,'model':'nova-3','status':'running','source_media_sha256':sha,
                         'config_fingerprint':fingerprint,'cache_hit_count':0,'fallback':None,
                         'confidence_semantics':'Deepgram word recognition confidence; not calibrated against Whisper',
                         'timestamp_interpolation':False,'requested_language':language_hint}
        try:
            with tempfile.TemporaryDirectory(prefix='cutsell-deepgram-') as tmp:
                audio=Path(tmp)/'audio.mp3'
                subprocess.run(['ffmpeg','-v','error','-y','-i',path,'-vn','-ac','1','-ar','16000','-b:a','64k',str(audio)],check=True,capture_output=True,timeout=120)
                with audio.open('rb') as f:
                    response=requests.post('https://api.deepgram.com/v1/listen',params={'model':'nova-3','language':language_hint or 'multi','punctuate':'true'},headers={'Authorization':'Token '+key,'Content-Type':'audio/mpeg'},data=f,timeout=(20,300))
                if response.status_code != 200:
                    raise RuntimeError(f'Deepgram HTTP {response.status_code}')
                payload=response.json()
                result=checked_segments(payload,source_asset_id,duration)
                self.last_audit.update(status='passed',elapsed_sec=round(time.monotonic()-began,3),
                    request_id=payload.get('metadata',{}).get('request_id'),model_info=payload.get('metadata',{}).get('model_info'),
                    duration_sec=duration,word_count=sum(len(s.words) for s in result),segment_count=len(result),
                    overlapping_word_pairs=sum(b.start < a.end-0.002 for a,b in zip([w for s in result for w in s.words],[w for s in result for w in s.words][1:])))
                self._source_cache[cache_key]=(result,copy.deepcopy(self.last_audit))
                return result
        except Exception as exc:
            self.last_audit.update(status='failed',error_type=type(exc).__name__)
            print('DEEPGRAM_ASR_AUDIT='+json.dumps(self.last_audit),flush=True)
            raise

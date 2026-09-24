"""CPU-only artifact packaging; never starts an inference call."""
from pathlib import Path
import hashlib
import json
import shutil
import sys


def package(root, index):
    if index not in range(1,6):
        raise ValueError('Only five authorized trials')
    source=root/'stability-artifacts'/f'trial-{index}'
    dest=root/'stability-transfer'/f'trial-{index}'
    reports=dest/'reports'
    reports.mkdir(parents=True,exist_ok=True)
    for f in source.iterdir():
        if f.suffix in ('.json','.log'):shutil.copyfile(f,reports/f.name)
    videos=list(source.glob('*.mp4'))
    if len(videos)>1:raise ValueError('Unexpected multiple videos')
    for video in videos:
        parts=[];digest=hashlib.sha256()
        with video.open('rb') as f:
            for i,letter in enumerate('ABCDEFGHIJKL'):
                block=f.read(20*1024*1024)
                if not block:break
                d=dest/letter;d.mkdir(exist_ok=True)
                name=video.name+f'.part{i}';(d/name).write_bytes(block);digest.update(block)
                parts.append({'filename':name,'sha256':hashlib.sha256(block).hexdigest()})
            if f.read(1):raise ValueError('Video exceeds 240MiB transport bound')
        (reports/'parts.json').write_text(json.dumps({'filename':video.name,'sha256':digest.hexdigest(),'parts':parts},indent=2))


if __name__=='__main__':
    package(Path(__file__).resolve().parent.parent,int(sys.argv[1]))

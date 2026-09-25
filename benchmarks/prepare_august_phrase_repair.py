"""Build a derived August snapshot with only the approved merge repair."""
import ast, hashlib, shutil
from pathlib import Path


def prepare(root):
    source=root/'benchmarks/editdna_august'
    target=root/'benchmarks/editdna_august_repaired_runtime'
    old=(source/'worker/pipeline.py').read_text()
    assert hashlib.sha256(old.encode()).hexdigest()=='6285b8dae54c94691cc5be8fe0b5f61ff60c6373f6e330f062aa817c6e59fec1'
    current=(root/'worker/pipeline.py').read_text()
    def span(text):
        fn=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='merge_incomplete_phrases')
        return fn.lineno-1,fn.end_lineno
    a,b=span(old);c,d=span(current)
    lines=old.splitlines(keepends=True)
    fixed=''.join(lines[:a]+current.splitlines(keepends=True)[c:d]+lines[b:])
    shutil.copytree(source,target,dirs_exist_ok=True)
    (target/'worker/pipeline.py').write_text(fixed)
    return hashlib.sha256(fixed.encode()).hexdigest()

# /workspace/EditDNA-worker/worker/__init__.py

# expose submodules so "from worker import s3" works
from . import s3
from . import asr
from . import video

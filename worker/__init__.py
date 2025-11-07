# /workspace/EditDNA-worker/worker/__init__.py

# make submodules visible when someone does: `from worker import ...`

from . import video
from . import asr
from . import s3

# optional – only if the file exists
try:
    from . import vision_sampler
except Exception:
    vision_sampler = None  # so imports don’t blow up

__all__ = ["video", "asr", "s3", "vision_sampler"]

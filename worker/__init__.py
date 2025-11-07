# /workspace/EditDNA-worker/worker/__init__.py

# Make submodules visible when someone does: `from worker import ...`

from . import video
from . import asr
from . import s3

# Optional: only load if file exists
try:
    from . import vision_sampler
except Exception:
    vision_sampler = None  # fallback if not present

__all__ = ["video", "asr", "s3", "vision_sampler"]

"""
EditDNA worker package bootstrap.

RQ will execute tasks.job_render (module "tasks.py" at repo root).
That root tasks.py just forwards to worker/tasks.py::job_render.

No need to expose job_render or submodules here.
We keep this file minimal to avoid circular imports.
"""

__all__ = []

"""
EditDNA worker package bootstrap.

Notes:
- RQ will execute tasks.job_render (module `tasks.py` at repo root).
- That root tasks.py just forwards to worker/tasks.py::job_render
- No need to expose job_render here.
We keep this file minimal to avoid import confusion.
"""

# no exports on purpose

# tasks.py — shim so RQ resolves "tasks.job_render"
from jobs import job_render
__all__ = ["job_render"]

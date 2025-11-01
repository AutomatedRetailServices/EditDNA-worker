# compatibility shim for RQ:
# expose job_render so RQ can call "tasks.job_render"
from worker.tasks import job_render

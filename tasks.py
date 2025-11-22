# tasks.py  (WORKER SIDE)
# ---------------------------------------
# Este archivo existe SOLO para que RQ pueda resolver "tasks.job_render"
# cuando el job llega desde Redis.
#
# No contiene lógica nueva; simplemente reenvía a jobs.job_render,
# que es donde vive tu pipeline real.

from typing import Any, Dict

try:
    # jobs.py ya existe en tu repo y contiene la lógica real
    from jobs import job_render as _job_render_impl
except Exception as e:
    # Log fuerte para que si algo falla se vea en los logs del worker
    print("[FATAL] No se pudo importar jobs.job_render desde tasks.py:", e)
    _job_render_impl = None  # type: ignore


def job_render(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Wrapper del worker llamado por RQ como 'tasks.job_render'.
    """
    if _job_render_impl is None:
        raise RuntimeError(
            "jobs.job_render no está disponible. "
            "Revisa que jobs.py exista y defina job_render(data)."
        )

    return _job_render_impl(data)

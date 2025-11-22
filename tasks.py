# worker/tasks.py
# Funciones que el Worker GPU ejecuta via RQ
# El API encola "tasks.job_render" → y aquí debe existir

import uuid
from typing import Any, Dict, List

# Importa la función verdadera del worker
from jobs import job_render as real_job_render


def job_render(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Esta es la función que el worker ejecuta cuando recibe "tasks.job_render"
    Hace puente hacia la función real que procesa el render.
    """
    session_id = data.get("session_id", f"session-{uuid.uuid4().hex[:8]}")

    result = real_job_render(
        session_id=session_id,
        files=data.get("files", []),
        portrait=bool(data.get("portrait", True)),
        max_duration=float(data.get("max_duration", 120.0)),
        s3_prefix=data.get("s3_prefix", "editdna/outputs/")
    )

    return {
        "ok": True,
        "session_id": session_id,
        "result": result
    }

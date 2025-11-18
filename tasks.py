import logging
import os
import traceback
from typing import Dict, Any, List, Optional

import requests

# 👇 Importa el pipeline desde /worker/pipeline.py correctamente
from worker import pipeline

logger = logging.getLogger(__name__)


def _download_first_file(files: List[str], session_id: str) -> str:
    """
    Descarga el PRIMER video de la lista `files` (URL) a:

        /tmp/TMP/editdna/{session_id}/input.mp4

    Devuelve esa ruta local para pasarla a pipeline.run_pipeline(...).
    """
    if not files:
        raise ValueError("_download_first_file: empty files list")

    url = files[0]
    logger.info("⬇️  Downloading first file", extra={"url": url})

    tmp_root = os.environ.get("TMP_DIR", "/tmp/TMP/editdna")
    session_dir = os.path.join(tmp_root, session_id)
    os.makedirs(session_dir, exist_ok=True)

    local_path = os.path.join(session_dir, "input.mp4")

    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()

    with open(local_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    logger.info("✅  Downloaded file", extra={"local_path": local_path})
    return local_path


def job_render(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Job render para RQ.

    El Web layer lo manda así:

        q.enqueue("tasks.job_render", {
            "session_id": "...",
            "files": ["https://...mp4"]
        })

    Proceso:
      1. Validamos payload.
      2. Descargamos el primer video.
      3. Llamamos pipeline.run_pipeline(session_id=..., input_local=...).
      4. Devolvemos dict completo (clips, slots, composer, etc.).
    """
    try:
        logger.info("🎬 job_render called", extra={"payload": payload})

        if not isinstance(payload, dict):
            raise ValueError(f"job_render expected dict payload, got: {type(payload)}")

        # 🧷 Obtener session_id
        session_id: Optional[str] = payload.get("session_id") or payload.get("id")
        if not session_id:
            raise ValueError("job_render: missing 'session_id' in payload")

        # 🧷 Obtener lista de archivos
        files: Optional[List[str]] = payload.get("files") or payload.get("file_urls")
        if not files or not isinstance(files, list):
            raise ValueError("job_render: 'files' must be a non-empty list in payload")

        # ⬇️ Descargar el primer archivo (video)
        input_local = _download_first_file(files, session_id)

        # 🚀 Ejecutar pipeline existente (firma original)
        result = pipeline.run_pipeline(
            session_id=session_id,
            input_local=input_local,  # NO file_urls — usa input_local
        )

        if not isinstance(result, dict):
            raise ValueError("pipeline.run_pipeline must return a dict")

        result.setdefault("ok", True)
        return result

    except Exception as e:
        logger.exception("job_render failed")
        return {
            "ok": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }

"""
Wrapper para compatibilidad interna.

La implementación real del job vive en el módulo raíz `tasks.py`.
De esta forma:
    - RQ usa:      tasks.job_render
    - Código viejo puede usar: worker.tasks.job_render
y ambos ejecutan exactamente lo mismo.
"""

from tasks import job_render  # noqa: F401

"""
EditDNA worker package bootstrap.

We intentionally keep this empty to avoid circular imports.
RQ will call the top-level tasks.py (which forwards to worker/tasks.py).
"""

__all__ = []

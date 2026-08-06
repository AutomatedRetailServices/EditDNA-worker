import os
import re
from typing import Any, Dict, List, Sequence
from urllib.parse import urlsplit


def sanitize_source_identifier(source: Any, source_index: int = 0) -> str:
    """Return a stable persistable source name without local paths or secrets."""
    raw = str(source or "").strip().replace("\\", "/")
    fallback = f"source_{source_index:03d}"
    if not raw:
        return fallback
    parsed = urlsplit(raw)
    if parsed.scheme:
        path = parsed.path.lstrip("/")
        if parsed.scheme.lower() == "s3":
            return path or fallback
        return os.path.basename(path) or fallback
    clean = raw.split("?", 1)[0].split("#", 1)[0]
    if clean.startswith("/") or re.match(r"^[a-zA-Z]:/", clean):
        return os.path.basename(clean.rstrip("/")) or fallback
    parts = [part for part in clean.split("/") if part not in ("", ".")]
    if not parts or ".." in parts:
        return os.path.basename(clean.rstrip("/")) or fallback
    return "/".join(parts)


def sanitize_clean_cut_discard_diagnostics(
    diagnostics: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Sanitize diagnostic source identifiers at every persistence boundary."""
    sanitized: List[Dict[str, Any]] = []
    for item in diagnostics or []:
        clean = dict(item)
        source_index = int(clean.get("source_index", 0))
        clean["source_index"] = source_index
        clean["source_local"] = sanitize_source_identifier(
            clean.get("source_local"), source_index
        )
        sanitized.append(clean)
    return sanitized

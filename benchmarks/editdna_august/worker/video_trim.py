import cv2
import numpy as np


def smart_trim_clip(clip, words):
    """
    --- AUTO TRIM INTELIGENTE ---
    Recorta el INICIO y FINAL del clip usando:
    1) Momento donde empieza la frase real (ASR word start)
    2) Momento donde termina la frase real (ASR word end)
    3) Verificación visual (detecta momentos de "no habla")

    Si no detecta basura → NO corta nada.
    """

    start = clip["start"]
    end = clip["end"]

    # 1) Si no hay palabras asociadas, no recortamos
    if not words:
        return start, end

    # Palabra con menor timestamp (inicio real de la frase)
    phrase_start = min(w["start"] for w in words)

    # Palabra con mayor timestamp (final real)
    phrase_end = max(w["end"] for w in words)

    # 2) Márgenes opcionales (evita cortes muy bruscos)
    margin_before = 0.08   # 80ms
    margin_after = 0.12    # 120ms

    new_start = max(start, phrase_start - margin_before)
    new_end = min(end, phrase_end + margin_after)

    # 3) Verificación visual: si la persona NO está hablando al inicio o al final
    # (Movimiento mínimo + boca abierta/cerrada estimado)
    # *Muy ligero para CPU*
    if clip.get("visual_flags", {}):
        flags = clip["visual_flags"]

        # si detectamos que el inicio es malo → movemos el start
        if flags.get("start_bad", False):
            new_start = phrase_start

        # si detectamos que el final es malo → movemos el end
        if flags.get("end_bad", False):
            new_end = phrase_end

    # Si el recorte es demasiado pequeño, devolvemos original
    if new_end - new_start < 0.25:  # mínimo 250ms
        return start, end

    return float(new_start), float(new_end)

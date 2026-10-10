"""Says whether the Modal secret 'cutsell-worker' sets CUTSELL_SIMPLE_ENGINE_MODEL (which overrides the
engine's DEFAULT_MODEL). Prints only that, and the value only if it looks like a model id. No keys."""
import os, re
import modal

app = modal.App("cutsell-env-check")


@app.function(image=modal.Image.debian_slim(), secrets=[modal.Secret.from_name("cutsell-worker")], timeout=120)
def check():
    raw = os.environ.get("CUTSELL_SIMPLE_ENGINE_MODEL")
    if raw is None:
        return "CUTSELL_SIMPLE_ENGINE_MODEL: NO existe en el secreto (manda el codigo)"
    value = raw.strip()
    if not value:
        return "CUTSELL_SIMPLE_ENGINE_MODEL: existe pero vacia (manda el codigo)"
    shown = value if re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,60}", value) else "(valor que no parece un modelo; no se muestra)"
    return f"CUTSELL_SIMPLE_ENGINE_MODEL: EXISTE y vale {shown} (manda sobre el codigo)"


@app.local_entrypoint()
def main():
    print(check.remote())

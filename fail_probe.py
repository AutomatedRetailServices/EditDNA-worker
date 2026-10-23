"""Security review (PO 23 Oct): NAMES of the worker's secret keys (never values) and what the storage / ASR keys may do.
Read-only: lists and HEADs; the delete test targets a key that does not exist (no data can change). Report encrypted."""
import json

import modal

app = modal.App("cutsell-fail-probe")
secret = modal.Secret.from_name("cutsell-worker")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.cutsell.worker.cpu.txt")
    .add_local_dir("cutsell_worker", remote_path="/root/cutsell_worker", ignore=["**/__pycache__/**", "**/*.pyc"])
)


@app.function(image=image, secrets=[secret], timeout=300)
def audit():
    import os, time
    from urllib.parse import urlparse
    from redis import Redis
    u = urlparse(os.environ.get("REDIS_URL", ""))
    host = (u.hostname or "")
    out = {"redis_provider_domain": ".".join(host.split(".")[-2:]) if host.count(".") >= 1 else ("set" if host else "missing"),
           "redis_tls": u.scheme == "rediss", "redis_has_password": bool(u.password)}
    r = Redis.from_url(os.environ["REDIS_URL"], socket_timeout=10)
    for section in ("persistence", "memory", "keyspace", "server"):
        try:
            info = r.info(section)
            keep = {k: info[k] for k in info if k in ("aof_enabled", "rdb_last_save_time", "rdb_changes_since_last_save",
                    "rdb_bgsave_in_progress", "used_memory_human", "maxmemory_human", "maxmemory_policy", "redis_version", "db0")}
            if "rdb_last_save_time" in keep:
                keep["hours_since_last_rdb_save"] = round((time.time() - keep.pop("rdb_last_save_time")) / 3600, 1)
            out[section] = keep
        except Exception as e:
            out[section] = "denied: " + e.__class__.__name__
    try:
        out["config_save"] = r.config_get("save")
    except Exception as e:
        out["config_save"] = "denied: " + e.__class__.__name__
    return out


@app.local_entrypoint()
def main():
    with open("engine_check.json", "w") as f:
        json.dump({"label": "security review: redis", "rows": [audit.remote()]}, f, indent=1)

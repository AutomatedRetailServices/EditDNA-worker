# app.py — unified web API + worker with Redis-backed logs
import os, json, time, threading, logging, traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse
import redis
from jobs import process_job

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("editdna")

REDIS_URL = os.environ["REDIS_URL"]
r = redis.from_url(REDIS_URL, decode_responses=True)
JOBS_Q = "editdna:jobs"
JOB_KEY = lambda jid: f"editdna:job:{jid}"
JOB_LOG = lambda jid: f"editdna:job:{jid}:logs"

def push_log(job_id, msg):
    line = f"{int(time.time())}|{msg}"
    with r.pipeline() as p:
        p.rpush(JOB_LOG(job_id), line)
        p.ltrim(JOB_LOG(job_id), -400, -1)
        p.execute()
    log.info(f"[{job_id}] {msg}")

def update_job(job_id, **kv):
    kv["updated_at"] = int(time.time())
    r.hset(JOB_KEY(job_id), mapping=kv)

def new_job_id():
    return str(int(time.time() * 1000))

class Handler(BaseHTTPRequestHandler):
    def _json(self, code, obj):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self): self._json(200, {"ok": True})

    def do_POST(self):
        if self.path == "/render":
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
            files = body.get("files") or []
            if not files:
                return self._json(400, {"error": "files[] required"})
            job_id = new_job_id()
            job = {
                "id": job_id,
                "session_id": body.get("session_id", ""),
                "files": json.dumps(files),
                "portrait": str(bool(body.get("portrait", True))),
                "max_duration": int(body.get("max_duration", 60)),
                "audio": body.get("audio", "original"),
                "output_prefix": body.get("output_prefix", "editdna/outputs"),
                "status": "queued",
                "created_at": int(time.time()),
                "updated_at": int(time.time()),
            }
            r.hset(JOB_KEY(job_id), mapping=job)
            r.lpush(JOBS_Q, job_id)
            log.info(f"[{job_id}] queued")
            return self._json(202, {"job_id": job_id, "status": "queued"})
        self._json(404, {"error": "not found"})

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/health":
            try:
                pong = r.ping()
            except Exception as e:
                return self._json(500, {"ok": False, "redis_error": str(e)})
            return self._json(200, {"ok": True, "redis": pong})
        if url.path.startswith("/jobs/"):
            parts = url.path.split("/")
            if len(parts) >= 3:
                job_id = parts[2]
                if len(parts) >= 4 and parts[3] == "logs":
                    logs = r.lrange(JOB_LOG(job_id), 0, -1)
                    return self._json(200, {"job_id": job_id, "logs": logs})
                job = r.hgetall(JOB_KEY(job_id))
                if not job:
                    return self._json(404, {"error": "job not found"})
                return self._json(200, job)
        self._json(404, {"error": "not found"})

def run_web():
    port = int(os.getenv("PORT", "8000"))
    srv = HTTPServer(("0.0.0.0", port), Handler)
    log.info(f"web listening on :{port}")
    srv.serve_forever()

def worker_loop():
    log.info("worker starting")
    while True:
        try:
            job_id = r.rpop(JOBS_Q)
            if not job_id:
                time.sleep(1)
                continue
            update_job(job_id, status="processing")
            push_log(job_id, "picked up by worker")
            try:
                process_job(job_id, r, push_log, update_job)
                update_job(job_id, status="done")
                push_log(job_id, "completed successfully")
            except Exception as e:
                tb = traceback.format_exc(limit=8)
                update_job(job_id, status="error", error=str(e))
                push_log(job_id, f"ERROR: {e}\n{tb}")
        except Exception as e:
            log.exception(f"worker_loop error: {e}")
            time.sleep(2)

if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "worker"
    if mode == "web":
        run_web()
    elif mode == "worker":
        worker_loop()
    elif mode == "both":
        threading.Thread(target=worker_loop, daemon=True).start()
        run_web()

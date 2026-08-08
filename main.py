# main.py
from fastapi import FastAPI
from web.routes_render import router as render_router
from web.routes_benchmark import router as benchmark_router
from web.routes_v1 import router as v1_router

app = FastAPI(title="EditDNA.ai API", version="1.0")

# Add routes
app.include_router(render_router)
app.include_router(benchmark_router)
app.include_router(v1_router)

# Optional health check
@app.get("/healthz")
def healthz():
    return {"ok": True}

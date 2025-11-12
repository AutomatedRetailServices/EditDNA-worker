# main.py
from fastapi import FastAPI
from web.routes_render import router as render_router

app = FastAPI(title="EditDNA.ai API", version="1.0")

# Add routes
app.include_router(render_router)

# Optional health check
@app.get("/healthz")
def healthz():
    return {"ok": True}

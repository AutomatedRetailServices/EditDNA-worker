FROM python:3.11-slim

# --- System dependencies (FFmpeg included) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg build-essential && \
    rm -rf /var/lib/apt/lists/*

# --- Environment settings ---
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# --- Python dependencies ---
COPY requirements.txt .
RUN pip install -r requirements.txt

# --- App code ---
COPY app.py ./

# --- Expose port (Render sets PORT) ---
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "${PORT:-8000}"]


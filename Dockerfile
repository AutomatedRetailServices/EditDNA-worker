FROM python:3.11-slim

# System deps (from apt.txt)
COPY apt.txt /tmp/apt.txt
RUN apt-get update \
 && xargs -a /tmp/apt.txt apt-get install -y --no-install-recommends \
 && rm -rf /var/lib/apt/lists/* /tmp/apt.txt

# Python env
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Python deps
COPY requirements.txt .
RUN pip install -r requirements.txt

# App code
COPY . ./

# Expose (Render sets $PORT)
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]

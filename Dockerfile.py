FROM python:3.12-slim

# Install ffmpeg, required by moviepy for video/audio encoding
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Railway injects the actual port via $PORT — don't hardcode 8000
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
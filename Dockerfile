FROM python:3.11-slim

# Install ffmpeg + yt-dlp + system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    yt-dlp \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# App
WORKDIR /app
COPY . /app

# Hugging Face Spaces uses port 7860
ENV PORT=7860
EXPOSE 7860

# FastAPI runs on 0.0.0.0:7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--timeout-keep-alive", "300"]

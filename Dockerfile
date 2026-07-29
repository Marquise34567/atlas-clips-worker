FROM python:3.11-slim

# Install ffmpeg + system deps + fonts for caption rendering (yt-dlp via pip)
# fonts-dejavu-core provides DejaVu Sans (Arial equivalent) for ASS subtitles
# fonts-liberation provides Liberation Sans (another Arial equivalent)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    fonts-dejavu-core \
    fonts-liberation \
    fontconfig \
    && rm -rf /var/lib/apt/lists/* \
    && fc-cache -f

# Install yt-dlp (latest) + Python deps
RUN pip install --no-cache-dir -U yt-dlp
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

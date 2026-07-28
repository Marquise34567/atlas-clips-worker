---
title: Atlas Clips
emoji: 🎬
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Atlas Clips — Hugging Face Space

Video processing pipeline for Atlas Clips. Analyzes Twitch VODs / YouTube videos, finds the best moments with Groq, and exports vertical 9:16 clips with webcam on top + gameplay on bottom.

## Setup

### 1. Create the Space

1. Go to https://huggingface.co/new-space
2. Name it `atlas-clips`
3. SDK: **Docker**
4. License: MIT
5. Create

### 2. Add secrets

In the Space → **Settings** → **Repository secrets**, add:

| Key | Value |
|-----|-------|
| `R2_ACCOUNT_ID` | your Cloudflare account ID |
| `R2_ACCESS_KEY_ID` | R2 access key |
| `R2_SECRET_ACCESS_KEY` | R2 secret key |
| `R2_BUCKET` | `autoeditor` |
| `GROQ_API_KEY` | `gsk_...` |

### 3. Push the code

```bash
cd atlas-clips-hf
git init
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/atlas-clips
git add .
git commit -m "Atlas Clips worker"
git push space main
```

The Space builds the Dockerfile automatically and exposes the API at:
```
https://YOUR_USERNAME-atlas-clips.hf.space
```

### 4. Wire Vercel

The `vercel.json` in the repo root already routes atlas-clips endpoints to the HF Space. Just update the URL to match your Space:

```json
{
  "source": "/api/atlas-clips/analyze",
  "destination": "https://YOUR_USERNAME-atlas-clips.hf.space/api/atlas-clips/analyze"
}
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/api/atlas-clips/analyze` | Groq moment detection from transcript |
| POST | `/api/atlas-clips/analyze-comments` | Groq comment-style analysis |
| POST | `/api/atlas-clips/process-clip` | Cut + reframe + upload single clip |
| POST | `/api/atlas-clips/process-batch` | Parallel clip processing |
| POST | `/api/atlas-clips/pipeline` | Full end-to-end pipeline |

## Webcam Layout

Vertical 9:16 (1080×1920):
- **Top half** (1080×960): webcam feed (speaker)
- **Bottom half** (1080×960): gameplay / background video

## Cost

**$0** — Hugging Face Spaces free tier includes 2 vCPU + 16GB RAM. No credit card required. The Space sleeps after 48h of inactivity and wakes on the first request (~30s cold start).

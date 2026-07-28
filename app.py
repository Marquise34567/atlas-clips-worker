"""
Atlas Clips — Hugging Face Space worker
========================================

Standalone FastAPI app that handles the Atlas Clips video-processing pipeline:
  1. Groq moment detection (port of backend/src/routes/atlasClips.ts)
  2. Twitch/YouTube download (yt-dlp)
  3. ffmpeg webcam-overlay + reframing (webcam on top, gameplay on bottom)
  4. Upload finished clips to Cloudflare R2
  5. Parallel clip processing with concurrent.futures

Deploy:
  Push to a Hugging Face Space (Docker SDK type).
  Set secrets in the Space Settings → Repository secrets:
    R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET, GROQ_API_KEY

The app runs on port 7860 (HF Spaces default).
"""

import json
import os
import subprocess
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3
from botocore.config import Config as BotoConfig
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GROQ_FAST_MODEL = "llama-3.1-8b-instant"
GROQ_QUALITY_MODEL = "llama-3.3-70b-versatile"

# Working directory for downloaded videos + processed clips
WORK_DIR = Path(tempfile.gettempdir()) / "atlas-clips"
WORK_DIR.mkdir(parents=True, exist_ok=True)

# Max parallel clip-processing workers (HF free tier: 2 vCPU)
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "2"))

app = FastAPI(title="Atlas Clips API", version="1.0.0")

# CORS — allow Vercel frontend + Railway backend to call this
ALLOWED_ORIGINS = os.environ.get("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_source_type(url: str) -> Optional[str]:
    import re
    trimmed = (url or "").strip()
    if not trimmed:
        return None
    if re.match(r"^(https?://)?(www\.)?(twitch\.tv/videos/\d+)", trimmed, re.I):
        return "twitch"
    if re.match(r"^(https?://)?(www\.)?(youtube\.com/(watch\?v=|shorts/|live/|embed/)|youtu\.be/)[\w-]{6,}", trimmed, re.I):
        return "youtube"
    return None


def _get_groq_client():
    from groq import Groq
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GROQ_API_KEY not configured")
    keys = [k.strip() for k in key.split(",") if k.strip()]
    return Groq(api_key=keys[0])


def _get_r2_client():
    account_id = os.environ.get("R2_ACCOUNT_ID", "").strip()
    access_key = os.environ.get("R2_ACCESS_KEY_ID", "").strip()
    secret_key = os.environ.get("R2_SECRET_ACCESS_KEY", "").strip()
    bucket = os.environ.get("R2_BUCKET", "autoeditor").strip()
    if not all([account_id, access_key, secret_key]):
        raise RuntimeError("R2 credentials not configured")
    endpoint = f"https://{account_id}.r2.cloudflarestorage.com"
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=BotoConfig(retries={"max_attempts": 5, "mode": "adaptive"}),
    ), bucket


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def detect_moments(
    transcript: str,
    video_duration: float = 0,
    source_type: Optional[str] = None,
    reframe_config: Optional[Dict] = None,
    fast_mode: bool = False,
) -> Dict[str, Any]:
    """Analyze a transcript with Groq and return the top viral moments."""

    client = _get_groq_client()
    source_label = {
        "youtube": "YouTube video",
        "twitch": "Twitch VOD",
    }.get(source_type, "video")

    rc = reframe_config or {}
    webcam_pip_first = rc.get("webcamPipFirst", True)
    enable_captions = rc.get("enableCaptions", False)
    use_ai_reframe = rc.get("useAiReframe", False)
    edit_style = rc.get("editStyle", "retention")

    prompt = f"""Analyze this {source_label} transcript and identify the top 10 most viral moments for short-form VERTICAL clips. Focus on moments that would work well for TikTok, YouTube Shorts, or Instagram Reels.

For each moment, provide:
1. Start time (in seconds)
2. End time (in seconds)
3. A catchy title optimized for social media
4. Brief description of why it's viral
5. Viral score (0-100) based on engagement potential
6. Category (hook, emotional_peak, story_arc, insight, funny, controversial, tutorial, cliffhanger, chat_reaction)
7. The exact transcript text for that moment
8. Recommended edit style for this specific clip (retention, commentary, or reaction)

{source_label} duration: {video_duration} seconds
{('AI Reframe will be applied for vertical 9:16 format.' if use_ai_reframe else 'Standard reframing will be applied.')}
{('A webcam feed is present — place it as a picture-in-picture (PIP) overlay FIRST so the speaker stays visible, then reframe the background around it.' if webcam_pip_first else 'No webcam PIP overlay requested.')}
{('Open-caption subtitles will be burned into the exported clips.' if enable_captions else 'No captions will be burned in.')}
Selected edit style: {edit_style}

Transcript:
{transcript}

Return only valid JSON in this exact format:
{{
  "clips": [
    {{
      "startTime": number,
      "endTime": number,
      "title": string,
      "description": string,
      "viralScore": number,
      "category": "hook|emotional_peak|story_arc|insight|funny|controversial|tutorial|cliffhanger|chat_reaction",
      "transcript": string,
      "triggeredBy": "ai",
      "recommendedStyle": "retention|commentary|reaction"
    }}
  ],
  "analysisSummary": string,
  "topRecommendation": {{
    "startTime": number,
    "endTime": number,
    "title": string,
    "description": string,
    "viralScore": number,
    "category": string,
    "transcript": string,
    "triggeredBy": "ai"
  }}
}}"""

    model = GROQ_FAST_MODEL if fast_mode else GROQ_QUALITY_MODEL
    completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": f"You are an expert at identifying viral content moments in {source_label}s. You analyze transcripts to find the most engaging, shareable clips that will perform well on short-form platforms.",
            },
            {"role": "user", "content": prompt},
        ],
        model=model,
        temperature=0.7,
        max_tokens=4000,
        response_format={"type": "json_object"},
    )

    text = completion.choices[0].message.content or "{}"
    result = json.loads(text)
    result["totalDuration"] = video_duration
    return result


def analyze_comments(transcript: str, comments: str = "") -> Dict[str, Any]:
    """Analyze transcript + comments to recommend the best editing style."""

    client = _get_groq_client()
    comments_text = comments or "No specific comments provided"

    prompt = f"""Analyze this video transcript and any viewer comments to determine the best editing style for vertical short-form content.

Transcript:
{transcript}

Viewer Comments:
{comments_text}

Analyze and recommend the best editing style from these options:
1. "retention" - Fast-paced, jump cuts, rapid visual changes to maximize viewer retention
2. "commentary" - Slower pace, focus on content delivery, educational/informational style
3. "reaction" - Emphasis on emotional responses, dramatic reveals, audience engagement

For each style, provide a score (0-100) and reasoning. Then recommend the best overall style.

Return only valid JSON in this exact format:
{{
  "detectedStyle": "retention|commentary|reaction",
  "confidence": number,
  "reasoning": "brief explanation of why this style was chosen",
  "styleRecommendations": {{
    "retention": {{ "score": number, "reasoning": "string" }},
    "commentary": {{ "score": number, "reasoning": "string" }},
    "reaction": {{ "score": number, "reasoning": "string" }}
  }}
}}"""

    completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are an expert video editor and content strategist who analyzes video content and viewer engagement to recommend the best editing style for short-form vertical videos.",
            },
            {"role": "user", "content": prompt},
        ],
        model=GROQ_QUALITY_MODEL,
        temperature=0.7,
        max_tokens=2000,
        response_format={"type": "json_object"},
    )

    text = completion.choices[0].message.content or "{}"
    return json.loads(text)


def download_source(url: str, job_id: str) -> Dict[str, Any]:
    """Download a Twitch VOD or YouTube video with yt-dlp."""

    source_type = _detect_source_type(url)
    if not source_type:
        raise ValueError(f"Unsupported URL: {url}")

    cache_dir = WORK_DIR / job_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(cache_dir / "source.%(ext)s")

    cmd = [
        "yt-dlp",
        "-f", "bestvideo[height<=1080]+bestaudio/best[height<=1080]/best",
        "--merge-output-format", "mp4",
        "-o", output_template,
        "--no-playlist",
        url,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=800)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {result.stderr[-2000:]}")

    files = list(cache_dir.glob("source.*"))
    if not files:
        raise RuntimeError("Download completed but no file found")

    video_path = str(files[0])

    # Probe duration with ffprobe
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", video_path],
        capture_output=True, text=True,
    )
    duration = 0.0
    if probe.returncode == 0:
        try:
            duration = float(json.loads(probe.stdout)["format"]["duration"])
        except (KeyError, ValueError):
            pass

    return {
        "videoPath": video_path,
        "duration": duration,
        "sourceType": source_type,
        "sizeBytes": files[0].stat().st_size,
    }


def _build_reframe_filter(
    has_webcam: bool,
    webcam_position: str = "top",
    background_position: str = "bottom",
    enable_captions: bool = False,
) -> str:
    """Build the ffmpeg filter_complex string for vertical 9:16 reframing.

    Layout (1080x1920 vertical):
      - With webcam: top half = webcam (speaker), bottom half = gameplay
      - Without webcam: full frame = scaled source
    """

    W, H = 1080, 1920
    half_h = H // 2  # 960

    if has_webcam:
        filters = [
            "[0:v]split=2[webcam_src][bg_src]",
            # Webcam: top half of source → 1080x960
            f"[webcam_src]crop=iw:ih/2:0:0,scale={W}:{half_h}:force_original_aspect_ratio=increase,"
            f"crop={W}:{half_h}[webcam]",
            # Gameplay: bottom half of source → 1080x960
            f"[bg_src]crop=iw:ih/2:0:ih/2,scale={W}:{half_h}:force_original_aspect_ratio=increase,"
            f"crop={W}:{half_h}[bg]",
            # Stack: webcam on top, gameplay on bottom
            "[webcam][bg]vstack[out]",
        ]
    else:
        filters = [
            f"[0:v]scale={W}:{H}:force_original_aspect_ratio=increase,"
            f"crop={W}:{H}[out]",
        ]

    return ";".join(filters)


def process_single_clip(
    video_path: str,
    start_time: float,
    end_time: float,
    clip_title: str = "Clip",
    has_webcam: bool = False,
    webcam_position: str = "top",
    background_position: str = "bottom",
    enable_captions: bool = False,
    job_id: str = "",
) -> Dict[str, Any]:
    """Cut, reframe, and encode a single clip. Uploads to R2."""

    clip_id = str(uuid.uuid4())[:8]
    output_dir = WORK_DIR / job_id / "clips"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir / f"clip_{clip_id}.mp4")

    duration = max(0.1, end_time - start_time)
    filter_str = _build_reframe_filter(
        has_webcam=has_webcam,
        webcam_position=webcam_position,
        background_position=background_position,
        enable_captions=enable_captions,
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(start_time),
        "-i", video_path,
        "-t", str(duration),
        "-filter_complex", filter_str,
        "-map", "[out]",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-r", "30",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=540)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr[-3000:]}")

    # Upload to R2
    r2_client, bucket = _get_r2_client()
    r2_key = f"atlas-clips/{job_id}/clip_{clip_id}.mp4"

    file_size = os.path.getsize(output_path)
    with open(output_path, "rb") as f:
        r2_client.upload_fileobj(
            f,
            bucket,
            r2_key,
            ExtraArgs={"ContentType": "video/mp4"},
        )

    # Clean up local file
    try:
        os.remove(output_path)
    except OSError:
        pass

    account_id = os.environ.get("R2_ACCOUNT_ID", "").strip()
    public_url = f"https://{account_id}.r2.cloudflarestorage.com/{bucket}/{r2_key}"

    return {
        "clipId": clip_id,
        "r2Key": r2_key,
        "publicUrl": public_url,
        "title": clip_title,
        "startTime": start_time,
        "endTime": end_time,
        "duration": duration,
        "sizeBytes": file_size,
    }


def process_batch(
    video_path: str,
    clips: List[Dict[str, Any]],
    has_webcam: bool = False,
    webcam_position: str = "top",
    background_position: str = "bottom",
    enable_captions: bool = False,
    job_id: str = "",
) -> List[Dict[str, Any]]:
    """Process multiple clips in parallel using ThreadPoolExecutor."""

    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_clip = {}
        for i, clip in enumerate(clips):
            future = executor.submit(
                process_single_clip,
                video_path=video_path,
                start_time=clip["startTime"],
                end_time=clip["endTime"],
                clip_title=clip.get("title", f"Clip {i+1}"),
                has_webcam=has_webcam,
                webcam_position=webcam_position,
                background_position=background_position,
                enable_captions=enable_captions,
                job_id=job_id,
            )
            future_to_clip[future] = i

        for future in as_completed(future_to_clip):
            clip_idx = future_to_clip[future]
            try:
                results.append(future.result())
            except Exception as e:
                errors.append({"clipIndex": clip_idx, "error": str(e)})
                results.append({
                    "clipIndex": clip_idx,
                    "error": str(e),
                    "title": clips[clip_idx].get("title", f"Clip {clip_idx+1}"),
                })

    return results


def run_full_pipeline(
    source_url: str,
    transcript: str = "",
    reframe_config: Optional[Dict] = None,
    fast_mode: bool = False,
    max_clips: int = 10,
) -> Dict[str, Any]:
    """End-to-end: detect moments → download → process all clips in parallel."""

    job_id = str(uuid.uuid4())[:12]
    rc = reframe_config or {}
    has_webcam = rc.get("webcamPipFirst", True)
    webcam_position = rc.get("webcamPosition", "top")
    background_position = rc.get("backgroundPosition", "bottom")
    enable_captions = rc.get("enableCaptions", False)

    # Step 1: Detect moments
    moments = detect_moments(
        transcript=transcript,
        source_type=_detect_source_type(source_url),
        reframe_config=rc,
        fast_mode=fast_mode,
    )

    clips = moments.get("clips", [])[:max_clips]
    if not clips:
        return {"jobId": job_id, "clips": [], "error": "no_clips_detected"}

    # Step 2: Download
    download_result = download_source(url=source_url, job_id=job_id)
    video_path = download_result["videoPath"]

    # Step 3: Process all clips in parallel
    processed = process_batch(
        video_path=video_path,
        clips=clips,
        has_webcam=has_webcam,
        webcam_position=webcam_position,
        background_position=background_position,
        enable_captions=enable_captions,
        job_id=job_id,
    )

    return {
        "jobId": job_id,
        "analysis": moments,
        "downloadedVideo": download_result,
        "clips": processed,
        "clipCount": len(processed),
    }


# ---------------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------------

class AnalyzeReq(BaseModel):
    transcript: str
    videoUrl: Optional[str] = None
    videoDuration: float = 0
    sourceType: Optional[str] = None
    reframeConfig: Optional[Dict] = None
    fastMode: bool = False

class AnalyzeCommentsReq(BaseModel):
    transcript: str
    comments: str = ""

class ProcessClipReq(BaseModel):
    videoPath: str
    startTime: float
    endTime: float
    clipTitle: str = "Clip"
    hasWebcam: bool = False
    webcamPosition: str = "top"
    backgroundPosition: str = "bottom"
    enableCaptions: bool = False
    jobId: str = ""

class ProcessBatchReq(BaseModel):
    videoPath: str
    clips: List[Dict[str, Any]]
    hasWebcam: bool = False
    webcamPosition: str = "top"
    backgroundPosition: str = "bottom"
    enableCaptions: bool = False
    jobId: str = ""

class PipelineReq(BaseModel):
    sourceUrl: str
    transcript: str = ""
    reframeConfig: Optional[Dict] = None
    fastMode: bool = False
    maxClips: int = 10


# ---------------------------------------------------------------------------
# Endpoints — paths match what the Vercel frontend expects
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "service": "atlas-clips"}


@app.post("/api/atlas-clips/analyze")
async def analyze(req: AnalyzeReq):
    try:
        source_type = req.sourceType or (_detect_source_type(req.videoUrl) if req.videoUrl else None)
        result = detect_moments(
            transcript=req.transcript,
            video_duration=req.videoDuration,
            source_type=source_type,
            reframe_config=req.reframeConfig,
            fast_mode=req.fastMode,
        )
        return result
    except Exception as e:
        return JSONResponse({"error": "analyze_failed", "details": str(e)}, status_code=500)


@app.post("/api/atlas-clips/analyze-comments")
async def analyze_comments_endpoint(req: AnalyzeCommentsReq):
    try:
        result = analyze_comments(transcript=req.transcript, comments=req.comments)
        return result
    except Exception as e:
        return JSONResponse({"error": "analyze_comments_failed", "details": str(e)}, status_code=500)


@app.post("/api/atlas-clips/process-clip")
async def process_clip(req: ProcessClipReq):
    try:
        result = process_single_clip(
            video_path=req.videoPath,
            start_time=req.startTime,
            end_time=req.endTime,
            clip_title=req.clipTitle,
            has_webcam=req.hasWebcam,
            webcam_position=req.webcamPosition,
            background_position=req.backgroundPosition,
            enable_captions=req.enableCaptions,
            job_id=req.jobId,
        )
        return result
    except Exception as e:
        return JSONResponse({"error": "process_clip_failed", "details": str(e)}, status_code=500)


@app.post("/api/atlas-clips/process-batch")
async def process_batch_endpoint(req: ProcessBatchReq):
    try:
        result = process_batch(
            video_path=req.videoPath,
            clips=req.clips,
            has_webcam=req.hasWebcam,
            webcam_position=req.webcamPosition,
            background_position=req.backgroundPosition,
            enable_captions=req.enableCaptions,
            job_id=req.jobId,
        )
        return {"clips": result, "count": len(result)}
    except Exception as e:
        return JSONResponse({"error": "process_batch_failed", "details": str(e)}, status_code=500)


@app.post("/api/atlas-clips/pipeline")
async def pipeline(req: PipelineReq):
    try:
        result = run_full_pipeline(
            source_url=req.sourceUrl,
            transcript=req.transcript,
            reframe_config=req.reframeConfig,
            fast_mode=req.fastMode,
            max_clips=req.maxClips,
        )
        return result
    except Exception as e:
        return JSONResponse({"error": "pipeline_failed", "details": str(e)}, status_code=500)

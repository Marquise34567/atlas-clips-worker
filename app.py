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
import sqlite3
import subprocess
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

# ---------------------------------------------------------------------------
# Atlas Heuristic Moment Detection (ported from backend/autoeditor.py)
# Text-only scoring: keyword, intrigue, structural, sentiment, sentiment_spike
# No video file needed — works on transcript segments alone
# ---------------------------------------------------------------------------

import math
import re as _re

HOOK_KEYWORDS: Dict[str, float] = {
    "wtf": 1.8, "shocking": 1.6, "unbelievable": 1.4, "no way": 1.3,
    "plot twist": 1.5, "insane": 1.4, "crazy": 1.3, "wild": 1.2,
    "unexpected": 1.1, "funny": 1.1, "funniest": 1.2, "hilarious": 1.2,
    "lol": 0.9, "laugh": 0.9, "laughing": 1.0, "scary": 1.2,
    "terrifying": 1.3, "creepy": 1.0, "spooky": 0.9, "emotional": 1.2,
    "heartbreaking": 1.4, "sad": 1.0, "cry": 1.0, "crying": 1.1,
    "happy": 1.0, "joy": 1.0, "exciting": 1.1, "excited": 1.0,
    "amazing": 1.1, "incredible": 1.2, "goosebumps": 1.2, "omg": 1.3,
    "let's go": 1.2, "clutch": 1.1, "play": 0.8, "play of the game": 1.6,
    "play of the game": 1.6, "potg": 1.5, "highlight": 1.0,
    "comeback": 1.3, "destroyed": 1.1, "wiped": 1.0, "rage": 1.2,
    "screaming": 1.1, "freaking out": 1.2, "broken": 1.0, "glitch": 0.9,
    "speedrun": 1.0, "world record": 1.5, "wr": 1.3, "pb": 1.0,
    "personal best": 1.1, "ranked": 0.8, "tier": 0.7, "champion": 1.0,
    "winner": 1.0, "victory": 1.1, "defeat": 0.9, "boss": 0.8,
    "level up": 0.9, "achievement": 0.9, "unlock": 0.8,
}

CLIFFHANGER_KEYWORDS: Dict[str, float] = {
    "next": 1.4, "part 2": 1.6, "to be continued": 1.8, "stay tuned": 1.4,
    "but then": 1.2, "not over": 1.4, "coming up": 1.2, "mystery": 1.3,
    "unfinished": 1.5, "wait": 0.8, "hold on": 1.0, "actually": 0.7,
}

_SENTENCE_END_RE = _re.compile(r"[.!?…]+[\"')\]]*$")

_POS_WORDS = ["great", "amazing", "love", "best", "awesome", "wow", "funny",
              "happy", "incredible", "perfect", "beautiful", "fantastic",
              "brilliant", "legendary", "epic", "god", "goat", "insane"]
_NEG_WORDS = ["hate", "worst", "bad", "awful", "scared", "angry", "sad",
              "terrible", "horrible", "disgusting", "stupid", "dumb",
              "broken", "trash", "garbage", "cringe", "fail"]


def _ends_sentence(text: str) -> bool:
    return bool(_SENTENCE_END_RE.search(text.strip()))


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _rule_based_sentiment(text: str) -> float:
    lowered = text.lower()
    pos = sum(1 for w in _POS_WORDS if w in lowered)
    neg = sum(1 for w in _NEG_WORDS if w in lowered)
    if pos + neg == 0:
        return 0.0
    return (pos - neg) / float(pos + neg)


def _text_features(text: str) -> Tuple[float, float, float]:
    """Returns (structural, keyword, intrigue) scores for a text segment."""
    lowered = text.lower()
    hook_score = sum(weight for key, weight in HOOK_KEYWORDS.items() if key in lowered)
    intrigue_score = sum(weight for key, weight in CLIFFHANGER_KEYWORDS.items() if key in lowered)
    punctuation_boost = lowered.count("?") * 0.22 + lowered.count("!") * 0.16
    uppercase_boost = 0.4 if _re.search(r"\b[A-Z]{3,}\b", text) else 0.0
    word_count = max(1, len(_re.findall(r"[a-zA-Z']+", text)))
    conciseness = 1.0 / math.sqrt(word_count)
    sentence_end_bonus = 0.18 if _ends_sentence(text) else 0.0
    structural = punctuation_boost + uppercase_boost + conciseness + sentence_end_bonus
    return structural, hook_score, intrigue_score


def _fit_window(start: float, end: float, min_duration: float, max_duration: float) -> Tuple[float, float]:
    start = max(0.0, start)
    end = max(start + 0.05, min(max_duration, end))
    current = end - start
    if current >= min_duration:
        return start, end
    needed = min_duration - current
    start = max(0.0, start - needed / 2)
    end = min(max_duration, end + needed / 2)
    if end - start < min_duration:
        if start <= 0.0:
            end = min(max_duration, min_duration)
        elif end >= max_duration:
            start = max(0.0, max_duration - min_duration)
    return start, end


@dataclass
class _TranscriptSeg:
    start: float
    end: float
    text: str


@dataclass
class _SegmentScore:
    segment: _TranscriptSeg
    structural: float
    keyword: float
    intrigue: float
    sentiment: float
    sentiment_spike: float
    total: float


def _score_segments(segments: List[_TranscriptSeg]) -> List[_SegmentScore]:
    """Score transcript segments using the Atlas heuristic (text-only)."""
    if not segments:
        return []

    sentiments: List[float] = []
    for seg in segments:
        sentiments.append(_rule_based_sentiment(seg.text))

    # Text weight is high since we have no video/audio features
    text_weight = 1.15
    out: List[_SegmentScore] = []
    for idx, seg in enumerate(segments):
        structural, keyword, intrigue = _text_features(seg.text)
        sentiment = sentiments[idx]
        prev_sent = sentiments[idx - 1] if idx > 0 else 0.0
        sentiment_spike = abs(sentiment - prev_sent)
        total = (
            structural * 0.95 * text_weight
            + keyword * 1.30 * text_weight
            + intrigue * 0.45 * text_weight
            + abs(sentiment) * 1.0 * text_weight
            + sentiment_spike * 1.10 * text_weight
        )
        out.append(_SegmentScore(
            segment=seg, structural=structural, keyword=keyword,
            intrigue=intrigue, sentiment=sentiment,
            sentiment_spike=sentiment_spike, total=total,
        ))
    return out


def _collect_top_moments(
    scores: List[_SegmentScore],
    duration: float,
    max_count: int = 10,
    min_clip_duration: float = 15.0,
    max_clip_duration: float = 60.0,
) -> List[Dict[str, Any]]:
    """Select top viral moments using the Atlas hook candidate algorithm.

    Applies opening pressure, keyword lift, emotion lift, and deduplication
    to avoid overlapping clips.
    """
    if not scores:
        return []

    duration = max(1e-6, duration)

    def hook_score(row: _SegmentScore) -> float:
        position = row.segment.start / duration
        opening_pressure = 0.38 * _clamp((0.28 - position) / 0.28, 0.0, 1.0)
        first_seconds_bonus = 0.10 if row.segment.start <= 14.0 else 0.0
        late_penalty = 0.06 if position > 0.90 else 0.0
        keyword_lift = _clamp(row.keyword * 0.50 + row.intrigue * 0.35, 0.0, 1.8)
        emotion_lift = _clamp(abs(row.sentiment) * 0.90 + row.sentiment_spike * 1.15, 0.0, 2.8)
        peak_intensity = _clamp(
            abs(row.sentiment) * 0.70 + row.sentiment_spike * 0.85,
            0.0, 2.8,
        )
        moment_lift = peak_intensity * 0.35
        late_moment_boost = 0.35 * _clamp((position - 0.40) / 0.60, 0.0, 1.0) * peak_intensity
        return (
            row.total
            + keyword_lift
            + emotion_lift
            + moment_lift
            + late_moment_boost
            + opening_pressure
            + first_seconds_bonus
            - late_penalty
        )

    ranked = sorted(scores, key=hook_score, reverse=True)
    all_hook_scores = [hook_score(row) for row in scores]
    max_score = max(all_hook_scores) if all_hook_scores else 1.0

    chosen: List[Dict[str, Any]] = []
    used_windows: List[Tuple[float, float]] = []

    for row in ranked:
        # Fit to a 15-60 second clip window
        clip_start, clip_end = _fit_window(
            row.segment.start - 2.0,
            row.segment.end + 3.0,
            min_clip_duration,
            max_clip_duration,
        )
        clip_start = max(0.0, clip_start)
        clip_end = min(duration, clip_end)

        # Deduplicate: skip if overlaps > 5s with an existing clip
        overlaps = [
            max(0.0, min(clip_end, end) - max(clip_start, start))
            for start, end in used_windows
        ]
        if any(x >= 5.0 for x in overlaps):
            continue

        # Build reason string
        reason_bits = []
        if row.segment.start <= 14.0:
            reason_bits.append("high opening-hook pressure")
        if row.keyword > 0.9:
            reason_bits.append("strong surprise/humor language")
        if row.sentiment_spike > 0.45:
            reason_bits.append("sharp emotional shift")
        if abs(row.sentiment) > 0.5:
            reason_bits.append("strong emotional tone")
        if row.intrigue > 0.8:
            reason_bits.append("cliffhanger/curiosity hook")

        # Normalize viral score to 0-100
        raw = hook_score(row)
        viral_score = int(_clamp((raw / max_score) * 100, 30, 99))

        # Determine category
        if row.keyword > 1.5:
            category = "funny"
        elif abs(row.sentiment) > 0.5:
            category = "emotional_peak"
        elif row.intrigue > 0.8:
            category = "cliffhanger"
        elif row.sentiment_spike > 0.45:
            category = "controversial"
        else:
            category = "hook"

        chosen.append({
            "startTime": round(clip_start, 2),
            "endTime": round(clip_end, 2),
            "title": "",  # Will be filled by Groq
            "description": " ".join(reason_bits) if reason_bits else "high engagement potential",
            "viralScore": viral_score,
            "category": category,
            "transcript": row.segment.text[:200],
            "triggeredBy": "atlas_heuristic",
            "recommendedStyle": "retention",
            "_hookScore": round(raw, 3),
        })
        used_windows.append((clip_start, clip_end))
        if len(chosen) >= max_count:
            break

    return chosen


def detect_moments_heuristic(
    segments: List[Dict[str, Any]],
    video_duration: float = 0,
) -> Dict[str, Any]:
    """Run the Atlas heuristic algorithm on timestamped transcript segments.

    This replaces the Groq LLM-based detect_moments. No API calls, no token
    limits, no rate limits — pure text scoring.

    Args:
        segments: List of {start, end, text} dicts from Speaches verbose_json
        video_duration: Total video duration in seconds

    Returns:
        {clips: [...], analysisSummary: str, topRecommendation: dict, totalDuration: float}
    """
    transcript_segs = [
        _TranscriptSeg(
            start=float(s.get("start", 0)),
            end=float(s.get("end", 0)),
            text=str(s.get("text", "")).strip(),
        )
        for s in segments
        if s.get("text", "").strip() and float(s.get("end", 0)) > float(s.get("start", 0))
    ]

    if not transcript_segs:
        return {
            "clips": [],
            "analysisSummary": "No transcript segments available for analysis.",
            "topRecommendation": None,
            "totalDuration": video_duration,
        }

    scores = _score_segments(transcript_segs)
    clips = _collect_top_moments(scores, video_duration, max_count=10)

    top_rec = None
    if clips:
        best = clips[0]
        top_rec = {
            "startTime": best["startTime"],
            "endTime": best["endTime"],
            "title": best["title"],
            "description": best["description"],
            "viralScore": best["viralScore"],
            "category": best["category"],
            "transcript": best["transcript"],
            "triggeredBy": "atlas_heuristic",
        }

    summary = (
        f"Analyzed {len(transcript_segs)} transcript segments over {video_duration:.0f}s. "
        f"Atlas heuristic scored {len(scores)} segments, selected top {len(clips)} moments "
        f"by hook score (keyword lift, sentiment spike, opening pressure, emotion)."
    )

    return {
        "clips": clips,
        "analysisSummary": summary,
        "topRecommendation": top_rec,
        "totalDuration": video_duration,
    }


def _generate_clip_titles_groq(clips: List[Dict[str, Any]], source_type: str) -> List[Dict[str, Any]]:
    """Use Groq to generate catchy titles + descriptions for heuristic-selected clips.

    This is a SMALL prompt (just the clip transcripts, not the full VOD transcript)
    so it stays well within token limits. Groq is only used for copywriting, not
    for moment detection.
    """
    if not clips:
        return clips

    client = _get_groq_client()
    source_label = {"youtube": "YouTube video", "twitch": "Twitch VOD"}.get(source_type, "video")

    # Build a compact prompt with just the clip transcripts
    clip_lines = []
    for i, c in enumerate(clips):
        clip_lines.append(f"Clip {i+1} ({c['startTime']:.0f}s-{c['endTime']:.0f}s, score {c['viralScore']}): {c['transcript'][:150]}")
    clips_text = "\n".join(clip_lines)

    prompt = f"""You are a short-form content expert. For each clip below, write a catchy viral title (max 60 chars) and a one-sentence description of why it's engaging. These are from a {source_label}.

{clips_text}

Return ONLY valid JSON:
{{
  "clips": [
    {{"index": 0, "title": "...", "description": "..."}},
    ...
  ]
}}"""

    try:
        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a viral content title generator. Return only JSON."},
                {"role": "user", "content": prompt},
            ],
            model=GROQ_FAST_MODEL,
            temperature=0.8,
            max_tokens=2000,
            response_format={"type": "json_object"},
        )
        text = completion.choices[0].message.content or "{}"
        result = json.loads(text)
        title_map = {c["index"]: c for c in result.get("clips", [])}
        for i, clip in enumerate(clips):
            if i in title_map:
                clip["title"] = title_map[i].get("title", clip.get("title", f"Clip {i+1}"))
                clip["description"] = title_map[i].get("description", clip.get("description", ""))
            else:
                clip["title"] = clip.get("title", f"Clip {i+1}")
    except Exception as e:
        print(f"Groq title generation failed: {e}")
        for i, clip in enumerate(clips):
            if not clip.get("title"):
                clip["title"] = f"Clip {i+1}"

    return clips


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
        "-f", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", output_template,
        "--no-playlist",
        "--retries", "3",
        "--fragment-retries", "3",
        # Anti-bot workarounds for YouTube
        "--extractor-args", "youtube:player_client=android,ios,web_safari,web",
        "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
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


def download_audio_only(url: str, job_id: str) -> Dict[str, Any]:
    """Download only the audio track (fast) for transcription.

    Uses yt-dlp to grab the smallest audio-only stream. For Twitch VODs
    this downloads at the lowest available bitrate, then we convert to
    8kHz mono WAV with ffmpeg — tiny files, fast transcription.
    """
    source_type = _detect_source_type(url)
    if not source_type:
        raise ValueError(f"Unsupported URL: {url}")

    cache_dir = WORK_DIR / job_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(cache_dir / "audio.%(ext)s")

    # Get duration first (quick metadata fetch, no download)
    duration = 0.0
    try:
        meta = subprocess.run(
            ["yt-dlp", "--dump-json", "--no-playlist", url],
            capture_output=True, text=True, timeout=60,
        )
        if meta.returncode == 0 and meta.stdout.strip():
            info = json.loads(meta.stdout.strip().splitlines()[0])
            duration = float(info.get("duration", 0))
    except Exception:
        pass

    cmd = [
        "yt-dlp",
        "-f", "bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/best[ext=mp4]/best",
        "--extract-audio",
        "--audio-format", "wav",
        "--audio-quality", "5",  # low quality = small file = fast
        "-o", output_template,
        "--no-playlist",
        "--no-warnings",
        "--retries", "3",
        "--fragment-retries", "3",
        # Anti-bot workarounds for YouTube
        "--extractor-args", "youtube:player_client=android,ios,web_safari,web",
        "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        url,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp audio failed: {result.stderr[-1500:]}")

    files = list(cache_dir.glob("audio.*"))
    if not files:
        raise RuntimeError("Audio download completed but no file found")

    audio_path = str(files[0])

    # If duration wasn't in metadata, probe the file
    if duration == 0:
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", audio_path],
            capture_output=True, text=True,
        )
        if probe.returncode == 0:
            try:
                duration = float(json.loads(probe.stdout)["format"]["duration"])
            except (KeyError, ValueError):
                pass

    return {
        "audioPath": audio_path,
        "duration": duration,
        "sourceType": source_type,
    }


def _transcribe_chunk_speaches(
    chunk_path: str, endpoint: str, headers: dict, model: str, chunk_offset: float = 0.0
) -> List[Dict[str, Any]]:
    """Transcribe a single audio chunk via Speaches. Returns segments with timestamps.

    Uses verbose_json to get timestamped segments. The chunk_offset is added
    to each segment's start/end time to get the absolute time in the original VOD.
    """
    import requests
    try:
        with open(chunk_path, "rb") as f:
            files = {"file": (os.path.basename(chunk_path), f, "audio/wav")}
            data = {"model": model, "response_format": "verbose_json"}
            resp = requests.post(endpoint, headers=headers, files=files, data=data, timeout=300)
        if resp.status_code == 200:
            result = resp.json()
            segments = result.get("segments", [])
            # Adjust timestamps by chunk offset
            for seg in segments:
                seg["start"] = seg.get("start", 0) + chunk_offset
                seg["end"] = seg.get("end", 0) + chunk_offset
            return segments
        print(f"Speaches chunk {os.path.basename(chunk_path)} failed: HTTP {resp.status_code}")
    except Exception as e:
        print(f"Speaches chunk {os.path.basename(chunk_path)} error: {e}")
    return []


def transcribe_audio(audio_path: str, duration: float = 0) -> List[Dict[str, Any]]:
    """Transcribe audio using the Speaches faster-whisper service.

    Returns a list of timestamped segments: [{start, end, text}, ...]

    For long VODs (>5 min), splits the audio into 5-minute chunks and
    transcribes them in PARALLEL — a 3-hour VOD transcribes in ~15 min
    instead of ~45 min sequential.

    NOTE: We do NOT use silenceremove here because it would break the
    timestamp alignment. Instead we keep original timestamps so the
    heuristic algorithm can map clips back to the correct VOD position.
    """
    speaches_url = os.environ.get("SPEACHES_URL", "").strip().rstrip("/")
    speaches_key = os.environ.get("SPEACHES_API_KEY", "").strip()

    if speaches_url:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        endpoint = f"{speaches_url}/v1/audio/transcriptions"
        headers = {}
        if speaches_key:
            headers["Authorization"] = f"Bearer {speaches_key}"
        model = os.environ.get("SPEACHES_MODEL", "Systran/faster-whisper-tiny")

        cache_dir = Path(audio_path).parent
        chunk_prefix = str(cache_dir / "chunk")

        # Split into 5-minute chunks WITHOUT silence removal (preserves timestamps).
        # 8kHz mono = ~1MB per 5-min chunk (tiny, fast to upload).
        chunk_seconds = 300  # 5 minutes
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", audio_path,
                "-f", "segment", "-segment_time", str(chunk_seconds),
                "-ar", "8000", "-ac", "1",
                f"{chunk_prefix}_%03d.wav",
            ],
            capture_output=True, text=True, timeout=180,
        )

        chunk_files = sorted(cache_dir.glob("chunk_*.wav"))
        if not chunk_files:
            # Fallback: transcribe the whole file
            return _transcribe_chunk_speaches(audio_path, endpoint, headers, model)

        # Transcribe all chunks in parallel (up to 4 concurrent)
        num_chunks = len(chunk_files)
        max_parallel = min(4, num_chunks)
        print(f"Transcribing {num_chunks} chunks in parallel ({max_parallel} workers)…")

        all_segments: List[List[Dict[str, Any]]] = [[]] * num_chunks
        with ThreadPoolExecutor(max_workers=max_parallel) as pool:
            futures = {
                pool.submit(
                    _transcribe_chunk_speaches,
                    str(cf), endpoint, headers, model,
                    float(idx * chunk_seconds),  # chunk_offset
                ): idx
                for idx, cf in enumerate(chunk_files)
            }
            for future in as_completed(futures):
                idx = futures[future]
                all_segments[idx] = future.result()

        # Clean up chunks
        for cf in chunk_files:
            cf.unlink(missing_ok=True)

        # Flatten all segments
        segments = []
        for chunk_segs in all_segments:
            segments.extend(chunk_segs)
        return segments

    # Fallback: Groq Whisper (has 25MB file size limit, returns text only)
    client = _get_groq_client()
    file_size = os.path.getsize(audio_path)
    MAX_SIZE = 24 * 1024 * 1024  # 24MB safety margin

    if file_size <= MAX_SIZE:
        with open(audio_path, "rb") as f:
            resp = client.audio.transcriptions.create(
                model="whisper-large-v3-turbo",
                file=f,
                response_format="verbose_json",
            )
        # Groq returns a parsed object, convert to dicts
        segments = []
        if hasattr(resp, "segments") and resp.segments:
            for seg in resp.segments:
                segments.append({
                    "start": getattr(seg, "start", 0),
                    "end": getattr(seg, "end", 0),
                    "text": getattr(seg, "text", "").strip(),
                })
        return segments

    # Groq fallback with no segments: return empty (Speaches should be configured)
    return []


def _detect_webcam(video_path: str) -> bool:
    """Detect if a video has a webcam overlay in the top portion.

    Samples a frame at 25% into the video and checks if the top half has
    a distinct face-like region (higher contrast + edge density than the
    bottom half, which is typically gameplay).

    For Twitch VODs, the webcam is usually in the top-left or top-center
    as a small overlay. We check if the top half has significantly more
    edge detail (face features) than expected for pure gameplay.
    """
    try:
        # Extract a frame at 25% into the video
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", video_path],
            capture_output=True, text=True, timeout=15,
        )
        duration = 0.0
        if probe.returncode == 0:
            try:
                duration = float(json.loads(probe.stdout)["format"]["duration"])
            except (KeyError, ValueError):
                pass
        if duration <= 0:
            return False

        sample_time = duration * 0.25
        frame_path = str(Path(video_path).parent / "webcam_probe.jpg")
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(sample_time), "-i", video_path,
             "-frames:v", "1", "-q:v", "2", frame_path],
            capture_output=True, text=True, timeout=30,
        )
        if not Path(frame_path).exists():
            return False

        # Use OpenCV to check for faces in the top half
        try:
            import cv2
            import numpy as np
        except ImportError:
            # No OpenCV — assume webcam for Twitch VODs (common layout)
            Path(frame_path).unlink(missing_ok=True)
            return True

        frame = cv2.imread(frame_path)
        Path(frame_path).unlink(missing_ok=True)
        if frame is None:
            return False

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Check top half for faces
        top_half = gray[:h // 2, :]
        cascade_path = str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            return True  # Fallback: assume webcam

        faces = cascade.detectMultiScale(top_half, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
        return len(faces) > 0

    except Exception as e:
        print(f"Webcam detection error: {e}")
        return True  # Fallback: assume webcam (safe default for Twitch)


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
    has_webcam: Optional[bool] = None,
    webcam_position: str = "top",
    background_position: str = "bottom",
    enable_captions: bool = False,
    job_id: str = "",
) -> Dict[str, Any]:
    """Cut, reframe, and encode a single clip. Uploads to R2.

    If has_webcam is None, auto-detects webcam presence from the video.
    When webcam is detected: webcam goes on TOP, gameplay goes on BOTTOM.
    """

    # Auto-detect webcam if not specified
    if has_webcam is None:
        has_webcam = _detect_webcam(video_path)
        print(f"Webcam auto-detected: {has_webcam}")

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


# ---------------------------------------------------------------------------
# SQLite Job Store
# ---------------------------------------------------------------------------

DB_PATH = "/tmp/atlas_clips.db"

_JOB_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY,
  source_url TEXT,
  status TEXT DEFAULT 'pending',
  progress INTEGER DEFAULT 0,
  clip_count INTEGER DEFAULT 0,
  clips_json TEXT,
  analysis_json TEXT,
  error TEXT,
  created_at REAL,
  updated_at REAL
);
"""


class JobStore:
    """Lightweight SQLite-backed job store."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._lock:
            conn = self._conn()
            try:
                conn.executescript(_JOB_SCHEMA)
                conn.commit()
            finally:
                conn.close()

    def create(self, job_id: str, source_url: str = "") -> Dict[str, Any]:
        now = time.time()
        with self._lock:
            conn = self._conn()
            try:
                conn.execute(
                    "INSERT INTO jobs (id, source_url, status, progress, clip_count, created_at, updated_at) "
                    "VALUES (?, ?, 'pending', 0, 0, ?, ?)",
                    (job_id, source_url, now, now),
                )
                conn.commit()
            finally:
                conn.close()
        return self.get(job_id)

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        conn = self._conn()
        try:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            return self._row_to_dict(row) if row else None
        finally:
            conn.close()

    def update(self, job_id: str, **fields) -> Optional[Dict[str, Any]]:
        if not fields:
            return self.get(job_id)
        fields["updated_at"] = time.time()
        set_clause = ", ".join(f"{k} = ?" for k in fields.keys())
        values = list(fields.values()) + [job_id]
        with self._lock:
            conn = self._conn()
            try:
                conn.execute(f"UPDATE jobs SET {set_clause} WHERE id = ?", values)
                conn.commit()
            finally:
                conn.close()
        return self.get(job_id)

    def list(self) -> List[Dict[str, Any]]:
        conn = self._conn()
        try:
            rows = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()
            return [self._row_to_dict(r) for r in rows]
        finally:
            conn.close()

    def delete(self, job_id: str) -> bool:
        with self._lock:
            conn = self._conn()
            try:
                cur = conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
                conn.commit()
                return cur.rowcount > 0
            finally:
                conn.close()

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        for key in ("clips_json", "analysis_json"):
            if d.get(key):
                try:
                    d[key.replace("_json", "")] = json.loads(d[key])
                except (json.JSONDecodeError, TypeError):
                    d[key.replace("_json", "")] = None
            else:
                d[key.replace("_json", "")] = None
            d.pop(key, None)
        return d


job_store = JobStore()


# ---------------------------------------------------------------------------
# Job request models
# ---------------------------------------------------------------------------

class CreateJobReq(BaseModel):
    sourceUrl: str
    transcript: str = ""
    reframeConfig: Optional[Dict] = None
    fastMode: bool = False
    maxClips: int = 10

class ReframeClipReq(BaseModel):
    clipIndex: int
    reframeConfig: Optional[Dict] = None

class ManualClipReq(BaseModel):
    startTime: float
    endTime: float
    title: Optional[str] = None
    reframeConfig: Optional[Dict] = None

class ReframeBatchReq(BaseModel):
    clipIndices: List[int]
    reframeConfig: Optional[Dict] = None


# ---------------------------------------------------------------------------
# Background processing
# ---------------------------------------------------------------------------

def _run_pipeline_background(job_id: str, source_url: str, transcript: str, reframe_config: Optional[Dict], fast_mode: bool, max_clips: int):
    """Run analysis-only pipeline in a background thread.

    Flow:
      1. Download audio (yt-dlp, low quality)
      2. Transcribe with Speaches faster-whisper (parallel chunks, verbose_json)
      3. Detect moments with Atlas heuristic algorithm (NO Groq LLM — pure text scoring)
      4. Generate catchy titles with Groq (small prompt, just clip transcripts)

    The full video is NOT downloaded here — that only happens when the user
    clicks "Reframe" on a specific clip (see _run_single_clip_background).
    """
    try:
        vod_duration = 0
        segments: List[Dict[str, Any]] = []

        # If no transcript was provided, download audio and transcribe.
        if not transcript or not transcript.strip():
            job_store.update(job_id, status="downloading", progress=15)
            audio_result = download_audio_only(url=source_url, job_id=job_id)
            vod_duration = audio_result.get("duration", 0)
            job_store.update(job_id, status="transcribing", progress=30)
            segments = transcribe_audio(audio_result["audioPath"], duration=vod_duration)
            # Clean up audio file to save disk
            try:
                os.unlink(audio_result["audioPath"])
            except OSError:
                pass
        else:
            # Transcript was provided — parse into segments (no timestamps available)
            # Create pseudo-segments with estimated timestamps
            lines = [l.strip() for l in transcript.split("\n") if l.strip()]
            cursor = 0.0
            for line in lines:
                words = max(1, len(line.split()))
                dur = max(2.0, words / 2.8)
                segments.append({"start": cursor, "end": cursor + dur, "text": line})
                cursor += dur
            vod_duration = cursor

        # Run Atlas heuristic moment detection (NO Groq LLM — pure text scoring)
        job_store.update(job_id, status="analyzing", progress=70)
        source_type = _detect_source_type(source_url)
        moments = detect_moments_heuristic(
            segments=segments,
            video_duration=vod_duration,
        )
        clips = moments.get("clips", [])[:max_clips]

        # Use Groq ONLY to generate catchy titles for the selected clips
        if clips:
            job_store.update(job_id, status="titling", progress=85)
            clips = _generate_clip_titles_groq(clips, source_type or "")
            moments["clips"] = clips
            if moments.get("topRecommendation") and clips:
                moments["topRecommendation"]["title"] = clips[0].get("title", "")

        job_store.update(
            job_id,
            status="completed",
            progress=100,
            clip_count=len(clips),
            analysis_json=json.dumps(moments),
        )
    except Exception as e:
        job_store.update(job_id, status="failed", error=str(e), progress=0)


def _run_single_clip_background(
    job_id: str,
    source_url: str,
    start_time: float,
    end_time: float,
    clip_title: str,
    reframe_config: Optional[Dict],
):
    """Download source, cut a single clip, reframe, upload to R2."""
    try:
        job_store.update(job_id, status="downloading", progress=20)
        download_result = download_source(url=source_url, job_id=job_id)
        video_path = download_result["videoPath"]

        job_store.update(job_id, status="processing", progress=50)
        rc = reframe_config or {}
        # If user explicitly set webcamPipFirst, use it; otherwise auto-detect (None)
        webcam_setting = rc.get("webcamPipFirst")
        has_webcam = None if webcam_setting is None else bool(webcam_setting)
        result = process_single_clip(
            video_path=video_path,
            start_time=start_time,
            end_time=end_time,
            clip_title=clip_title,
            has_webcam=has_webcam,
            webcam_position=rc.get("webcamPosition", "top"),
            background_position=rc.get("backgroundPosition", "bottom"),
            enable_captions=rc.get("enableCaptions", False),
            job_id=job_id,
        )

        job_store.update(
            job_id,
            status="completed",
            progress=100,
            clip_count=1,
            clips_json=json.dumps([result]),
        )
    except Exception as e:
        job_store.update(job_id, status="failed", error=str(e), progress=0)


# ---------------------------------------------------------------------------
# Job management endpoints
# ---------------------------------------------------------------------------

@app.post("/api/atlas-clips/jobs")
async def create_job(req: CreateJobReq):
    job_id = str(uuid.uuid4())[:12]
    job_store.create(job_id, source_url=req.sourceUrl)
    thread = threading.Thread(
        target=_run_pipeline_background,
        args=(job_id, req.sourceUrl, req.transcript, req.reframeConfig, req.fastMode, req.maxClips),
        daemon=True,
    )
    thread.start()
    return {"id": job_id, "status": "pending"}


@app.get("/api/atlas-clips/jobs/{job_id}")
async def get_job(job_id: str):
    job = job_store.get(job_id)
    if not job:
        return JSONResponse({"error": "job_not_found"}, status_code=404)
    return {
        "id": job["id"],
        "status": job.get("status"),
        "progress": job.get("progress", 0),
        "clipCount": job.get("clip_count", 0),
        "clips": job.get("clips"),
        "analysis": job.get("analysis"),
        "error": job.get("error"),
    }


@app.get("/api/atlas-clips/jobs")
async def list_jobs():
    jobs = job_store.list()
    return {"jobs": jobs}


@app.delete("/api/atlas-clips/jobs/{job_id}")
async def delete_job(job_id: str):
    deleted = job_store.delete(job_id)
    if not deleted:
        return JSONResponse({"error": "job_not_found"}, status_code=404)
    return {"id": job_id, "deleted": True}


@app.post("/api/atlas-clips/jobs/{job_id}/reframe")
async def reframe_clip(job_id: str, req: ReframeClipReq):
    parent = job_store.get(job_id)
    if not parent:
        return JSONResponse({"error": "job_not_found"}, status_code=404)
    analysis = parent.get("analysis") or {}
    clips = analysis.get("clips", [])
    if req.clipIndex < 0 or req.clipIndex >= len(clips):
        return JSONResponse({"error": "clip_index_out_of_range"}, status_code=400)
    clip = clips[req.clipIndex]

    new_job_id = str(uuid.uuid4())[:12]
    job_store.create(new_job_id, source_url=parent.get("source_url", ""))
    thread = threading.Thread(
        target=_run_single_clip_background,
        args=(
            new_job_id,
            parent.get("source_url", ""),
            clip.get("startTime", 0),
            clip.get("endTime", 0),
            clip.get("title", f"Clip {req.clipIndex + 1}"),
            req.reframeConfig,
        ),
        daemon=True,
    )
    thread.start()
    return {"id": new_job_id, "status": "pending"}


@app.post("/api/atlas-clips/jobs/{job_id}/manual")
async def manual_clip(job_id: str, req: ManualClipReq):
    parent = job_store.get(job_id)
    if not parent:
        return JSONResponse({"error": "job_not_found"}, status_code=404)

    new_job_id = str(uuid.uuid4())[:12]
    job_store.create(new_job_id, source_url=parent.get("source_url", ""))
    thread = threading.Thread(
        target=_run_single_clip_background,
        args=(
            new_job_id,
            parent.get("source_url", ""),
            req.startTime,
            req.endTime,
            req.title or "Manual Clip",
            req.reframeConfig,
        ),
        daemon=True,
    )
    thread.start()
    return {"id": new_job_id, "status": "pending"}


@app.post("/api/atlas-clips/jobs/{job_id}/reframe-batch")
async def reframe_batch(job_id: str, req: ReframeBatchReq):
    parent = job_store.get(job_id)
    if not parent:
        return JSONResponse({"error": "job_not_found"}, status_code=404)
    analysis = parent.get("analysis") or {}
    clips = analysis.get("clips", [])

    created = []
    threads = []
    for idx in req.clipIndices:
        if idx < 0 or idx >= len(clips):
            continue
        clip = clips[idx]
        new_job_id = str(uuid.uuid4())[:12]
        job_store.create(new_job_id, source_url=parent.get("source_url", ""))
        t = threading.Thread(
            target=_run_single_clip_background,
            args=(
                new_job_id,
                parent.get("source_url", ""),
                clip.get("startTime", 0),
                clip.get("endTime", 0),
                clip.get("title", f"Clip {idx + 1}"),
                req.reframeConfig,
            ),
            daemon=True,
        )
        t.start()
        threads.append(t)
        created.append({"id": new_job_id, "status": "pending", "clipIndex": idx})

    return {"jobs": created}


# ---------------------------------------------------------------------------
# Chat monitoring endpoints (stubs — prevents 404s from the frontend)
# ---------------------------------------------------------------------------

# In-memory chat monitoring sessions (not persisted across restarts).
_chat_sessions: Dict[str, Dict[str, Any]] = {}


class ChatMonitorReq(BaseModel):
    channelId: str
    triggerPhrases: List[str] = []
    clipDuration: int = 30
    autoRecord: bool = False


class ChatStopReq(BaseModel):
    channelId: Optional[str] = None


class ChatTriggerReq(BaseModel):
    channelId: str
    chatMessage: str
    timestamp: Optional[float] = None
    username: Optional[str] = None


@app.post("/api/atlas-clips/chat/monitor")
async def chat_monitor(req: ChatMonitorReq):
    _chat_sessions[req.channelId] = {
        "channelId": req.channelId,
        "triggerPhrases": req.triggerPhrases,
        "clipDuration": req.clipDuration,
        "autoRecord": req.autoRecord,
        "startTime": time.time(),
        "clipsRecorded": 0,
        "clips": [],
        "isMonitoring": True,
    }
    return {
        "status": "monitoring",
        "channelId": req.channelId,
        "monitoring": True,
    }


@app.post("/api/atlas-clips/chat/stop")
async def chat_stop(req: ChatStopReq):
    clips_recorded = 0
    if req.channelId:
        session = _chat_sessions.pop(req.channelId, None)
        if session:
            clips_recorded = session.get("clipsRecorded", 0)
    else:
        clips_recorded = sum(s.get("clipsRecorded", 0) for s in _chat_sessions.values())
        _chat_sessions.clear()
    return {
        "status": "stopped",
        "channelId": req.channelId,
        "clipsRecorded": clips_recorded,
    }


@app.post("/api/atlas-clips/chat/trigger")
async def chat_trigger(req: ChatTriggerReq):
    session = _chat_sessions.get(req.channelId)
    if not session or not session.get("isMonitoring"):
        return {"status": "not_monitoring", "channelId": req.channelId}

    # Check if message matches any trigger phrase
    msg_lower = req.chatMessage.lower()
    matched = any(p.lower() in msg_lower for p in session.get("triggerPhrases", []))
    if not matched:
        return {"status": "no_match", "channelId": req.channelId}

    # Create a clip entry
    clip_duration = session.get("clipDuration", 30)
    now = req.timestamp or time.time()
    clip = {
        "startTime": now,
        "endTime": now + clip_duration,
        "title": f"Chat clip by {req.username or 'user'}",
        "description": f'Triggered by: "{req.chatMessage}"',
        "viralScore": 50,
        "category": "chat_reaction",
        "transcript": req.chatMessage,
        "chatMessage": req.chatMessage,
        "triggeredBy": "chat",
    }
    session["clips"].append(clip)
    session["clipsRecorded"] = session.get("clipsRecorded", 0) + 1

    return {
        "status": "clip_recorded",
        "channelId": req.channelId,
        "clip": clip,
        "autoRecorded": session.get("autoRecord", False),
    }


@app.get("/api/atlas-clips/chat/status/{channel_id}")
async def chat_status_channel(channel_id: str):
    session = _chat_sessions.get(channel_id)
    if not session:
        return {
            "status": "not_monitoring",
            "channelId": channel_id,
            "isMonitoring": False,
            "clips": [],
            "clipsRecorded": 0,
        }
    return {
        "status": "monitoring",
        "channelId": channel_id,
        "isMonitoring": True,
        "startTime": session.get("startTime"),
        "triggerPhrases": session.get("triggerPhrases", []),
        "clipDuration": session.get("clipDuration", 30),
        "autoRecord": session.get("autoRecord", False),
        "clipsRecorded": session.get("clipsRecorded", 0),
        "clips": session.get("clips", []),
    }


@app.get("/api/atlas-clips/chat/status")
async def chat_status_all():
    return {
        "status": "ok",
        "activeChannels": list(_chat_sessions.keys()),
        "sessions": [
            {
                "channelId": s["channelId"],
                "isMonitoring": s.get("isMonitoring", False),
                "clipsRecorded": s.get("clipsRecorded", 0),
                "startTime": s.get("startTime"),
            }
            for s in _chat_sessions.values()
        ],
    }

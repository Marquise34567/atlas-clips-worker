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

import glob
import json
import math
import os
import re
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
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GROQ_FAST_MODEL = "llama-3.1-8b-instant"
GROQ_QUALITY_MODEL = "llama-3.3-70b-versatile"
# Vision-capable models for webcam/scene detection from a video frame.
GROQ_VISION_MODEL = os.environ.get("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

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

def _retry(fn, attempts: int = 3, backoff: float = 2.0, label: str = "operation"):
    """Retry a callable with exponential backoff. Returns the result on success,
    raises the last exception if all attempts fail. Improves reliability for
    flaky network operations (transcription, downloads, encoding)."""
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if attempt < attempts:
                wait = backoff ** (attempt - 1)
                print(f"[retry] {label} attempt {attempt}/{attempts} failed: {e} — retrying in {wait:.1f}s")
                time.sleep(wait)
    print(f"[retry] {label} exhausted {attempts} attempts")
    raise last_exc

def _detect_source_type(url: str) -> Optional[str]:
    import re as _re
    trimmed = (url or "").strip()
    if not trimmed:
        return None
    if _re.match(r"^(https?://)?(www\.)?(twitch\.tv/videos/\d+)", trimmed, _re.I):
        return "twitch"
    # Twitch clip: twitch.tv/<channel>/clip/<id> or clips.twitch.tv/<id>
    if _re.match(r"^(https?://)?(www\.)?(twitch\.tv/[\w-]+/clip/[\w-]+|clips\.twitch\.tv/[\w-]+)", trimmed, _re.I):
        return "twitch_clip"
    if _re.match(r"^(https?://)?(www\.)?(youtube\.com/(watch\?v=|shorts/|live/|embed/)|youtu\.be/)[\w-]{6,}", trimmed, _re.I):
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
    bucket = os.environ.get("R2_BUCKET", "").strip() or "autoeditor"
    if not all([account_id, access_key, secret_key]):
        raise RuntimeError("R2 credentials not configured")
    endpoint = f"https://{account_id}.r2.cloudflarestorage.com"
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=BotoConfig(
            retries={"max_attempts": 5, "mode": "adaptive"},
            signature_version="s3v4",
            s3={"addressing_style": "path"},
        ),
    )
    return client, bucket


# ---------------------------------------------------------------------------
# Atlas Smart Moments — heuristic transcript scoring
# ---------------------------------------------------------------------------
# Pure-Python moment detection. No LLM required. Analyzes the FULL
# video, not just the opening. Mixes Atlas heuristic concepts (keyword
# lift, sentiment spike, structural features) with energy/burst detection.

# ── Trigger keywords (gaming / streaming / general viral) ──────────────────
# Weight = how strongly this word signals an exciting moment.
HOOK_KEYWORDS: Dict[str, float] = {
    # Extreme reactions (highest signal)
    "wtf": 2.0, "holy shit": 2.0, "oh my god": 1.8, "omg": 1.6, "omfg": 2.0,
    "no way": 1.6, "no shot": 1.6, "you're kidding": 1.5, "are you serious": 1.5,
    "i can't believe": 1.5, "unbelievable": 1.5, "this is insane": 1.6,
    # Excitement / hype
    "insane": 1.4, "crazy": 1.2, "wild": 1.1, "nuts": 1.1, "unreal": 1.3,
    "absurd": 1.0, "ridiculous": 1.0, "stupid": 0.8, "cracked": 1.3,
    "broken": 1.0, "overpowered": 1.0, "op": 0.8, "god mode": 1.4,
    # Gaming highlights
    "clutch": 1.4, "play of the game": 1.8, "potg": 1.6, "highlight": 1.0,
    "comeback": 1.4, "destroyed": 1.2, "wiped": 1.0, "dominated": 1.1,
    "speedrun": 1.2, "world record": 1.8, "wr": 1.5, "pb": 1.2,
    "personal best": 1.3, "ranked": 0.7, "champion": 1.0, "winner": 1.0,
    "victory": 1.1, "defeat": 0.9, "boss": 0.8, "level up": 0.8,
    "achievement": 0.9, "ace": 1.4, "triple kill": 1.5, "quad": 1.3,
    "pentakill": 1.8, "headshot": 1.2, "sniped": 1.0, "flick": 1.1,
    # Emotion
    "hilarious": 1.3, "funniest": 1.4, "laughing": 1.1, "lol": 0.9,
    "lmao": 1.1, "lmfao": 1.3, "haha": 0.8, "scary": 1.1,
    "terrifying": 1.3, "emotional": 1.2, "heartbreaking": 1.4,
    "crying": 1.1, "tears": 1.0, "rage": 1.3, "angry": 1.0,
    "furious": 1.2, "freaking out": 1.3, "screaming": 1.2,
    # Discovery / achievement
    "amazing": 1.1, "incredible": 1.2, "goosebumps": 1.3,
    "let's go": 1.3, "let me go": 1.0, "finally": 1.0,
    "first time": 0.9, "never seen": 1.1, "brand new": 0.8,
    # Drama / controversy
    "drama": 1.2, "exposed": 1.3, "controversy": 1.2, "beef": 1.0,
    "called out": 1.1, "confrontation": 1.0, "argument": 0.9,
}

# Curiosity / cliffhanger words (lower weight — useful but not primary signal)
CLIFFHANGER_KEYWORDS: Dict[str, float] = {
    "but then": 1.0, "not over": 1.0, "coming up": 0.9, "mystery": 1.0,
    "wait": 0.7, "hold on": 0.8, "actually": 0.5, "look at this": 1.0,
    "did you see": 1.0, "watch this": 1.1, "check this out": 1.0,
    "you won't believe": 1.2, "here's the thing": 0.8,
}

# Filler / low-value words (penalize — these indicate dead air)
FILLER_WORDS: set = {
    "um", "uh", "like", "you know", "i mean", "basically", "literally",
    "sort of", "kind of", "whatever", "anyway", "so yeah", "i guess",
}

_SENTENCE_END_RE = re.compile("[.!?…]+[\"')\\]]*$")
_CAPS_RE = re.compile(r"\b[A-Z]{3,}\b", re.IGNORECASE)
_PROFANITY_RE = re.compile(r"\b(fuck|shit|damn|bitch|ass|hell)\b", re.IGNORECASE)

_POS_WORDS = ["great", "amazing", "love", "best", "awesome", "wow", "funny",
              "happy", "incredible", "perfect", "beautiful", "fantastic",
              "brilliant", "legendary", "epic", "god", "goat", "insane",
              "cracked", "unreal", "pog", "poggers", "let's go", "hype"]
_NEG_WORDS = ["hate", "worst", "bad", "awful", "scared", "angry", "sad",
              "terrible", "horrible", "disgusting", "stupid", "dumb",
              "broken", "trash", "garbage", "cringe", "fail", "ruined",
              "rigged", "bs", "bullshit"]


def _ends_sentence(text: str) -> bool:
    return bool(_SENTENCE_END_RE.search(text.strip()))


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _rule_based_sentiment(text: str) -> float:
    """Sentiment in range [-1, 1]. Both strong positive AND strong negative
    indicate an engaging moment — we care about intensity, not polarity."""
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

    # Structural energy: punctuation, caps, profanity, conciseness
    punctuation_boost = 0.25 * text.count("?") + 0.2 * text.count("!")
    caps_count = len(_CAPS_RE.findall(text))
    uppercase_boost = min(caps_count * 0.35, 1.2)
    profanity_boost = min(len(_PROFANITY_RE.findall(text)) * 0.8, 0.3)

    word_count = len(re.findall(r"[a-zA-Z']+", text))
    conciseness = 1.0 if 0 < word_count <= 40 else 0.5

    sentence_end_bonus = 0.15 if _ends_sentence(text) else 0.0
    filler_count = sum(1 for w in FILLER_WORDS if w in lowered)
    filler_penalty = min(filler_count * 0.15, 0.3)

    structural = punctuation_boost + uppercase_boost + profanity_boost + conciseness + sentence_end_bonus - filler_penalty
    return structural, hook_score, intrigue_score


def _fit_window(start: float, end: float, min_duration: float, max_duration: float) -> Tuple[float, float]:
    """Fit a clip window to [min_duration, max_duration], centered on the current segment."""
    current = end - start
    if current >= max_duration:
        needed = max_duration
        mid = (start + end) / 2
        return (mid - needed / 2, mid + needed / 2)
    if current < min_duration:
        needed = min_duration
        mid = (start + end) / 2
        return (max(0.0, mid - needed / 2), mid + needed / 2)
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
    """Score transcript segments using the Atlas Smart Moments algorithm.

    Scores each segment on:
    - Keyword density (hook words per word — normalized so short segments
      with one strong keyword don't dominate over longer rich segments)
    - Emotional intensity (abs(sentiment) — both positive and negative)
    - Sentiment spike (sudden shift vs the LOCAL context, not just prev seg)
    - Structural energy (punctuation, caps, profanity, conciseness)
    - Intrigue / curiosity hooks
    - Speech density (words per second — rapid speech = excitement)
    """
    out: List[_SegmentScore] = []
    sentiments = [_rule_based_sentiment(seg.text) for seg in segments]

    for idx, seg in enumerate(segments):
        structural, keyword, intrigue = _text_features(seg.text)
        sentiment = sentiments[idx]

        # Local context sentiment spike: compare to a 5-segment window average
        window_start = max(0, idx - 5)
        window_end = min(len(sentiments), idx + 6)
        local_avg = sum(sentiments[window_start:window_end]) / max(1, window_end - window_start)
        sentiment_spike = abs(sentiment - local_avg)

        # Speech density: words per second
        seg_duration = max(0.1, seg.end - seg.start)
        word_count = len(re.findall(r"[a-zA-Z']+", seg.text))
        words_per_sec = word_count / seg_duration
        density_boost = _clamp((words_per_sec - 2.5) * 0.15, 0.0, 0.6)

        # Keyword density: keywords per word (avoids bias toward long segments)
        keyword_density = keyword / max(1, word_count) * 10  # scale up

        total = (
            structural * 0.80
            + keyword_density * 1.40
            + keyword * 0.60        # raw keyword weight still matters
            + intrigue * 0.35
            + abs(sentiment) * 1.20  # emotional INTENSITY (either polarity)
            + sentiment_spike * 1.50  # sudden shifts = engaging
            + density_boost * 0.80
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
    min_clip_duration: float = 20.0,
    max_clip_duration: float = 60.0,
) -> List[Dict[str, Any]]:
    """Select top viral moments using windowed scoring + diversity spreading.

    Key differences from the old algorithm:
    1. NO opening pressure bias — great moments can be anywhere in the VOD.
    2. Windowed grouping: aggregates nearby segment scores into 15-60s windows
       so we find complete moments, not just one loud sentence.
    3. Temporal diversity: spreads clips across the video so we don't cluster
       all picks in one section. Divides the VOD into zones and picks the best
       from each zone before filling from the global pool.
    4. Context-aware sentiment spike: compares to local window, not just prev.
    """
    if not scores:
        return []

    n = len(scores)
    candidates: List[Dict[str, Any]] = []
    chosen: List[Dict[str, Any]] = []
    used_windows: List[Tuple[float, float]] = []

    # Slide a window across the timeline, aggregating segment scores
    for i in range(n):
        window_start = scores[i].segment.start
        window_end = min(duration, window_start + max_clip_duration)
        window_segs: List[_SegmentScore] = []
        for j in range(i, n):
            if scores[j].segment.start > window_end:
                break
            window_segs.append(scores[j])
        if not window_segs:
            continue

        window_duration = window_segs[-1].segment.end - window_segs[0].segment.start
        if window_duration < 1e-6:
            continue

        # Weighted average: more recent segments get higher weight (recency bias
        # within the window — the "punchline" matters more than the setup)
        total_score = 0.0
        for k, ws in enumerate(window_segs):
            weight = 1.0 + k * 0.05
            total_score += ws.total * weight

        # Normalize by window duration (longer windows need more total energy)
        normalized = total_score / max(1.0, window_duration / 15.0)

        # Length bonus: 20-45s clips are ideal, penalize very short or very long
        length_bonus = 1.0
        if 20 <= window_duration <= 45:
            length_bonus = 1.15
        elif window_duration < 15:
            length_bonus = 0.5

        final_score = normalized * length_bonus

        best_seg = max(window_segs, key=lambda s: s.total)
        avg_sentiment = sum(s.sentiment for s in window_segs) / len(window_segs)
        max_spike = max(s.sentiment_spike for s in window_segs)
        total_keyword = sum(s.keyword for s in window_segs)
        total_intrigue = sum(s.intrigue for s in window_segs)

        candidates.append({
            "start": window_start,
            "end": window_end,
            "score": final_score,
            "best_seg": best_seg,
            "avg_sentiment": avg_sentiment,
            "max_spike": max_spike,
            "total_keyword": total_keyword,
            "total_intrigue": total_intrigue,
            "transcript": " ".join(s.segment.text for s in window_segs)[:200],
        })

    # Sort by score descending
    candidates.sort(key=lambda c: c["score"], reverse=True)

    # Temporal diversity: divide the VOD into zones, pick best from each
    zone_size = max(duration / max_count, 1.0)
    zones: List[List[Dict[str, Any]]] = [[] for _ in range(max_count)]
    for c in candidates:
        zone_idx = min(int(c["start"] / zone_size), max_count - 1)
        zones[zone_idx].append(c)

    def _try_add(cand: Dict[str, Any]) -> bool:
        clip_start, clip_end = _fit_window(
            cand["start"], cand["end"], min_clip_duration, max_clip_duration
        )
        # Check overlap with already-chosen clips
        for ws, we in used_windows:
            overlap = max(0, min(we, clip_end) - max(ws, clip_start))
            if overlap > 5.0:
                return False
        used_windows.append((clip_start, clip_end))

        best = cand["best_seg"]
        raw = cand["score"]
        max_score = max(c["score"] for c in candidates) if candidates else 1.0
        viral_score = round(_clamp(raw / max(max_score, 0.01) * 10, 0, 10), 1)

        category = "highlight"
        reason_bits = []
        if cand["total_keyword"] > 2.0:
            category = "funny"
            reason_bits.append("strong hype/excitement language")
        elif abs(cand["avg_sentiment"]) > 0.4:
            category = "emotional_peak"
            reason_bits.append("sharp emotional shift")
        if cand["max_spike"] > 0.4:
            category = "emotional_peak"
            reason_bits.append("strong emotional tone")
        if cand["total_intrigue"] > 1.5:
            category = "cliffhanger"
            reason_bits.append("curiosity/cliffhanger hook")
        if not reason_bits:
            reason_bits.append("high engagement potential")

        chosen.append({
            "startTime": int(clip_start),
            "endTime": int(clip_end),
            "title": "",
            "description": " ".join(reason_bits),
            "viralScore": viral_score,
            "category": category,
            "transcript": cand["transcript"],
            "triggeredBy": "atlas_smart_moments",
            "recommendedStyle": "retention",
            "_hookScore": raw,
        })
        return True

    # Pick best from each zone first (temporal diversity)
    for zone in zones:
        if zone and len(chosen) < max_count:
            _try_add(zone[0])

    # Fill remaining slots from the global pool
    for cand in candidates:
        if len(chosen) >= max_count:
            break
        _try_add(cand)

    chosen.sort(key=lambda c: c["startTime"])
    return chosen



def detect_moments(
    transcript: str = "",
    video_duration: float = 0,
    source_type: Optional[str] = None,
    reframe_config: Optional[Dict] = None,
    fast_mode: bool = False,
    prompt: str = "",
) -> Dict[str, Any]:
    """Detect viral moments from a transcript using Groq LLM.

    This is the LLM-based moment detection. It sends the transcript to Groq
    and asks it to identify the most engaging moments. Falls back to the
    heuristic algorithm if Groq is unavailable.

    Args:
        transcript: Full video transcript text
        video_duration: Total video duration in seconds
        source_type: 'twitch' or 'youtube'
        reframe_config: Optional reframe configuration
        fast_mode: If True, use the faster Groq model
        prompt: Additional user prompt to guide moment selection

    Returns:
        {clips: [...], analysisSummary: str, topRecommendation: dict, totalDuration: float}
    """
    # Try Groq LLM-based detection first
    try:
        client = _get_groq_client()
        model = GROQ_FAST_MODEL if fast_mode else GROQ_QUALITY_MODEL

        source_label = "Twitch VOD" if source_type == "twitch" else "YouTube video" if source_type == "youtube" else "video"

        system_prompt = (
            "You are an expert short-form content strategist who identifies the most viral "
            "and engaging moments from video transcripts. You analyze speech patterns, emotional "
            "peaks, hook words, and narrative structure to find moments that will perform well "
            "on TikTok, YouTube Shorts, and Instagram Reels."
        )

        user_prompt = (
            f"Analyze this transcript from a {source_label} (duration: {video_duration:.0f}s) "
            f"and identify the top 10 most viral moments.{' Additional context: ' + prompt if prompt else ''}\n\n"
            f"Transcript:\n{transcript[:8000]}\n\n"
            "For each moment, provide:\n"
            '- startTime: in seconds (integer)\n'
            '- endTime: in seconds (integer, 20-60s after start)\n'
            '- title: catchy viral title (max 60 chars)\n'
            '- description: why this moment is engaging (one sentence)\n'
            '- viralScore: 0-10 (how likely to go viral)\n'
            '- category: one of "funny", "emotional_peak", "highlight", "cliffhanger", "controversial"\n'
            '- transcript: the relevant transcript snippet (max 200 chars)\n'
            '- triggeredBy: "groq_llm"\n'
            '- recommendedStyle: one of "retention", "commentary", "reaction"\n\n'
            'Return ONLY valid JSON:\n'
            '{"clips": [...], "analysisSummary": "...", "topRecommendation": {...}}'
        )

        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=model,
            temperature=0.7,
            max_tokens=4000,
            response_format={"type": "json_object"},
        )

        text = completion.choices[0].message.content
        result = json.loads(text)

        # Ensure required fields
        if "clips" not in result:
            result["clips"] = []
        if "analysisSummary" not in result:
            result["analysisSummary"] = f"Groq LLM identified {len(result['clips'])} viral moments."
        if "topRecommendation" not in result:
            clips = result.get("clips", [])
            result["topRecommendation"] = max(clips, key=lambda c: c.get("viralScore", 0)) if clips else None
        result["totalDuration"] = video_duration

        # Clean up clip fields
        for clip in result.get("clips", []):
            clip.setdefault("triggeredBy", "groq_llm")
            clip.setdefault("recommendedStyle", "retention")
            clip.setdefault("category", "highlight")
            clip.setdefault("description", "")
            clip.setdefault("transcript", "")

        return result

    except Exception as e:
        print(f"Groq LLM moment detection failed (falling back to heuristic): {e}")
        # Fall back to heuristic if we have segments
        # Parse transcript into pseudo-segments if needed
        return {
            "clips": [],
            "analysisSummary": f"Moment detection failed: {e}",
            "topRecommendation": None,
            "totalDuration": video_duration,
        }


def detect_moments_heuristic(
    segments: List[Dict[str, Any]],
    video_duration: float,
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
    if not segments:
        return {"clips": [], "analysisSummary": "No transcript segments available for analysis.", "topRecommendation": None, "totalDuration": video_duration}

    transcript_segs = [
        _TranscriptSeg(
            start=float(s.get("start", 0)),
            end=float(s.get("end", 0)),
            text=str(s.get("text", "")).strip(),
        )
        for s in segments if s.get("text", "").strip()
    ]

    scores = _score_segments(transcript_segs)
    clips = _collect_top_moments(scores, video_duration, max_count=10)

    # Pick top recommendation
    top_rec = max(clips, key=lambda c: c.get("viralScore", 0)) if clips else None
    if top_rec:
        top_rec = dict(top_rec)
        top_rec.pop("_hookScore", None)

    # Clean up clip dicts
    for c in clips:
        c.pop("_hookScore", None)

    summary = (
        f"Analyzed {len(transcript_segs)} transcript segments over "
        f"{video_duration:.0f}s. Atlas Smart Moments scored {len(scores)} "
        f"segments, selected top {len(clips)} "
        "moments using windowed grouping + temporal diversity spreading "
        "(keyword density, emotional intensity, sentiment spikes, speech energy)."
    )

    return {
        "clips": clips,
        "analysisSummary": summary,
        "topRecommendation": top_rec,
        "totalDuration": video_duration,
    }


def _generate_heuristic_titles(clips: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate titles from clip transcripts without any LLM.

    Picks the most punchy phrase from the transcript (first all-caps reaction,
    or the shortest sentence with a hook keyword). Falls back to a time-based
    title if nothing interesting is found.
    """
    for i, clip in enumerate(clips):
        text = clip.get("transcript", "").strip()
        title = f"Clip {i + 1}"

        if text:
            sentences = re.split(r"[.!?]+", text)
            best = None
            best_score = 0
            for s in sentences:
                s = s.strip()
                if not s or len(s) > 50:
                    continue
                low = s.lower()
                score = 0
                for kw, w in HOOK_KEYWORDS.items():
                    if kw in low:
                        score += w
                # All-caps = excitement
                if s.isupper():
                    score += 2.0
                if score > best_score:
                    best_score = score
                    best = s

            if best and best_score > 0.5:
                title = best[:50].rstrip() + "..." if len(best) > 50 else best

        if title == f"Clip {i + 1}":
            mins = int(clip.get("startTime", 0)) // 60
            secs = int(clip.get("startTime", 0)) % 60
            title = f"Highlight at {mins:02d}:{secs:02d}"

        clip["title"] = title
        clip["description"] = clip.get("description", "")
        clip["viralScore"] = clip.get("viralScore", 0)

    return clips


def _generate_clip_titles_groq(clips: List[Dict[str, Any]], source_type: str) -> List[Dict[str, Any]]:
    """Use Groq to generate catchy titles + descriptions for heuristic-selected clips.

    Groq is OPTIONAL — if the API key is missing or the request fails, falls back
    to heuristic title generation. The pipeline must NEVER fail because of Groq.
    """
    source_label = "YouTube video" if source_type == "youtube" else "Twitch VOD" if source_type == "twitch" else "video"

    clip_lines = []
    for i, c in enumerate(clips):
        clip_lines.append(
            f"Clip {i + 1} ({c.get('startTime', 0):.0f}s-{c.get('endTime', 0):.0f}s, "
            f"score {c.get('viralScore', 0)}): {c.get('transcript', '')}"
        )
    clips_text = "\n".join(clip_lines)

    prompt = (
        f"You are a short-form content expert. For each clip below, write a catchy viral title "
        f"(max 60 chars) and a one-sentence description of why it's engaging. These are from a "
        f"{source_label}.\n\n"
        f"{clips_text}\n\n"
        f'Return ONLY valid JSON:\n{{\n  "clips": [\n    {{"index": 0, "title": "...", "description": "..."}},\n    ...\n  ]\n}}'
    )

    try:
        client = _get_groq_client()
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
        text = completion.choices[0].message.content
        result = json.loads(text)
        title_map = {c["index"]: c for c in result.get("clips", [])}
        for i, clip in enumerate(clips):
            if i in title_map:
                clip["title"] = title_map[i].get("title", clip.get("title", ""))
                clip["description"] = title_map[i].get("description", clip.get("description", ""))
        return clips
    except Exception as e:
        print(f"Groq title generation failed (using heuristic fallback): {e}")
        return _generate_heuristic_titles(clips)


def analyze_comments(transcript: str, comments: str = "") -> Dict[str, Any]:
    """Analyze transcript + comments to recommend the best editing style.

    Groq is OPTIONAL — falls back to a heuristic style detector if Groq is
    unavailable so the endpoint never 500s.
    """
    comments_text = comments.strip() or "No specific comments provided"

    def _heuristic_style() -> Dict[str, Any]:
        low = transcript.lower()
        reaction_score = sum(w for kw, w in HOOK_KEYWORDS.items() if kw in low)
        retention_score = 0
        commentary_score = 0
        for kw in ("how to", "tutorial", "explain", "learn", "guide", "step by step", "why", "because"):
            if kw in low:
                commentary_score += 20
        mx = max(reaction_score, commentary_score, 10)
        scores = {
            "retention": min(int(mx / max(reaction_score, 1) * 60), 95) if reaction_score > 0 else 60,
            "commentary": min(int(commentary_score * 5), 100) if commentary_score > 0 else 50,
            "reaction": min(int(reaction_score * 10), 100) if reaction_score > 0 else 50,
        }
        best = max(scores, key=scores.get)
        reasoning = "Heuristic analysis based on keyword density (Groq unavailable)."
        if best == "retention":
            reasoning = "Fast-paced content detected."
        elif best == "commentary":
            reasoning = "Educational content detected."
        else:
            reasoning = "Emotional reactions detected."
        return {
            "detectedStyle": best,
            "confidence": scores[best],
            "reasoning": reasoning,
            "styleRecommendations": {k: {"score": v, "reasoning": reasoning} for k, v in scores.items()},
        }

    try:
        client = _get_groq_client()
    except Exception as e:
        print(f"Groq unavailable for comment analysis (using heuristic): {e}")
        return _heuristic_style()

    prompt = (
        f"Analyze this video transcript and any viewer comments to determine the best editing style "
        f"for vertical short-form content.\n\nTranscript:\n{transcript[:2000]}\n\nViewer Comments:\n"
        f"{comments_text[:1000]}\n\nAnalyze and recommend the best editing style from these options:\n"
        '1. "retention" - Fast-paced, jump cuts, rapid visual changes to maximize viewer retention\n'
        '2. "commentary" - Slower pace, focus on content delivery, educational/informational style\n'
        '3. "reaction" - Emphasis on emotional responses, dramatic reveals, audience engagement\n\n'
        "For each style, provide a score (0-100) and reasoning. Then recommend the best overall style.\n\n"
        'Return only valid JSON in this exact format:\n{"detectedStyle": "retention|commentary|reaction", '
        '"confidence": 85, "reasoning": "...", "styleRecommendations": {"retention": {"score": 85, '
        '"reasoning": "..."}, "commentary": {"score": 60, "reasoning": "..."}, "reaction": {"score": 70, "reasoning": "..."}}}'
    )

    try:
        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an expert video editor and content strategist who analyzes video content and viewer engagement to recommend the best editing style for short-form vertical videos."},
                {"role": "user", "content": prompt},
            ],
            model=GROQ_QUALITY_MODEL,
            temperature=0.7,
            max_tokens=2000,
            response_format={"type": "json_object"},
        )
        text = completion.choices[0].message.content
        return json.loads(text)
    except Exception as e:
        print(f"Groq comment analysis failed (using heuristic): {e}")
        return _heuristic_style()


# ---------------------------------------------------------------------------
# Download (yt-dlp)
# ---------------------------------------------------------------------------

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
        "--no-playlist",
        "--retries", "3",
        "--fragment-retries", "3",
        "--extractor-args", "youtube:player_client=android,ios,web_safari,web",
        "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "-o", output_template,
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=800)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {result.stderr[-2000:]}")

    files = list(cache_dir.glob("source.*"))
    if not files:
        raise RuntimeError("Download completed but no file found")
    video_path = str(files[0])

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
        "sizeBytes": os.stat(video_path).st_size,
    }


def download_clip_segment(url: str, job_id: str, start_time: float, end_time: float) -> Dict[str, Any]:
    """Download ONLY the clip segment (start_time → end_time) from the source.

    Uses yt-dlp with --download-sections + 5 concurrent fragment downloads
    for ~5x speedup over the old yt-dlp-URL + ffmpeg-seek approach.
    """
    source_type = _detect_source_type(url)
    if not source_type:
        raise ValueError(f"Unsupported URL: {url}")

    cache_dir = WORK_DIR / job_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(cache_dir / "clip_source.mp4")

    # Twitch clips are already short (30-60s) — download the whole thing
    # instead of trying to download a segment (which fails for clips).
    if source_type == "twitch_clip":
        print("[download] Twitch clip detected — downloading full clip")
        result = download_source(url, job_id)
        # Move/rename the downloaded file to clip_source.mp4
        downloaded = result["videoPath"]
        if downloaded != output_path:
            import shutil
            shutil.move(downloaded, output_path)
        return {
            "videoPath": output_path,
            "duration": result["duration"],
            "sourceType": source_type,
            "sizeBytes": result["sizeBytes"],
            "segmentOffset": 0.0,
        }

    pad = 3.0
    dl_start = max(0.0, start_time - pad)
    dl_end = end_time + pad
    duration = dl_end - dl_start

    ytdlp_cmd = [
        "yt-dlp",
        "-f", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[ext=mp4]/best",
        "--no-playlist",
        "--retries", "3",
        "--concurrent-fragments", "5",
        "--throttled-request-rate", "10",
        "--download-sections", f"*{dl_start}-{dl_end}",
        "--force-keyframes-at-cuts",
        "--merge-output-format", "mp4",
        "--extractor-args", "youtube:player_client=android,ios,web_safari,web",
        "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "-o", output_path,
        url,
    ]
    ytdlp_result = subprocess.run(ytdlp_cmd, capture_output=True, text=True, timeout=180)
    if ytdlp_result.returncode != 0:
        print(f"yt-dlp download-sections failed, falling back to ffmpeg seek: {ytdlp_result.stderr[-500:]}")
        return _download_clip_segment_ffmpeg(url, job_id, dl_start, duration, output_path)

    if not os.path.exists(output_path):
        candidates = list(cache_dir.glob("clip_source.*"))
        if candidates:
            os.rename(str(candidates[0]), output_path)
        else:
            raise RuntimeError("yt-dlp completed but no output file found")

    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", output_path],
        capture_output=True, text=True,
    )
    seg_duration = 0.0
    if probe.returncode == 0:
        try:
            seg_duration = float(json.loads(probe.stdout)["format"]["duration"])
        except (KeyError, ValueError):
            pass

    return {
        "videoPath": output_path,
        "duration": seg_duration,
        "sourceType": source_type,
        "sizeBytes": os.path.getsize(output_path),
        "segmentOffset": dl_start,
    }


def _download_clip_segment_ffmpeg(url: str, job_id: str, dl_start: float, duration: float, output_path: str) -> Dict[str, Any]:
    """Fallback: yt-dlp -g + ffmpeg single-connection seek (slower)."""
    ytdlp_cmd = [
        "yt-dlp", "-g", "-f",
        "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[ext=mp4]/best",
        "--no-playlist", "--retries", "3",
        "--extractor-args", "youtube:player_client=android,ios,web_safari,web",
        "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        url,
    ]
    ytdlp_result = subprocess.run(ytdlp_cmd, capture_output=True, text=True, timeout=120)
    if ytdlp_result.returncode != 0:
        raise RuntimeError(f"yt-dlp URL fetch failed: {ytdlp_result.stderr[-2000:]}")
    stream_urls = [u.strip() for u in ytdlp_result.stdout.strip().split("\n") if u.strip()]
    if len(stream_urls) == 0:
        raise RuntimeError("yt-dlp returned no stream URLs")

    # Twitch clips and some sources return a single muxed stream URL
    # (video+audio combined). Handle both single and dual stream cases.
    if len(stream_urls) == 1:
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-ss", str(dl_start),
            "-i", stream_urls[0],
            "-t", str(duration), "-c", "copy",
            "-movflags", "+faststart",
            output_path,
        ]
    else:
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-ss", str(dl_start),
            "-i", stream_urls[0], "-i", stream_urls[1],
            "-t", str(duration), "-c", "copy",
            "-movflags", "+faststart",
            output_path,
        ]
    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg segment download failed: {result.stderr[-3000:]}")
    if not os.path.exists(output_path):
        raise RuntimeError("ffmpeg completed but no output file found")

    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", output_path],
        capture_output=True, text=True,
    )
    seg_duration = 0.0
    if probe.returncode == 0:
        try:
            seg_duration = float(json.loads(probe.stdout).get("format", {}).get("duration", 0))
        except (KeyError, ValueError, json.JSONDecodeError):
            pass

    return {
        "videoPath": output_path,
        "duration": seg_duration,
        "sourceType": "unknown",
        "sizeBytes": os.path.getsize(output_path),
        "segmentOffset": dl_start,
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

    # First, get duration via --dump-json (no download)
    duration = 0.0
    meta = subprocess.run(
        ["yt-dlp", "--dump-json", "--no-playlist", url],
        capture_output=True, text=True, timeout=60,
    )
    if meta.returncode == 0:
        try:
            info = json.loads(meta.stdout.strip().splitlines()[0])
            duration = float(info.get("duration", 0))
        except (KeyError, ValueError, IndexError, Exception):
            pass

    cmd = [
        "yt-dlp",
        "-f", "bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/best[ext=mp4]/best",
        "--extract-audio", "--audio-format", "wav", "--audio-quality", "5",
        "--no-warnings", "--retries", "3", "--fragment-retries", "3",
        "--extractor-args", "youtube:player_client=android,ios,web_safari,web",
        "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "-o", output_template,
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp audio failed: {result.stderr[-1500:]}")

    files = list(cache_dir.glob("audio.*"))
    if not files:
        raise RuntimeError("Audio download completed but no file found")
    audio_path = str(files[0])

    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", audio_path],
        capture_output=True, text=True,
    )
    if probe.returncode == 0:
        try:
            duration = float(json.loads(probe.stdout)["format"].get("duration", duration))
        except (KeyError, ValueError):
            pass

    return {"audioPath": audio_path, "duration": duration, "sourceType": source_type}


# ---------------------------------------------------------------------------
# Transcription (Speaches faster-whisper, parallel chunks)
# ---------------------------------------------------------------------------

def _transcribe_chunk_speaches(
    chunk_path: str, endpoint: str, headers: dict, model: str, chunk_offset: float
) -> List[Dict[str, Any]]:
    """Transcribe a single audio chunk via Speaches. Returns segments with timestamps.

    Uses verbose_json to get timestamped segments. The chunk_offset is added
    to each segment's start/end time to get the absolute time in the original VOD.
    """
    import requests
    with open(chunk_path, "rb") as f:
        files = {"file": (os.path.basename(chunk_path), f, "audio/wav")}
        data = {"model": model, "response_format": "verbose_json"}
        resp = requests.post(endpoint, headers=headers, files=files, data=data, timeout=300)

    if resp.status_code != 200:
        raise RuntimeError(f"Speaches chunk {chunk_path} failed: HTTP {resp.status_code}: {resp.text}")

    result = resp.json()
    segments = []
    for seg in result.get("segments", []):
        segments.append({
            "start": seg["start"] + chunk_offset,
            "end": seg["end"] + chunk_offset,
            "text": seg["text"],
        })
    return segments


def transcribe_audio(audio_path: str, duration: float) -> List[Dict[str, Any]]:
    """Transcribe audio using the Speaches faster-whisper service.

    Returns a list of timestamped segments: [{start, end, text}, ...]

    For long VODs (>5 min), splits the audio into 5-minute chunks and
    transcribes them in PARALLEL — a 3-hour VOD transcribes in ~15 min
    instead of ~45 min sequential.

    NOTE: We do NOT use silenceremove here because it would break the
    timestamp alignment. Instead we keep original timestamps so the
    heuristic algorithm can map clips back to the correct VOD time.
    """
    speaches_url = os.environ.get("SPEACHES_URL", "").strip().rstrip("/")
    speaches_key = os.environ.get("SPEACHES_API_KEY", "").strip()

    if not speaches_url:
        # Fallback: Groq Whisper (has 25MB file size limit, returns text only)
        print("Speaches not configured, trying Groq Whisper fallback...")
        try:
            client = _get_groq_client()
        except Exception as e:
            print(f"Groq Whisper fallback skipped (no key): {e}")
            return []

        try:
            file_size = os.path.getsize(audio_path)
            MAX_SIZE = 25 * 1024 * 1024  # 25MB
            if file_size > MAX_SIZE:
                print("No transcription available (Speaches down, file too large for Groq Whisper)")
                return []

            with open(audio_path, "rb") as f:
                resp = client.audio.transcriptions.create(
                    model="whisper-large-v3-turbo",
                    file=f,
                    response_format="verbose_json",
                )
            segments = []
            if hasattr(resp, "segments"):
                for seg in resp.segments:
                    segments.append({
                        "start": getattr(seg, "start", 0),
                        "end": getattr(seg, "end", 0),
                        "text": getattr(seg, "text", ""),
                    })
            return segments
        except Exception as e:
            print(f"Groq Whisper transcription failed: {e}")
            print("No transcription available (Speaches down, file too large for Groq Whisper)")
            return []

    endpoint = speaches_url + "/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {speaches_key}"} if speaches_key else {}
    model = os.environ.get("SPEACHES_MODEL", "Systran/faster-whisper-tiny")

    # Split audio into 5-minute chunks for parallel transcription
    chunk_seconds = 300
    cache_dir = Path(audio_path).parent
    chunk_prefix = str(cache_dir / "chunk")

    subprocess.run(
        ["ffmpeg", "-y", "-i", audio_path, "-f", "segment",
         "-segment_time", str(chunk_seconds), "-ar", "8000", "-ac", "1",
         f"{chunk_prefix}_%03d.wav"],
        capture_output=True, text=True, timeout=180,
    )

    chunk_files = sorted(glob.glob(str(cache_dir / "chunk_*.wav")))
    if not chunk_files:
        return []

    num_chunks = len(chunk_files)
    max_parallel = min(4, num_chunks)
    print(f"Transcribing {num_chunks} chunks in parallel ({max_parallel} workers)...")

    from concurrent.futures import ThreadPoolExecutor, as_completed
    all_segments: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=max_parallel) as pool:
        futures = {
            pool.submit(_transcribe_chunk_speaches, cf, endpoint, headers, model, idx * chunk_seconds): idx
            for idx, cf in enumerate(chunk_files)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                segments = future.result()
                all_segments.extend(segments)
            except Exception as e:
                print(f"Chunk {idx} transcription error: {e}")
            # Clean up chunk file
            try:
                cf = chunk_files[idx]
                Path(cf).unlink(missing_ok=True)
            except Exception:
                pass

    all_segments.sort(key=lambda s: s["start"])
    return all_segments


# ---------------------------------------------------------------------------
# Webcam detection (Groq vision + OpenCV fallback)
# ---------------------------------------------------------------------------

def _detect_webcam_groq(video_path: str) -> Optional[Dict[str, Any]]:
    """Use Groq vision to scan a video frame and identify the webcam overlay.

    Samples a frame at 25% into the video, sends it to a Groq vision model,
    and asks it to locate the webcam overlay (if any). Returns a dict with:
      - has_webcam: bool
      - corner: str | None
      - bbox: (x, y, w, h) of the webcam region in source pixels

    Groq vision scans the ENTIRE frame and understands the full scene
    (gameplay vs. face cam), making it far more accurate than haar cascades.
    Returns None on any error so the caller can fall back to OpenCV.
    """
    try:
        import base64

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
            return None

        sample_time = duration * 0.25
        frame_path = str(Path(video_path).parent / "webcam_groq_probe.jpg")
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(sample_time), "-i", video_path,
             "-frames:v", "1", "-q:v", "2", frame_path],
            capture_output=True, text=True, timeout=30,
        )
        if not Path(frame_path).exists():
            return None

        # Get frame dimensions
        frame_w, frame_h = 1920, 1080
        try:
            import cv2 as _cv2
            _img = _cv2.imread(frame_path)
            if _img is not None:
                frame_h, frame_w = _img.shape[:2]
        except ImportError:
            try:
                vprobe = subprocess.run(
                    ["ffprobe", "-v", "quiet", "-print_format", "json",
                     "-show_streams", video_path],
                    capture_output=True, text=True, timeout=15,
                )
                if vprobe.returncode == 0:
                    for s in json.loads(vprobe.stdout).get("streams", []):
                        if s.get("codec_type") == "video":
                            frame_w = int(s.get("width", 1920))
                            frame_h = int(s.get("height", 1080))
                            break
            except Exception:
                pass

        # Encode frame as base64 data URL
        with open(frame_path, "rb") as fh:
            b64_data = base64.b64encode(fh.read()).decode("utf-8")
        Path(frame_path).unlink(missing_ok=True)

        # Send to Groq vision model with JSON mode
        client = _get_groq_client()
        prompt = (
            "You are analyzing a single frame from a Twitch stream or gameplay video. "
            "Identify whether there is a webcam / face-cam overlay (a small video feed "
            "showing a person talking, usually in a corner or along an edge, distinct "
            "from the main gameplay). Respond as a JSON object with these fields:\n"
            '  "has_webcam": true or false\n'
            '  "corner": one of "top-left", "top-right", "bottom-left", '
            '"bottom-right", "top-center", or null\n'
            '  "x_percent": left edge of the webcam overlay as percentage of frame width (0-100)\n'
            '  "y_percent": top edge of the webcam overlay as percentage of frame height (0-100)\n'
            '  "width_percent": width of the webcam overlay as percentage of frame width (0-100)\n'
            '  "height_percent": height of the webcam overlay as percentage of frame height (0-100)\n'
            "If has_webcam is false, set corner to null and all percentage fields to 0.\n"
            "Only output the JSON object, no other text."
        )

        completion = client.chat.completions.create(
            model=GROQ_VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_data}"},
                        },
                    ],
                }
            ],
            temperature=0.2,
            max_completion_tokens=256,
            response_format={"type": "json_object"},
            timeout=30,
        )

        raw = completion.choices[0].message.content.strip()
        print(f"[groq-vision] webcam detection response: {raw}")
        result = json.loads(raw)

        if not result.get("has_webcam", False):
            return {"has_webcam": False, "corner": None, "bbox": None}

        corner = result.get("corner")
        xp = float(result.get("x_percent", 0) or 0)
        yp = float(result.get("y_percent", 0) or 0)
        wp = float(result.get("width_percent", 0) or 0)
        hp = float(result.get("height_percent", 0) or 0)

        # Convert percentages to pixel bbox
        bx = int(frame_w * xp / 100)
        by = int(frame_h * yp / 100)
        bw = int(frame_w * wp / 100)
        bh = int(frame_h * hp / 100)

        # Clamp + sanity check
        bx = max(0, min(bx, frame_w - 1))
        by = max(0, min(by, frame_h - 1))
        bw = max(50, min(bw, frame_w - bx))
        bh = max(50, min(bh, frame_h - by))

        print(f"[groq-vision] webcam detected: corner={corner}, bbox=({bx},{by},{bw},{bh})")
        return {"has_webcam": True, "corner": corner, "bbox": (bx, by, bw, bh)}

    except Exception as e:
        print(f"[groq-vision] webcam detection error (falling back to OpenCV): {e}")
        return None


def _detect_webcam(video_path: str) -> Optional[Dict[str, Any]]:
    """Detect if a video has a webcam overlay and which corner it's in.

    Tries Groq vision first (scans the entire frame, understands the scene).
    Falls back to OpenCV haar cascade (corner face detection) if Groq is
    unavailable or fails. Returns a dict with:
      - has_webcam: bool
      - corner: 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right' | None
      - bbox: (x, y, w, h) of the webcam region in the source frame
    """
    # --- Groq vision (primary) ---
    groq_result = _detect_webcam_groq(video_path)
    if groq_result is not None:
        return groq_result

    # --- OpenCV haar cascade (fallback) ---
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
            return {"has_webcam": False, "corner": None, "bbox": None}

        sample_time = duration * 0.25
        frame_path = str(Path(video_path).parent / "webcam_probe.jpg")
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(sample_time), "-i", video_path,
             "-frames:v", "1", "-q:v", "2", frame_path],
            capture_output=True, text=True, timeout=30,
        )
        if not Path(frame_path).exists():
            return {"has_webcam": False, "corner": None, "bbox": None}

        # Use OpenCV to check for faces in each corner quadrant
        try:
            import cv2
            import numpy as np
        except ImportError:
            # No OpenCV — assume webcam top-right for Twitch VODs (common layout)
            Path(frame_path).unlink(missing_ok=True)
            return {"has_webcam": True, "corner": "top-right", "bbox": None}

        frame = cv2.imread(frame_path)
        Path(frame_path).unlink(missing_ok=True)
        if frame is None:
            return {"has_webcam": False, "corner": None, "bbox": None}

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cascade_path = str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            return {"has_webcam": True, "corner": "top-right", "bbox": None}

        # Define 4 corner quadrants (each is 40% of width/height from the corner)
        qw = int(w * 0.40)
        qh = int(h * 0.40)
        corners = {
            "top-left":     gray[0:qh, 0:qw],
            "top-right":    gray[0:qh, w - qw:w],
            "bottom-left":  gray[h - qh:h, 0:qw],
            "bottom-right": gray[h - qh:h, w - qw:w],
        }
        corner_offsets = {
            "top-left":     (0, 0),
            "top-right":    (0, w - qw),
            "bottom-left":  (h - qh, 0),
            "bottom-right": (h - qh, w - qw),
        }

        best_corner = None
        best_face_area = 0
        best_face_bbox = None

        for corner_name, corner_img in corners.items():
            faces = cascade.detectMultiScale(corner_img, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
            for (fx, fy, fw, fh) in faces:
                # Convert corner-local coords to frame-global coords
                ox, oy = corner_offsets[corner_name]
                gx, gy = fx + ox, fy + oy
                area = fw * fh
                if area > best_face_area:
                    best_face_area = area
                    best_corner = corner_name
                    best_face_bbox = (gx, gy, fw, fh)

        if best_corner and best_face_area > 0:
            # Expand the bbox a bit to include the full webcam frame, not just the face
            gx, gy, fw, fh = best_face_bbox
            pad_x = int(fw * 0.6)
            pad_y = int(fh * 0.8)
            x0 = max(0, gx - pad_x)
            y0 = max(0, gy - pad_y)
            x1 = min(w, gx + fw + pad_x)
            y1 = min(h, gy + fh + pad_y)
            webcam_bbox = (x0, y0, x1 - x0, y1 - y0)
            print(f"Webcam detected: corner={best_corner}, bbox={webcam_bbox}")
            return {"has_webcam": True, "corner": best_corner, "bbox": webcam_bbox}

        # No face found in any corner — also try the full top half (some streams
        # have the webcam centered top with a large overlay)
        top_half = gray[:h // 2, :]
        faces = cascade.detectMultiScale(top_half, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
        if len(faces) > 0:
            # Pick the largest face
            largest = max(faces, key=lambda f: f[2] * f[3])
            fx, fy, fw, fh = largest
            pad_x = int(fw * 0.6)
            pad_y = int(fh * 0.8)
            x0 = max(0, fx - pad_x)
            y0 = max(0, fy - pad_y)
            x1 = min(w, fx + fw + pad_x)
            y1 = min(h // 2, fy + fh + pad_y)
            webcam_bbox = (x0, y0, x1 - x0, y1 - y0)
            print(f"Webcam detected: corner=top-center, bbox={webcam_bbox}")
            return {"has_webcam": True, "corner": "top-left", "bbox": webcam_bbox}

        print("No webcam detected in any corner")
        return {"has_webcam": False, "corner": None, "bbox": None}

    except Exception as e:
        print(f"Webcam detection error: {e}")
        return {"has_webcam": False, "corner": None, "bbox": None}


# ---------------------------------------------------------------------------
# Caption style presets + ASS generation
# ---------------------------------------------------------------------------

# Caption style presets — maps a user-facing style name to ffmpeg
# subtitles force_style parameters. Colours are in ASS hex format
# (&HBBGGRR&, BGR byte order, NOT RGB).
CAPTION_STYLES = {
    "white": {
        "FontName": "Arial",
        "FontSize": 18,
        "PrimaryColour": "&HFFFFFF&",   # white
        "OutlineColour": "&H000000&",   # black
        "BorderStyle": 3,               # opaque box
        "Outline": 0,
        "Shadow": 0,
        "MarginV": 60,
        "Alignment": 2,
    },
    "typewriter": {
        "FontName": "Consolas",
        "FontSize": 20,
        "PrimaryColour": "&H00FF00&",   # green (BGR: 00FF00)
        "OutlineColour": "&H000000&",
        "BorderStyle": 1,
        "Outline": 1,
        "Shadow": 1,
        "MarginV": 60,
        "Alignment": 2,
    },
    "shake": {
        "FontName": "Arial Black",
        "FontSize": 26,
        "PrimaryColour": "&H5533FF&",   # red (BGR: FF3355)
        "OutlineColour": "&H000000&",
        "BorderStyle": 1,
        "Outline": 5,
        "Shadow": 0,
        "MarginV": 60,
        "Alignment": 2,
    },
    "rainbow": {
        "FontName": "Arial Black",
        "FontSize": 22,
        "PrimaryColour": "&HFFFFFF&",   # white (rainbow done via per-word color cycling)
        "OutlineColour": "&H000000&",
        "BorderStyle": 1,
        "Outline": 3,
        "Shadow": 0,
        "MarginV": 60,
        "Alignment": 2,
    },
    "outline-glow": {
        "FontName": "Arial Black",
        "FontSize": 24,
        "PrimaryColour": "&HFFFFFF&",   # white
        "OutlineColour": "&H7C55A8&",   # purple (BGR: A8557C)
        "BorderStyle": 1,
        "Outline": 3,
        "Shadow": 4,                    # thick glow
        "MarginV": 60,
        "Alignment": 2,
    },
}


def _caption_force_style(style_name: str) -> str:
    """Build the force_style string for ffmpeg subtitles filter."""
    preset = CAPTION_STYLES.get(style_name, CAPTION_STYLES["white"])
    parts = []
    for key, val in preset.items():
        if isinstance(val, str) and val.startswith("&H"):
            parts.append(f"{key}={val}")
        else:
            parts.append(f"{key}={val}")
    return ",".join(parts)


def _build_reframe_filter(
    has_webcam: bool,
    webcam_position: str = "top",
    background_position: str = "bottom",
    enable_captions: bool = False,
    srt_path: str = "",
    caption_style: str = "white",
    webcam_bbox: Optional[tuple] = None,
) -> str:
    """Build the ffmpeg filter_complex string for vertical 9:16 reframing.

    Layout (720x1280 vertical):
      - With webcam (bbox known): crop webcam region from source, scale to
        720x640 for the top; scale full source to 720x640 for the bottom.
      - With webcam (no bbox): split top/bottom halves of source.
      - Without webcam: full frame scaled to 720x1280.
      - Captions: burned into the frame using the selected style preset.

    webcam_bbox: (x, y, w, h) in source pixels for the webcam region.
    """

    W, H = 720, 1280
    half_h = H // 2  # 640

    if has_webcam:
        webcam_on_top = webcam_position not in ("bottom", "bottom-left", "bottom-right")
        if webcam_bbox and len(webcam_bbox) == 4:
            bx, by, bw, bh = webcam_bbox
            # Crop the webcam region from the source, scale to 720x640
            # Scale full source for the gameplay background, crop to 720x640
            filters = [
                # Webcam: crop the detected region -> scale to fill 720x640
                f"[0:v]crop={bw}:{bh}:{bx}:{by},scale={W}:{half_h}:force_original_aspect_ratio=increase,"
                f"crop={W}:{half_h}[webcam]",
                # Gameplay: scale full source to 720x640 (center crop)
                f"[0:v]scale={W}:{half_h}:force_original_aspect_ratio=increase,"
                f"crop={W}:{half_h}[bg]",
            ]
            if webcam_on_top:
                filters.append("[webcam][bg]vstack[stacked]")
            else:
                filters.append("[bg][webcam]vstack[stacked]")
        else:
            # No bbox — use PIP (picture-in-picture) overlay:
            # Full source scaled to 720x1280 as background, with a small
            # webcam crop (top-right corner, 25%w x 30%h) overlaid in the
            # top portion of the vertical video.
            pip_w = W // 3       # 240px wide PIP
            pip_h = half_h // 2  # 320px tall PIP
            # Position the PIP at top of the vertical frame
            pip_x = (W - pip_w) // 2  # centered horizontally
            pip_y = 20                 # near the top
            filters = [
                # Background: full source scaled to 720x1280
                f"[0:v]scale={W}:{H}:force_original_aspect_ratio=increase,"
                f"crop={W}:{H}[bg]",
                # PIP: crop top-right corner of source (where webcam usually is),
                # scale to PIP size
                f"[0:v]crop=iw//4:ih//3:iw-iw//4:0,scale={pip_w}:{pip_h}:force_original_aspect_ratio=increase,"
                f"crop={pip_w}:{pip_h}[pip]",
                # Overlay PIP onto background
                f"[bg][pip]overlay={pip_x}:{pip_y}[stacked]",
            ]
    else:
        filters = [
            f"[0:v]scale={W}:{H}:force_original_aspect_ratio=increase,"
            f"crop={W}:{H}[stacked]",
        ]

    # Burn captions if enabled and SRT/ASS file exists
    if enable_captions and srt_path and os.path.exists(srt_path):
        # Use ass= filter for .ass files (handles karaoke \kf tags properly),
        # subtitles= filter for .srt files.
        esc_path = srt_path.replace("\\", "/").replace(":", "\\:")
        if srt_path.endswith(".ass"):
            print(f"[clip] Burning captions from: {srt_path}")
            filters.append(f"[stacked]ass='{esc_path}'[out]")
        else:
            force_style = _caption_force_style(caption_style)
            filters.append(
                f"[stacked]subtitles='{esc_path}':force_style='{force_style}'[out]"
            )
    else:
        if enable_captions:
            print(f"[clip] No captions: enable={enable_captions}, path={srt_path}")
        filters.append("[stacked]null[out]")

    return ";".join(filters)


# ---------------------------------------------------------------------------
# Kinetic typography helpers (OpusClip-style animated captions)
# ---------------------------------------------------------------------------

# Emoji map: sentiment -> emoji. Injected after words that match a sentiment.
_SENTIMENT_EMOJI_POSITIVE = {
    "love": "\u2764\ufe0f", "amazing": "\u2728", "awesome": "\U0001f44f",
    "incredible": "\U0001f525", "insane": "\U0001f525", "crazy": "\U0001f92f",
    "wow": "\U0001f62e", "funny": "\U0001f602", "hilarious": "\U0001f602",
    "lol": "\U0001f602", "lmao": "\U0001f602", "lmfao": "\U0001f602",
    "best": "\U0001f44d", "perfect": "\U0001f44f", "epic": "\U0001f4aa",
    "legendary": "\U0001f3c6", "goat": "\U0001f3c6", "god": "\U0001f44d",
    "pog": "\U0001f525", "poggers": "\U0001f525", "hype": "\U0001f680",
    "cracked": "\U0001f525", "unreal": "\U0001f92f", "clutch": "\U0001f525",
    "winner": "\U0001f3c6", "victory": "\U0001f3c6", "champion": "\U0001f3c6",
}
_SENTIMENT_EMOJI_NEGATIVE = {
    "hate": "\U0001f621", "worst": "\U0001f4a9", "bad": "\U0001f44e",
    "awful": "\U0001f616", "terrible": "\U0001f616", "horrible": "\U0001f616",
    "trash": "\U0001f4a9", "garbage": "\U0001f4a9", "cringe": "\U0001f605",
    "fail": "\U0001f4a9", "ruined": "\U0001f622", "rigged": "\U0001f621",
    "bs": "\U0001f621", "bullshit": "\U0001f621", "rage": "\U0001f621",
    "angry": "\U0001f621", "furious": "\U0001f621", "scared": "\U0001f628",
    "sad": "\U0001f622", "crying": "\U0001f62d", "heartbreaking": "\U0001f62d",
}
_SENTIMENT_EMOJI_HYPE = {
    "let's go": "\U0001f680", "lets go": "\U0001f680",
    "no way": "\U0001f92f", "no shot": "\U0001f92f",
    "wtf": "\U0001f92f", "holy shit": "\U0001f92f",
    "oh my god": "\U0001f92f", "omg": "\U0001f92f", "omfg": "\U0001f92f",
    "unbelievable": "\U0001f92f", "world record": "\U0001f3c6",
    "play of the game": "\U0001f3c6", "potg": "\U0001f3c6",
    "headshot": "\U0001f4a5", "ace": "\U0001f525",
    "pentakill": "\U0001f525", "triple kill": "\U0001f525",
}

# Keyword highlight color (BGR ASS format) — bright yellow-green for keywords
KEYWORD_COLOR = "&H0000FFFF&"  # bright yellow
SHOUT_COLOR = "&H000000FF&"    # bright red for ALL-CAPS shouting


def _word_sentiment_emoji(word: str) -> str:
    """Return an emoji for a word based on sentiment, or empty string."""
    lowered = word.lower().strip("!?.,\"'")
    for key, emoji in _SENTIMENT_EMOJI_HYPE.items():
        if " " not in key and lowered == key:
            return emoji
    for key, emoji in _SENTIMENT_EMOJI_POSITIVE.items():
        if lowered == key:
            return emoji
    for key, emoji in _SENTIMENT_EMOJI_NEGATIVE.items():
        if lowered == key:
            return emoji
    return ""


def _is_hook_keyword_word(word: str) -> bool:
    """Check if a single word is a hook keyword (for colorization)."""
    lowered = word.lower().strip("!?.,\"'")
    for key in HOOK_KEYWORDS:
        if " " not in key and lowered == key:
            return True
    return False


def _is_shout_word(word: str) -> bool:
    """Detect ALL-CAPS shouting (3+ chars, all uppercase, has vowels)."""
    stripped = word.strip("!?.,\"'").strip()
    if len(stripped) < 3:
        return False
    if not stripped.isalpha():
        return False
    if stripped == stripped.upper() and any(c in stripped.lower() for c in "aeiou"):
        return True
    return False


def _generate_clip_srt(
    segments: List[Dict[str, Any]],
    start_time: float,
    end_time: float,
    segment_offset: float,
    job_id: str,
    caption_style: str = "white",
) -> str:
    """Generate an ASS subtitle file with OpusClip-style kinetic typography.

    Each word pops in with a scale + slide animation as it is spoken.
    Hook keywords are colorized (yellow), ALL-CAPS shout words get red +
    shake, and sentiment-matched emojis are injected after relevant words.
    Positioned center-screen. Rendered via the ffmpeg ass= filter.
    Returns the path to the .ass file.
    """
    ass_path = str(WORK_DIR / job_id / f"captions_{uuid.uuid4().hex[:8]}.ass")

    # Style configuration per preset
    # IMPORTANT: Use "DejaVu Sans" — available on Linux via fonts-dejavu-core.
    # "Arial"/"Arial Black"/"Consolas" are Windows-only fonts that don't exist
    # on the Render Linux container, causing ffmpeg to render colored boxes
    # instead of text.
    STYLE_CONFIG = {
        "white":         {"font": "DejaVu Sans",       "size": 48, "primary": "&H00FFFFFF&", "outline_c": "&H00000000&", "outline": 3, "shadow": 1, "bold": True},
        "yellow":        {"font": "DejaVu Sans",       "size": 52, "primary": "&H0000FFFF&", "outline_c": "&H00000000&", "outline": 4, "shadow": 0, "bold": True},
        "karaoke":       {"font": "DejaVu Sans",       "size": 56, "primary": "&H00FFFFFF&", "outline_c": "&H000000FF&", "outline": 4, "shadow": 1, "bold": True},
        "tiktok":        {"font": "DejaVu Sans",       "size": 64, "primary": "&H00FFFFFF&", "outline_c": "&H00000000&", "outline": 6, "shadow": 0, "bold": True},
        "minimal":       {"font": "DejaVu Sans",       "size": 36, "primary": "&H00DDDDDD&", "outline_c": "&H00000000&", "outline": 1, "shadow": 0, "bold": False},
        "neon-pop":      {"font": "DejaVu Sans",       "size": 56, "primary": "&H00FFFF00&", "outline_c": "&H00000000&", "outline": 3, "shadow": 4, "bold": True},
        "word-highlight":{"font": "DejaVu Sans",       "size": 52, "primary": "&H00FFFFFF&", "outline_c": "&H0000FFFF&", "outline": 5, "shadow": 0, "bold": True},
        "bouncy":        {"font": "DejaVu Sans",       "size": 56, "primary": "&H00FFFFFF&", "outline_c": "&H00000000&", "outline": 5, "shadow": 2, "bold": True},
        "gradient":      {"font": "DejaVu Sans",       "size": 60, "primary": "&H0000EDFF&", "outline_c": "&H00000000&", "outline": 4, "shadow": 0, "bold": True},
        "bold-box":      {"font": "DejaVu Sans",       "size": 52, "primary": "&H00FFFFFF&", "outline_c": "&H00000000&", "outline": 0, "shadow": 0, "bold": True},
        "typewriter":    {"font": "DejaVu Sans Mono",  "size": 48, "primary": "&H0000FF00&", "outline_c": "&H00000000&", "outline": 2, "shadow": 1, "bold": False},
        "shake":         {"font": "DejaVu Sans",       "size": 60, "primary": "&H005533FF&", "outline_c": "&H00000000&", "outline": 6, "shadow": 0, "bold": True},
        "rainbow":       {"font": "DejaVu Sans",       "size": 52, "primary": "&H00FFFFFF&", "outline_c": "&H00000000&", "outline": 4, "shadow": 0, "bold": True},
        "outline-glow":  {"font": "DejaVu Sans",       "size": 56, "primary": "&H00FFFFFF&", "outline_c": "&H007C55A8&", "outline": 4, "shadow": 5, "bold": True},
    }

    # Karaoke highlight color (the color words fill INTO as spoken)
    KARAOKE_COLORS = {
        "white": "&H0000FFFF&", "yellow": "&H0000FF00&", "karaoke": "&H000000FF&",
        "tiktok": "&H000000FF&", "minimal": "&H00FFFFFF&", "neon-pop": "&H00FF0000&",
        "word-highlight": "&H0000FFFF&", "bouncy": "&H000000FF&", "gradient": "&H000000FF&",
        "bold-box": "&H0000FFFF&", "typewriter": "&H00FFFFFF&", "shake": "&H00FFFFFF&",
        "rainbow": "&H000000FF&", "outline-glow": "&H0000FFFF&",
    }

    RAINBOW_COLORS = ["&H000000FF&", "&H0000A5FF&", "&H0000FFFF&", "&H0000FF00&", "&H00FF0000&", "&H00FF00A5&"]

    cfg = STYLE_CONFIG.get(caption_style, STYLE_CONFIG["white"])
    karaoke_color = KARAOKE_COLORS.get(caption_style, "&H0000FFFF&")

    # Build ASS header — Alignment 5 = middle-center (TikTok/OpusClip style)
    header_lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "PlayResX: 720",
        "PlayResY: 1280",
        "ScaledBorderAndShadow: yes",
        "WrapStyle: 2",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Default,{cfg['font']},{cfg['size']},{cfg['primary']},{karaoke_color},{cfg['outline_c']},{cfg['outline_c']},{'-1' if cfg['bold'] else '0'},0,0,0,100,100,0,0,1,{cfg['outline']},{cfg['shadow']},5,40,40,80,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    def fmt_ass_time(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        cs = int((seconds % 1) * 100)
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

    events = []

    for seg in segments:
        seg_start = float(seg.get("start", 0))
        seg_end = float(seg.get("end", 0))
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        if seg_end < start_time or seg_start > end_time:
            continue
        clamped_start = max(start_time, seg_start)
        clamped_end = min(end_time, seg_end)
        rel_start = clamped_start - start_time
        rel_end = clamped_end - start_time
        seg_duration = rel_end - rel_start
        if seg_duration < 0.1:
            continue

        words = text.split()
        if not words:
            continue

        total_chars = sum(len(w) for w in words)
        if total_chars == 0:
            total_chars = 1

        # --- OpusClip-style kinetic typography ---
        # Show 2-3 words at a time so context is visible, with the ACTIVE
        # word (the one being spoken) popping in the accent color + larger
        # scale. Keywords get a special highlight color. Shout words (ALL
        # CAPS) get extra emphasis + a shake. Emojis are injected after
        # sentiment-bearing words.
        WORDS_PER_LINE = 2
        groups = [words[i:i + WORDS_PER_LINE] for i in range(0, len(words), WORDS_PER_LINE)]

        # Per-group duration proportional to the group's total char count.
        group_weights = [max(1, sum(len(w) for w in g)) for g in groups]
        weight_sum = sum(group_weights)

        cursor = rel_start
        for gi, group in enumerate(groups):
            group_dur = seg_duration * group_weights[gi] / weight_sum
            group_start = cursor
            group_end = cursor + group_dur
            cursor = group_end
            group_start = max(rel_start, min(rel_start + seg_duration, group_start))
            group_end = max(rel_start, min(rel_start + seg_duration, group_end))
            if group_end - group_start < 0.08:
                continue

            group_chars = sum(len(w) for w in group)
            if group_chars == 0:
                group_chars = 1

            # Within the group, each word gets a sub-time-slice.
            word_durs = []
            for wi, word in enumerate(group):
                wd = max(0.12, (group_end - group_start) * len(word) / group_chars)
                word_durs.append(wd)

            word_starts = []
            t = group_start
            for wd in word_durs:
                word_starts.append(t)
                t += wd

            # Build the line: all words in the group are rendered, but each
            # word has its own animation that activates at its time slice.
            karaoke_parts = []
            for wi, word in enumerate(group):
                word_upper = word.upper()
                global_wi = gi * WORDS_PER_LINE + wi
                word_start_rel = word_starts[wi] - group_start
                word_dur_s = word_durs[wi]
                word_start_ms = int(word_start_rel * 1000)
                word_end_ms = int((word_start_rel + word_dur_s) * 1000)

                # --- Determine word styling ---
                is_keyword = _is_hook_keyword_word(word)
                is_shout = _is_shout_word(word)
                emoji = _word_sentiment_emoji(word)

                # Color: rainbow cycles, keyword=yellow, shout=red, else accent
                if caption_style == "rainbow":
                    word_color = RAINBOW_COLORS[global_wi % len(RAINBOW_COLORS)]
                elif is_shout:
                    word_color = SHOUT_COLOR
                elif is_keyword:
                    word_color = KEYWORD_COLOR
                else:
                    word_color = karaoke_color

                # Scale target: shout=130%, keyword=115%, normal=100%
                if is_shout:
                    active_scale = 130
                elif is_keyword:
                    active_scale = 115
                else:
                    active_scale = 108

                pop_in_ms = min(80, max(40, word_dur_s * 1000 * 0.3))
                settle_ms = pop_in_ms + 80

                tags = []
                # Base: start dimmed + small
                tags.append("\\alpha&H80&")
                tags.append(f"\\c{cfg['primary']}")
                tags.append("\\fscx80\\fscy80")

                # Slide in from bottom (12px offset -> 0)
                slide_px = 12

                # Animate to active state at word_start_ms
                if word_start_ms > 0:
                    tags.append(f"\\t(0,{word_start_ms},\\alpha&H80&\\fscx80\\fscy80)")
                # Pop in at word start
                tags.append(
                    f"\\t({word_start_ms},{word_start_ms + int(pop_in_ms)},"
                    f"\\alpha&H00&\\c{word_color}\\fscx{active_scale}\\fscy{active_scale})"
                )
                # Settle slightly after pop
                settle_scale = max(95, active_scale - 8)
                tags.append(
                    f"\\t({word_start_ms + int(pop_in_ms)},{word_start_ms + int(settle_ms)},"
                    f"\\fscx{settle_scale}\\fscy{settle_scale})"
                )
                # After the word: dim back
                if word_end_ms < int((group_end - group_start) * 1000):
                    tags.append(
                        f"\\t({word_end_ms},{word_end_ms + 60},"
                        f"\\alpha&HA0&\\fscx{settle_scale - 5}\\fscy{settle_scale - 5})"
                    )

                # Shout words get a shake effect during their active time
                if is_shout:
                    shake_start = word_start_ms
                    shake_end = word_start_ms + int(settle_ms)
                    tags.append(f"\\t({shake_start},{shake_start + 30},\\frx3)")
                    tags.append(f"\\t({shake_start + 30},{shake_start + 60},\\frx-3)")
                    tags.append(f"\\t({shake_start + 60},{shake_end},\\frx0)")

                # Fade in/out for the whole event
                tags.append("\\fad(50,40)")

                tag_str = "".join(tags)
                display_text = word_upper
                if emoji:
                    display_text = word_upper + " " + emoji
                karaoke_parts.append(f"{{{tag_str}}}{display_text}")

            line_text = " ".join(karaoke_parts)
            events.append(
                f"Dialogue: 0,{fmt_ass_time(group_start)},{fmt_ass_time(group_end)},Default,,0,0,0,,{line_text}"
            )

    if not events:
        return ""

    with open(ass_path, "w", encoding="utf-8") as f:
        f.write("\n".join(header_lines) + "\n" + "\n".join(events) + "\n")

    print(f"[clip] ASS file written: {ass_path} ({len(events)} events)")
    return ass_path


def _format_srt_time(seconds: float) -> str:
    """Format seconds as SRT timestamp: HH:MM:SS,mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ---------------------------------------------------------------------------
# Clip processing (cut, reframe, encode, upload)
# ---------------------------------------------------------------------------

def process_single_clip(
    video_path: str,
    start_time: float,
    end_time: float,
    clip_title: str = "Clip",
    has_webcam: Optional[bool] = None,
    webcam_position: str = "top",
    background_position: str = "bottom",
    enable_captions: bool = False,
    caption_style: str = "white",
    job_id: str = "",
    transcript_segments: Optional[List[Dict[str, Any]]] = None,
    segment_offset: float = 0.0,
    webcam_corner: Optional[str] = None,
) -> Dict[str, Any]:
    """Cut, reframe, and encode a single clip. Uploads to R2.

    If has_webcam is None, auto-detects webcam presence + corner from the video.
    webcam_position controls stacking order: 'top' = webcam on top,
    'bottom' = gameplay on top, webcam on bottom.
    webcam_corner can override the detected corner ('top-left', 'top-right',
    'bottom-left', 'bottom-right') for manual placement.
    If enable_captions and transcript_segments are provided, burns captions
    using the selected caption_style preset.
    """

    # Auto-detect webcam if not specified
    webcam_bbox = None
    if has_webcam is None:
        detection = _detect_webcam(video_path)
        has_webcam = detection.get("has_webcam", False)
        webcam_bbox = detection.get("bbox")
        detected_corner = detection.get("corner")
        print(f"Webcam auto-detected: has_webcam={has_webcam}, corner={detected_corner}, bbox={webcam_bbox}")
        # Always place the webcam at the TOP of the reframed vertical video,
        # regardless of which corner it was detected in the source. The
        # detected corner only tells us WHERE to crop from in the source.
        if has_webcam and detected_corner:
            webcam_position = "top"
    elif has_webcam and webcam_corner:
        # Manual webcam corner — estimate bbox from corner using source dimensions
        try:
            probe = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", video_path],
                capture_output=True, text=True, timeout=15,
            )
            if probe.returncode == 0:
                streams = json.loads(probe.stdout).get("streams", [])
                for s in streams:
                    if s.get("codec_type") == "video":
                        sw = int(s.get("width", 1920))
                        sh = int(s.get("height", 1080))
                        break
                # Estimate webcam region: 25% of width, 30% of height from corner
                ww = int(sw * 0.25)
                wh = int(sh * 0.30)
                corner_map = {
                    "top-left":     (0, 0),
                    "top-right":    (sw - ww, 0),
                    "bottom-left":  (0, sh - wh),
                    "bottom-right": (sw - ww, sh - wh),
                }
                if webcam_corner in corner_map:
                    bx, by = corner_map[webcam_corner]
                    webcam_bbox = (bx, by, ww, wh)
                    print(f"Manual webcam corner={webcam_corner}, bbox={webcam_bbox}")
        except Exception as e:
            print(f"Manual webcam bbox estimation error: {e}")

    clip_id = uuid.uuid4().hex[:8]
    output_dir = WORK_DIR / job_id / "clips"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir / f"clip_{clip_id}.mp4")
    duration = max(0.1, end_time - start_time)

    # Generate captions ASS file if enabled
    srt_path = ""
    if enable_captions and transcript_segments:
        # The downloaded segment starts at segment_offset in the VOD.
        # The clip starts at start_time within the downloaded segment.
        # So the clip's VOD time range is: (start_time + segment_offset) to (end_time + segment_offset)
        vod_start = start_time + segment_offset
        vod_end = end_time + segment_offset
        srt_path = _generate_clip_srt(
            transcript_segments, vod_start, vod_end, segment_offset, job_id,
            caption_style=caption_style,
        )
        print(f"[clip {job_id}] SRT generated: path={srt_path}, exists={os.path.exists(srt_path) if srt_path else 'N/A'}")
    elif enable_captions and not transcript_segments:
        print(f"[clip {job_id}] WARNING: captions enabled but no transcript segments - captions will NOT be burned in")

    print(f"[clip {job_id}] webcam: has_webcam={has_webcam}, position={webcam_position}, bbox={webcam_bbox}, corner={webcam_corner}")

    filter_str = _build_reframe_filter(
        has_webcam=has_webcam,
        webcam_position=webcam_position,
        background_position=background_position,
        enable_captions=enable_captions,
        srt_path=srt_path,
        caption_style=caption_style,
        webcam_bbox=webcam_bbox,
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(start_time),
        "-i", video_path,
        "-t", str(duration),
        "-filter_complex", filter_str,
        "-map", "[out]",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "28",
        "-pix_fmt", "yuv420p",
        "-r", "30",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr[-3000:]}")

    # Clean up SRT file
    if srt_path:
        try:
            os.unlink(srt_path)
        except OSError:
            pass

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

    # Generate a presigned URL (valid for 7 days) since the R2 bucket
    # is not publicly accessible.
    presigned_url = r2_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": r2_key},
        ExpiresIn=604800,  # 7 days
    )

    return {
        "clipId": clip_id,
        "r2Key": r2_key,
        "publicUrl": presigned_url,
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
    prompt: str = "",
) -> Dict[str, Any]:
    """End-to-end: detect moments -> download -> process all clips in parallel."""

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
        prompt=prompt,
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
    prompt: str = ""

class AnalyzeCommentsReq(BaseModel):
    transcript: str
    comments: str = ""
    prompt: str = ""

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
    prompt: str = ""


# ---------------------------------------------------------------------------
# API endpoints — synchronous
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "service": "atlas-clips"}

@app.post("/api/atlas-clips/analyze")
def analyze(req: AnalyzeReq):
    try:
        source_type = req.sourceType or _detect_source_type(req.videoUrl) if req.videoUrl else None
        result = detect_moments(
            transcript=req.transcript,
            video_duration=req.videoDuration,
            source_type=source_type,
            reframe_config=req.reframeConfig,
            fast_mode=req.fastMode,
            prompt=req.prompt,
        )
        return result
    except Exception as e:
        return JSONResponse({"error": "analyze_failed", "details": str(e)}, status_code=500)

@app.post("/api/atlas-clips/analyze-comments")
def analyze_comments_endpoint(req: AnalyzeCommentsReq):
    try:
        result = analyze_comments(transcript=req.transcript, comments=req.comments, prompt=req.prompt)
        return result
    except Exception as e:
        return JSONResponse({"error": "analyze_comments_failed", "details": str(e)}, status_code=500)

@app.post("/api/atlas-clips/process-clip")
def process_clip(req: ProcessClipReq):
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
def process_batch_endpoint(req: ProcessBatchReq):
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
def pipeline(req: PipelineReq):
    try:
        result = run_full_pipeline(
            source_url=req.sourceUrl,
            transcript=req.transcript,
            reframe_config=req.reframeConfig,
            fast_mode=req.fastMode,
            max_clips=req.maxClips,
            prompt=req.prompt,
        )
        return result
    except Exception as e:
        return JSONResponse({"error": "pipeline_failed", "details": str(e)}, status_code=500)


# ---------------------------------------------------------------------------
# SQLite job store (persistent across restarts)
# ---------------------------------------------------------------------------

DB_PATH = str(WORK_DIR / "jobs.db")

_JOB_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    source_url TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    progress INTEGER NOT NULL DEFAULT 0,
    clip_count INTEGER NOT NULL DEFAULT 0,
    clips_json TEXT DEFAULT '[]',
    analysis_json TEXT DEFAULT '',
    segments_json TEXT DEFAULT '[]',
    error TEXT DEFAULT '',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at DESC);
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
        conn = self._conn()
        conn.executescript(_JOB_SCHEMA)
        conn.commit()
        conn.close()

    def create(self, job_id: str, source_url: str) -> Dict[str, Any]:
        now = time.time()
        with self._lock:
            conn = self._conn()
            conn.execute(
                "INSERT INTO jobs (id, source_url, status, progress, clip_count, created_at, updated_at) VALUES (?, ?, 'pending', 0, 0, ?, ?)",
                (job_id, source_url, now, now),
            )
            conn.commit()
            conn.close()
        return self.get(job_id)

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        conn = self._conn()
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        conn.close()
        return self._row_to_dict(row) if row else None

    def update(self, job_id: str, fields: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.get(job_id):
            return None
        fields["updated_at"] = time.time()
        set_clause = ", ".join(f"{k} = ?" for k in fields.keys())
        values = list(fields.values())
        with self._lock:
            conn = self._conn()
            conn.execute(f"UPDATE jobs SET {set_clause} WHERE id = ?", (*values, job_id))
            conn.commit()
            conn.close()
        return self.get(job_id)

    def list(self) -> List[Dict[str, Any]]:
        conn = self._conn()
        rows = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()
        conn.close()
        return [self._row_to_dict(r) for r in rows]

    def delete(self, job_id: str) -> bool:
        with self._lock:
            conn = self._conn()
            cur = conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            conn.commit()
            deleted = cur.rowcount > 0
            conn.close()
        return deleted

    @staticmethod
    def _row_to_dict(row) -> Dict[str, Any]:
        d = dict(row)
        # Deserialize JSON columns
        for key in ("clips_json", "analysis_json", "segments_json"):
            if key in d and d[key]:
                try:
                    d[key.replace("_json", "")] = json.loads(d[key])
                except (json.JSONDecodeError, TypeError):
                    d[key.replace("_json", "")] = ""
            elif key in d:
                d[key.replace("_json", "")] = ""
            d.pop(key, None)
        return d


job_store = JobStore()


# ---------------------------------------------------------------------------
# Request models for async job endpoints
# ---------------------------------------------------------------------------

class CreateJobReq(BaseModel):
    sourceUrl: str
    transcript: str = ""
    reframeConfig: Optional[Dict] = None
    fastMode: bool = False
    maxClips: int = 10
    prompt: str = ""

class ReframeClipReq(BaseModel):
    clipIndex: int
    reframeConfig: Optional[Dict] = None

class ManualClipReq(BaseModel):
    startTime: float
    endTime: float
    title: Optional[str] = ""
    reframeConfig: Optional[Dict] = None

class ReframeBatchReq(BaseModel):
    clipIndices: List[int]
    reframeConfig: Optional[Dict] = None


# ---------------------------------------------------------------------------
# Background task functions
# ---------------------------------------------------------------------------

def _run_pipeline_background(
    job_id: str,
    source_url: str,
    transcript: str,
    reframe_config: Dict,
    fast_mode: bool,
    max_clips: int,
):
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
        job_store.update(job_id, {"status": "downloading", "progress": 0})

        # Step 1: Download audio
        audio_result = download_audio_only(url=source_url, job_id=job_id)
        vod_duration = audio_result.get("duration", 0)
        job_store.update(job_id, {"status": "transcribing", "progress": 15})

        # Step 2: Transcribe
        segments = transcribe_audio(
            audio_path=audio_result["audioPath"],
            duration=vod_duration,
        )

        # Clean up audio file
        try:
            os.unlink(audio_result["audioPath"])
        except OSError:
            pass

        # If no transcript provided, parse it from the transcription segments
        if not transcript.strip() and segments:
            lines = []
            cursor = 0.0
            for seg in segments:
                # Add time gaps as silence
                gap = seg["start"] - cursor
                if gap > 2.0:
                    lines.append("")
                lines.append(seg["text"])
                cursor = seg["end"]
            transcript = "\n".join(lines)

        job_store.update(job_id, {"status": "analyzing", "progress": 30})

        # Step 3: Detect moments with heuristic algorithm
        source_type = _detect_source_type(source_url)
        moments = detect_moments_heuristic(
            segments=segments,
            video_duration=max(vod_duration, 1.0),
        )
        clips = moments.get("clips", [])

        job_store.update(job_id, {"status": "titling", "progress": 70})

        # Step 4: Generate titles with Groq (optional, falls back to heuristic)
        if clips:
            clips = _generate_clip_titles_groq(clips, source_type or "")

        # Store segments for later use by reframe endpoints
        job_store.update(job_id, {
            "status": "completed",
            "progress": 100,
            "clip_count": len(clips),
            "analysis_json": json.dumps({
                "clips": clips,
                "analysisSummary": moments.get("analysisSummary", ""),
                "topRecommendation": moments.get("topRecommendation"),
            }),
            "segments_json": json.dumps(segments),
        })

    except Exception as e:
        job_store.update(job_id, {
            "status": "failed",
            "error": str(e),
            "progress": 0,
        })


def _run_single_clip_background(
    job_id: str,
    source_url: str,
    start_time: float,
    end_time: float,
    clip_title: str,
    reframe_config: Dict,
):
    """Download ONLY the clip segment, cut, reframe, upload to R2.

    Uses yt-dlp --download-sections to fetch just the timestamp range
    instead of downloading the entire VOD. This makes reframing a clip
    from a 3-hour VOD take ~30s instead of ~10min.
    """
    try:
        job_store.update(job_id, {"status": "downloading", "progress": 20})

        # Download just the clip segment
        download_result = download_clip_segment(
            url=source_url,
            job_id=job_id,
            start_time=start_time,
            end_time=end_time,
        )
        video_path = download_result["videoPath"]
        segment_offset = download_result.get("segmentOffset", 0.0)

        # The downloaded segment may have padding, so we need to find
        # the local start/end within the downloaded file
        local_start = max(0.0, start_time - segment_offset)
        local_end = min(download_result["duration"], end_time - segment_offset)

        job_store.update(job_id, {"status": "processing", "progress": 50})

        rc = reframe_config or {}
        # Auto-detect webcam unless explicitly set
        auto_detect = rc.get("autoDetectWebcam", True)
        has_webcam = rc.get("hasWebcam") if not auto_detect else None
        webcam_corner = rc.get("webcamCorner", "none")
        if webcam_corner == "none":
            webcam_corner = None

        # Find transcript segments for this clip from the parent job
        transcript_segments = None
        try:
            all_jobs = job_store.list()
            for j in all_jobs:
                jid = j.get("id", "")
                jsegs = j.get("segments", [])
                jurl = j.get("source_url", "")
                if jsegs and jurl == source_url:
                    print(f"[reframe {job_id}] Found transcript segments from job {jid}: {len(jsegs)} segments")
                    transcript_segments = jsegs
                    break
            if not transcript_segments:
                print(f"[reframe {job_id}] WARNING: No transcript segments found - captions will NOT be burned in")
        except Exception as e:
            print(f"[reframe {job_id}] Error fetching transcript segments: {e}")

        result = process_single_clip(
            video_path=video_path,
            start_time=local_start,
            end_time=local_end,
            clip_title=clip_title,
            has_webcam=has_webcam,
            webcam_position=rc.get("webcamPosition", "top"),
            background_position=rc.get("backgroundPosition", "bottom"),
            enable_captions=rc.get("enableCaptions", False),
            caption_style=rc.get("captionStyle", "white"),
            job_id=job_id,
            transcript_segments=transcript_segments,
            segment_offset=segment_offset,
            webcam_corner=webcam_corner,
        )

        # Clean up downloaded segment
        try:
            os.unlink(video_path)
        except OSError:
            pass

        job_store.update(job_id, {
            "status": "completed",
            "progress": 100,
            "clip_count": 1,
            "clips_json": json.dumps([result]),
        })

    except Exception as e:
        job_store.update(job_id, {
            "status": "failed",
            "error": str(e),
            "progress": 0,
        })


def _run_download_only_background(
    job_id: str,
    source_url: str,
    start_time: float,
    end_time: float,
    clip_title: str,
):
    """Download ONLY the clip segment and upload it as-is (no reframe, no captions).

    Uses download_clip_segment to fetch just the timestamp range, then
    uploads the raw clip to R2 with stream copy (no re-encoding).
    """
    try:
        job_store.update(job_id, {"status": "downloading", "progress": 30})

        download_result = download_clip_segment(
            url=source_url,
            job_id=job_id,
            start_time=start_time,
            end_time=end_time,
        )
        video_path = download_result["videoPath"]
        segment_offset = download_result.get("segmentOffset", 0.0)
        local_start = max(0.0, start_time - segment_offset)
        local_end = min(download_result["duration"], end_time - segment_offset)
        duration = local_end - local_start

        job_store.update(job_id, {"status": "processing", "progress": 70})

        clip_id = uuid.uuid4().hex[:8]
        output_dir = WORK_DIR / job_id / "clips"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"clip_{clip_id}.mp4")

        # Try stream copy first (fast, no re-encoding)
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(local_start),
            "-i", video_path,
            "-t", str(duration),
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            "-movflags", "+faststart",
            output_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if proc.returncode != 0:
            # Fallback: re-encode
            cmd_fallback = [
                "ffmpeg", "-y",
                "-ss", str(local_start),
                "-i", video_path,
                "-t", str(duration),
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
                "-c:a", "aac",
                "-movflags", "+faststart",
                output_path,
            ]
            proc2 = subprocess.run(cmd_fallback, capture_output=True, text=True, timeout=180)
            if proc2.returncode != 0:
                raise RuntimeError(f"ffmpeg cut failed: {proc2.stderr[-300:]}")

        # Upload to R2
        r2_client, bucket = _get_r2_client()
        r2_key = f"atlas-clips/{job_id}/clip_{clip_id}.mp4"

        file_size = os.path.getsize(output_path)
        with open(output_path, "rb") as f:
            r2_client.upload_fileobj(
                f, bucket, r2_key,
                ExtraArgs={"ContentType": "video/mp4"},
            )

        # Clean up
        try:
            os.remove(output_path)
            os.unlink(video_path)
        except OSError:
            pass

        presigned_url = r2_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": r2_key},
            ExpiresIn=604800,
        )

        result = {
            "clipId": clip_id,
            "r2Key": r2_key,
            "publicUrl": presigned_url,
            "title": clip_title,
            "startTime": start_time,
            "endTime": end_time,
            "duration": duration,
            "sizeBytes": file_size,
        }

        job_store.update(job_id, {
            "status": "completed",
            "progress": 100,
            "clip_count": 1,
            "clips_json": json.dumps([result]),
        })

    except Exception as e:
        job_store.update(job_id, {
            "status": "failed",
            "error": str(e),
            "progress": 0,
        })


# ---------------------------------------------------------------------------
# Async job endpoints
# ---------------------------------------------------------------------------

@app.post("/api/atlas-clips/jobs")
def create_job(req: CreateJobReq):
    job_id = str(uuid.uuid4())[:12]
    job_store.create(job_id, req.sourceUrl)
    thread = threading.Thread(
        target=_run_pipeline_background,
        args=(job_id, req.sourceUrl, req.transcript, req.reframeConfig or {}, req.fastMode, req.maxClips),
        daemon=True,
    )
    thread.start()
    return {"id": job_id, "status": "pending"}


@app.get("/api/atlas-clips/jobs/{job_id}")
def get_job(job_id: str):
    job = job_store.get(job_id)
    if not job:
        return JSONResponse({"error": "job_not_found"}, status_code=404)
    return {
        "id": job.get("id"),
        "status": job.get("status"),
        "progress": job.get("progress", 0),
        "clipCount": job.get("clip_count", 0),
        "clips": job.get("clips", []),
        "analysis": job.get("analysis", {}),
        "error": job.get("error", ""),
    }


@app.get("/api/atlas-clips/jobs")
def list_jobs():
    jobs = job_store.list()
    return {"jobs": jobs}


@app.delete("/api/atlas-clips/jobs/{job_id}")
def delete_job(job_id: str):
    deleted = job_store.delete(job_id)
    if not deleted:
        return JSONResponse({"error": "job_not_found"}, status_code=404)
    return {"id": job_id, "deleted": True}


@app.post("/api/atlas-clips/jobs/{job_id}/reframe")
def reframe_clip(job_id: str, req: ReframeClipReq):
    parent = job_store.get(job_id)
    if not parent:
        return JSONResponse({"error": "job_not_found"}, status_code=404)
    analysis = parent.get("analysis", {})
    clips = analysis.get("clips", [])
    if req.clipIndex < 0 or req.clipIndex >= len(clips):
        return JSONResponse({"error": "clip_index_out_of_range"}, status_code=400)
    clip = clips[req.clipIndex]
    new_job_id = str(uuid.uuid4())[:12]
    job_store.create(new_job_id, parent.get("source_url", ""))
    thread = threading.Thread(
        target=_run_single_clip_background,
        args=(new_job_id, parent["source_url"], clip["startTime"], clip["endTime"],
              clip.get("title", f"Clip {req.clipIndex + 1}"), req.reframeConfig or {}),
        daemon=True,
    )
    thread.start()
    return {"id": new_job_id, "status": "pending"}


@app.post("/api/atlas-clips/jobs/{job_id}/download")
def download_clip(job_id: str, req: ReframeClipReq):
    """Download a clip as-is (no reframe, no captions, no re-encoding).

    Cuts the clip segment from the source VOD with stream copy and
    uploads it to R2. Much faster than reframe since there's no encoding.
    """
    parent = job_store.get(job_id)
    if not parent:
        return JSONResponse({"error": "job_not_found"}, status_code=404)
    analysis = parent.get("analysis", {})
    clips = analysis.get("clips", [])
    if req.clipIndex < 0 or req.clipIndex >= len(clips):
        return JSONResponse({"error": "clip_index_out_of_range"}, status_code=400)
    clip = clips[req.clipIndex]
    new_job_id = str(uuid.uuid4())[:12]
    job_store.create(new_job_id, parent.get("source_url", ""))
    thread = threading.Thread(
        target=_run_download_only_background,
        args=(new_job_id, parent["source_url"], clip["startTime"], clip["endTime"],
              clip.get("title", f"Clip {req.clipIndex + 1}")),
        daemon=True,
    )
    thread.start()
    return {"id": new_job_id, "status": "pending"}


@app.post("/api/atlas-clips/jobs/{job_id}/manual")
def manual_clip(job_id: str, req: ManualClipReq):
    parent = job_store.get(job_id)
    if not parent:
        return JSONResponse({"error": "job_not_found"}, status_code=404)
    new_job_id = str(uuid.uuid4())[:12]
    job_store.create(new_job_id, parent.get("source_url", ""))
    thread = threading.Thread(
        target=_run_single_clip_background,
        args=(new_job_id, parent["source_url"], req.startTime, req.endTime,
              req.title or "Manual Clip", req.reframeConfig or {}),
        daemon=True,
    )
    thread.start()
    return {"id": new_job_id, "status": "pending"}


@app.post("/api/atlas-clips/jobs/{job_id}/reframe-batch")
def reframe_batch(job_id: str, req: ReframeBatchReq):
    parent = job_store.get(job_id)
    if not parent:
        return JSONResponse({"error": "job_not_found"}, status_code=404)
    analysis = parent.get("analysis", {})
    clips = analysis.get("clips", [])
    created = []
    threads = []
    for idx in req.clipIndices:
        if idx < 0 or idx >= len(clips):
            continue
        clip = clips[idx]
        new_job_id = str(uuid.uuid4())[:12]
        job_store.create(new_job_id, parent.get("source_url", ""))
        t = threading.Thread(
            target=_run_single_clip_background,
            args=(new_job_id, parent["source_url"], clip["startTime"], clip["endTime"],
                  clip.get("title", f"Clip {idx + 1}"), req.reframeConfig or {}),
            daemon=True,
        )
        t.start()
        threads.append(t)
        created.append({"id": new_job_id, "status": "pending", "clipIndex": idx})
    return {"jobs": created}


# ---------------------------------------------------------------------------
# Chat monitoring (trigger-based clip recording)
# ---------------------------------------------------------------------------

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
    username: str = ""


@app.post("/api/atlas-clips/chat/monitor")
def chat_monitor(req: ChatMonitorReq):
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
    return {"status": "monitoring", "channelId": req.channelId, "monitoring": True}


@app.post("/api/atlas-clips/chat/stop")
def chat_stop(req: ChatStopReq):
    if req.channelId:
        session = _chat_sessions.pop(req.channelId, {})
        clips_recorded = session.get("clipsRecorded", 0)
    else:
        clips_recorded = sum(s.get("clipsRecorded", 0) for s in _chat_sessions.values())
        _chat_sessions.clear()
    return {"status": "stopped", "channelId": req.channelId, "clipsRecorded": clips_recorded}


@app.post("/api/atlas-clips/chat/trigger")
def chat_trigger(req: ChatTriggerReq):
    session = _chat_sessions.get(req.channelId)
    if not session or not session.get("isMonitoring"):
        return {"status": "not_monitoring", "channelId": req.channelId}

    msg_lower = req.chatMessage.lower()
    matched = any(p.lower() in msg_lower for p in session.get("triggerPhrases", []))
    if not matched:
        return {"status": "no_match", "channelId": req.channelId}

    clip_duration = session.get("clipDuration", 30)
    now = req.timestamp or time.time()
    clip = {
        "startTime": max(0, now - clip_duration),
        "endTime": now,
        "title": f"Chat clip by {req.username}",
        "description": f'Triggered by: "{req.chatMessage}"',
        "viralScore": 50,
        "category": "chat_reaction",
        "transcript": "",
        "chatMessage": req.chatMessage,
        "triggeredBy": "chat",
    }
    session.setdefault("clips", []).append(clip)
    session["clipsRecorded"] = session.get("clipsRecorded", 0) + 1

    auto_record = session.get("autoRecord", False)
    return {
        "status": "clip_recorded",
        "channelId": req.channelId,
        "clip": clip,
        "autoRecorded": auto_record,
    }


@app.get("/api/atlas-clips/chat/status/{channel_id}")
def chat_status_channel(channel_id: str):
    session = _chat_sessions.get(channel_id)
    if not session:
        return {"status": "not_monitoring", "channelId": channel_id, "isMonitoring": False, "clips": [], "clipsRecorded": 0}
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
def chat_status_all():
    sessions = [
        {
            "channelId": s.get("channelId"),
            "isMonitoring": s.get("isMonitoring", False),
            "clipsRecorded": s.get("clipsRecorded", 0),
            "startTime": s.get("startTime", 0),
        }
        for s in _chat_sessions.values()
    ]
    return {"status": "ok", "activeChannels": len(_chat_sessions), "sessions": sessions}


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

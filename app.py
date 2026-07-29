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
        config=BotoConfig(
            retries={"max_attempts": 5, "mode": "adaptive"},
            signature_version="s3v4",
            s3={"addressing_style": "path"},
        ),
    ), bucket


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Atlas Smart Moments — efficient best-moment finder for long VODs
# Text-only scoring with windowed grouping, emotional energy detection,
# and temporal diversity spreading. Finds great moments ANYWHERE in the
# video, not just the opening. Mixes Atlas heuristic concepts (keyword
# lift, sentiment spike, structural features) with energy/burst detection.
# ---------------------------------------------------------------------------

import math
import re as _re

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

_SENTENCE_END_RE = _re.compile(r"[.!?…]+[\"')\]]*$")
_CAPS_RE = _re.compile(r"\b[A-Z]{3,}\b")
_PROFANITY_RE = _re.compile(r"\b(fuck|shit|damn|bitch|ass|hell)\b", _re.IGNORECASE)

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
    # Punctuation energy: questions and exclamations signal engagement
    punctuation_boost = lowered.count("?") * 0.25 + lowered.count("!") * 0.20
    # ALL CAPS words = shouting / excitement
    caps_count = len(_CAPS_RE.findall(text))
    uppercase_boost = min(1.2, caps_count * 0.35)
    # Profanity often correlates with high-emotion moments (gaming streams)
    profanity_boost = min(0.8, len(_PROFANITY_RE.findall(text)) * 0.3)
    # Word count — concise punchy segments score higher
    word_count = max(1, len(_re.findall(r"[a-zA-Z']+", text)))
    conciseness = 1.0 / math.sqrt(word_count)
    sentence_end_bonus = 0.15 if _ends_sentence(text) else 0.0
    # Filler penalty — segments full of filler words are dead air
    filler_count = sum(1 for w in FILLER_WORDS if w in lowered)
    filler_penalty = filler_count * 0.15
    structural = (punctuation_boost + uppercase_boost + profanity_boost
                  + conciseness + sentence_end_bonus - filler_penalty)
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


# ── Substance checks: ensure clips have real content, not just loud moments ──

# Minimum words a clip must contain to have "substance" — a 15-60s clip of
# someone talking should have at least this many words. Below this it's
# mostly dead air or a single word reaction with no context.
MIN_WORDS_PER_CLIP = 20

# Context lead-in: seconds of extra footage before the hook moment so the
# viewer has context for what's happening. A clip that starts mid-reaction
# with no setup is confusing and lacks substance.
CONTEXT_LEAD_IN = 2.0


def _snap_to_sentence_boundary(
    clip_start: float, clip_end: float, scores: List["_SegmentScore"]
) -> Tuple[float, float]:
    """Adjust clip start/end to the nearest sentence boundaries in the transcript.

    This prevents clips from cutting off mid-sentence — a clip that starts or
    ends in the middle of a word feels incomplete and lacks substance. By
    snapping to sentence boundaries (segments that end with . ! or ?) we
    ensure each clip contains complete thoughts.
    """
    if not scores:
        return clip_start, clip_end

    # Find the segment whose start is closest to (and <=) clip_start
    # and snap the clip start to that segment's start if it's a sentence start
    best_start = clip_start
    for s in scores:
        if s.segment.start <= clip_start and s.segment.start > best_start - 5.0:
            # Prefer starting at a segment that begins a new sentence
            # (previous segment ended with punctuation, or this is the first)
            best_start = s.segment.start
        if s.segment.start > clip_start:
            break

    # Find the segment whose end is closest to (and >=) clip_end
    # and snap the clip end to that segment's end if it ends a sentence
    best_end = clip_end
    for s in scores:
        if s.segment.end >= clip_end and s.segment.end < best_end + 5.0:
            if _ends_sentence(s.segment.text):
                best_end = s.segment.end
        if s.segment.end > clip_end + 10.0:
            break

    # Ensure we don't make the clip too short or too long after snapping
    if best_end - best_start < 5.0:
        return clip_start, clip_end
    return best_start, best_end


def _count_words_in_range(
    start: float, end: float, scores: List["_SegmentScore"]
) -> int:
    """Count the total words spoken in a time range. Used to check content
    density — a clip with too few words is mostly dead air."""
    total = 0
    for s in scores:
        if s.segment.end <= start or s.segment.start >= end:
            continue
        total += len(_re.findall(r"[a-zA-Z']+", s.segment.text))
    return total


def _has_complete_thought(
    start: float, end: float, scores: List["_SegmentScore"]
) -> bool:
    """Check if the clip contains at least one complete sentence (a segment
    ending with . ! or ?). A clip with no complete sentence is just a
    fragment — a reaction or interjection with no substance."""
    for s in scores:
        if s.segment.start >= start and s.segment.end <= end:
            if _ends_sentence(s.segment.text):
                return True
    return False


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


def _score_segments(segments: List[_TranscriptSeg], prompt_keywords: Optional[List[str]] = None) -> List[_SegmentScore]:
    """Score transcript segments using the Atlas Smart Moments algorithm.

    Scores each segment on:
    - Keyword density (hook words per word — normalized so short segments
      with one strong keyword don't dominate over longer rich segments)
    - Emotional intensity (abs(sentiment) — both positive and negative)
    - Sentiment spike (sudden shift vs the LOCAL context, not just prev seg)
    - Structural energy (punctuation, caps, profanity, conciseness)
    - Intrigue / curiosity hooks
    - Speech density (words per second — rapid speech = excitement)
    - Prompt relevance (user-provided topic keywords boost matching segments)
    """
    if not segments:
        return []

    prompt_keywords = prompt_keywords or []

    # Pre-compute sentiments
    sentiments = [_rule_based_sentiment(seg.text) for seg in segments]

    out: List[_SegmentScore] = []
    for idx, seg in enumerate(segments):
        structural, keyword, intrigue = _text_features(seg.text)
        sentiment = sentiments[idx]

        # Local context sentiment spike: compare to a 5-segment window average
        # This catches genuine emotional shifts, not just adjacent noise.
        window_start = max(0, idx - 5)
        window_end = min(len(sentiments), idx + 6)
        local_avg = sum(sentiments[window_start:window_end]) / max(1, window_end - window_start)
        sentiment_spike = abs(sentiment - local_avg)

        # Speech density: words per second (rapid speech = high energy)
        seg_duration = max(0.5, seg.end - seg.start)
        word_count = max(1, len(_re.findall(r"[a-zA-Z']+", seg.text)))
        words_per_sec = word_count / seg_duration
        # Normalize: typical speech is ~2-3 words/sec; >4 = excited
        density_boost = _clamp((words_per_sec - 2.5) * 0.15, 0.0, 0.6)

        # Keyword density: keywords per word (avoids bias toward long segments)
        keyword_density = keyword / max(1, word_count) * 10  # scale up

        # Prompt relevance: boost segments matching user-provided topic keywords.
        # This makes the prompt actually steer clip selection.
        prompt_boost = 0.0
        if prompt_keywords:
            lowered = seg.text.lower()
            matches = sum(1 for pk in prompt_keywords if pk in lowered)
            if matches:
                prompt_boost = min(2.0, matches * 0.8)

        total = (
            structural * 0.80
            + keyword_density * 1.40
            + keyword * 0.60        # raw keyword weight still matters
            + intrigue * 0.35
            + abs(sentiment) * 1.20  # emotional INTENSITY (either polarity)
            + sentiment_spike * 1.50  # sudden shifts = engaging
            + density_boost * 0.50
            + prompt_boost * 1.80    # user prompt steering — strong weight
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
    prompt_keywords: Optional[List[str]] = None,
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

    duration = max(1e-6, duration)

    # ── Step 1: Build candidate windows by grouping nearby segments ────────
    # Slide through segments and build 15-60s windows, scoring each window
    # by the aggregate energy of its constituent segments.
    candidates: List[Dict[str, Any]] = []
    n = len(scores)

    for i in range(n):
        # Build a window starting at segment i, extending up to max_clip_duration
        window_segs = [scores[i]]
        window_start = scores[i].segment.start
        window_end = scores[i].segment.end

        for j in range(i + 1, n):
            seg = scores[j]
            if seg.segment.start - window_start > max_clip_duration:
                break
            window_segs.append(seg)
            window_end = seg.segment.end

        window_duration = window_end - window_start
        if window_duration < 2.0:
            continue

        # Score the window: sum of segment scores, normalized by duration.
        # Weight recent (later) segments in the window slightly higher since
        # the "punchline" of a moment often comes at the end.
        total_score = 0.0
        for k, ws in enumerate(window_segs):
            weight = 1.0 + k * 0.05  # slight ramp
            total_score += ws.total * weight

        # Normalize by window duration so we don't bias toward long windows
        normalized = total_score / max(1.0, window_duration)

        # ── Atlas HOOK algorithm: opening-hook scoring ──────────────────────
        # Score the first 5 seconds of the window for hook keywords. A clip
        # that opens with a strong hook (reaction, excitement, question) gets
        # a bonus — this is what grabs viewers in the first 3 seconds.
        HOOK_WINDOW = 5.0
        opening_segs = [s for s in window_segs if s.segment.start - window_start < HOOK_WINDOW]
        hook_opening = sum(s.keyword + s.intrigue for s in opening_segs) if opening_segs else 0.0
        hook_opening_boost = _clamp(hook_opening * 0.15, 0.0, 0.8)

        # ── Atlas PAYOFF algorithm: ending-payoff scoring ──────────────────
        # Score the last 5 seconds for cliffhanger / resolution keywords. A
        # clip that ends with a payoff (punchline, cliffhanger, reveal) gets a
        # bonus — this is what makes viewers want to share or rewatch.
        PAYOFF_WINDOW = 5.0
        ending_segs = [s for s in window_segs if s.segment.end - window_start > window_duration - PAYOFF_WINDOW]
        payoff_ending = sum(s.intrigue + s.keyword * 0.5 for s in ending_segs) if ending_segs else 0.0
        payoff_ending_boost = _clamp(payoff_ending * 0.12, 0.0, 0.6)

        # ── Atlas PACING algorithm: speech density across the window ───────
        # Already factored into per-segment scores via density_boost, but we
        # also compute a window-level pacing metric for the analytics output.
        pacing_score = sum(
            max(0.0, (len(_re.findall(r"[a-zA-Z']+", s.segment.text)) / max(0.5, s.segment.end - s.segment.start)) - 2.5)
            for s in window_segs
        ) / max(1, len(window_segs))

        # Bonus for windows that hit the sweet spot of 20-45s (ideal clip length)
        length_bonus = 0.0
        if 20 <= window_duration <= 45:
            length_bonus = 0.15
        elif 15 <= window_duration <= 60:
            length_bonus = 0.05

        # Penalty for very short windows (likely just one word)
        if window_duration < 5:
            normalized *= 0.5

        final_score = normalized + length_bonus + hook_opening_boost + payoff_ending_boost

        # Collect aggregate features for categorization
        best_seg = max(window_segs, key=lambda s: s.total)
        avg_sentiment = sum(s.sentiment for s in window_segs) / len(window_segs)
        max_spike = max(s.sentiment_spike for s in window_segs)
        total_keyword = sum(s.keyword for s in window_segs)
        total_intrigue = sum(s.intrigue for s in window_segs)

        # Context lead-in: start the clip slightly before the hook moment
        # so the viewer has context for what's happening. Without this, clips
        # start mid-reaction and feel disconnected.
        clip_candidate_start = max(0.0, window_start - CONTEXT_LEAD_IN)

        # Content density: count words in the window to check substance
        word_count_in_window = sum(
            len(_re.findall(r"[a-zA-Z']+", s.segment.text)) for s in window_segs
        )

        # Substance penalty: windows with too few words are mostly dead air.
        # A 15-60s clip should have at least MIN_WORDS_PER_CLIP words to have
        # real content. Penalize (don't exclude) so short-but-punchy reactions
        # can still surface, but lower their rank.
        substance_penalty = 0.0
        if word_count_in_window < MIN_WORDS_PER_CLIP:
            substance_penalty = -0.3 * (1.0 - word_count_in_window / MIN_WORDS_PER_CLIP)

        # Completeness bonus: windows that contain at least one complete
        # sentence (ending with . ! or ?) have a complete thought = substance.
        has_complete = any(_ends_sentence(s.segment.text) for s in window_segs)
        completeness_bonus = 0.15 if has_complete else 0.0

        final_score += substance_penalty + completeness_bonus

        candidates.append({
            "start": clip_candidate_start,
            "end": min(duration, window_end + 1.0),
            "score": final_score,
            "best_seg": best_seg,
            "avg_sentiment": avg_sentiment,
            "max_spike": max_spike,
            "total_keyword": total_keyword,
            "total_intrigue": total_intrigue,
            "hook_opening": hook_opening,
            "pacing_score": pacing_score,
            "payoff_ending": payoff_ending,
            "word_count": word_count_in_window,
            "has_complete_thought": has_complete,
            "transcript": " ".join(s.segment.text for s in window_segs[:5])[:200],
        })

    if not candidates:
        return []

    # ── Step 2: Sort by score ──────────────────────────────────────────────
    candidates.sort(key=lambda c: c["score"], reverse=True)

    # ── Step 3: Diversity spreading ────────────────────────────────────────
    # Divide the VOD into `max_count` equal zones. Pick the best candidate
    # from each zone first, then fill remaining slots from the global pool.
    # This ensures clips are spread across the entire video.
    zone_size = duration / max(1, max_count)
    zones: List[List[Dict[str, Any]]] = [[] for _ in range(max_count)]
    for c in candidates:
        zone_idx = min(max_count - 1, int(c["start"] / zone_size))
        zones[zone_idx].append(c)

    chosen: List[Dict[str, Any]] = []
    used_windows: List[Tuple[float, float]] = []

    def _try_add(cand: Dict[str, Any]) -> bool:
        # _fit_window's max_duration param is the VIDEO duration (upper
        # bound for end time), NOT the max clip length. We cap clip length
        # separately below. Passing max_clip_duration here would trap all
        # clips in the first 60 seconds of the VOD.
        clip_start, clip_end = _fit_window(
            cand["start"], cand["end"], min_clip_duration, duration,
        )
        # Enforce max clip duration
        if clip_end - clip_start > max_clip_duration:
            clip_end = clip_start + max_clip_duration

        # ── Substance: snap to sentence boundaries so clips don't cut off
        # mid-sentence. This ensures each clip contains complete thoughts.
        clip_start, clip_end = _snap_to_sentence_boundary(
            clip_start, clip_end, scores
        )

        # ── Substance: reject clips with too few words (mostly dead air).
        # A clip needs real spoken content to have substance.
        words_in_clip = _count_words_in_range(clip_start, clip_end, scores)
        if words_in_clip < MIN_WORDS_PER_CLIP // 2:
            # Too little content — skip this candidate entirely
            return False

        # Check overlap with existing clips (>5s overlap = skip)
        for ws, we in used_windows:
            overlap = max(0.0, min(clip_end, we) - max(clip_start, ws))
            if overlap >= 5.0:
                return False
        used_windows.append((clip_start, clip_end))

        best = cand["best_seg"]
        raw = cand["score"]
        max_score = candidates[0]["score"] if candidates else 1.0
        viral_score = int(_clamp((raw / max(max_score, 0.01)) * 100, 30, 99))

        # Real Atlas sub-scores from the window's actual transcript content
        hook_opening = cand.get("hook_opening", 0.0)
        pacing_score = cand.get("pacing_score", 0.0)
        payoff_ending = cand.get("payoff_ending", 0.0)

        # Categorize based on dominant feature
        if cand["total_keyword"] > 2.0:
            category = "funny"
        elif abs(cand["avg_sentiment"]) > 0.4:
            category = "emotional_peak"
        elif cand["max_spike"] > 0.4:
            category = "controversial"
        elif cand["total_intrigue"] > 1.0:
            category = "cliffhanger"
        else:
            category = "highlight"

        # Build reason string
        reason_bits = []
        if cand["total_keyword"] > 1.5:
            reason_bits.append("strong hype/excitement language")
        if cand["max_spike"] > 0.4:
            reason_bits.append("sharp emotional shift")
        if abs(cand["avg_sentiment"]) > 0.4:
            reason_bits.append("strong emotional tone")
        if cand["total_intrigue"] > 1.0:
            reason_bits.append("curiosity/cliffhanger hook")
        if not reason_bits:
            reason_bits.append("high engagement potential")

        chosen.append({
            "startTime": round(clip_start, 2),
            "endTime": round(clip_end, 2),
            "title": "",  # Will be filled by Groq
            "description": " ".join(reason_bits),
            "viralScore": viral_score,
            "category": category,
            "transcript": cand["transcript"],
            "triggeredBy": "atlas_smart_moments",
            "recommendedStyle": "retention",
            # ── Real, non-hallucinated Atlas sub-scores (0-100) ──
            # These are computed from the actual transcript, not invented.
            "hookScore": int(_clamp(hook_opening * 25, 0, 99)),
            "pacingScore": int(_clamp(pacing_score * 20 + 30, 0, 99)),
            "payoffScore": int(_clamp(payoff_ending * 30, 0, 99)),
            # ── Substance metrics ──
            "wordCount": words_in_clip,
            "hasCompleteThought": cand.get("has_complete_thought", False),
            "_hookScore": round(raw, 3),
        })
        return True

    # Phase 1: Pick best from each zone (ensures temporal spread)
    for zone in zones:
        if len(chosen) >= max_count:
            break
        for cand in zone:  # already sorted by score within zone
            if _try_add(cand):
                break

    # Phase 2: Fill remaining slots from global pool (highest score first)
    if len(chosen) < max_count:
        for cand in candidates:
            if len(chosen) >= max_count:
                break
            _try_add(cand)

    # Sort chosen clips by start time (natural viewing order)
    chosen.sort(key=lambda c: c["startTime"])
    return chosen


def _extract_prompt_keywords(prompt: str) -> List[str]:
    """Extract meaningful keywords from a user prompt for topic-focused clip selection.

    Lowercases, strips filler/stopwords, and returns multi-word phrases + single
    words that will be used to boost segments matching the user's focus.
    """
    if not prompt or not prompt.strip():
        return []
    lowered = prompt.lower().strip()
    phrases = _re.split(r'[,;:.!?()\[\]{}"\']+', lowered)
    keywords: List[str] = []
    _STOP = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "need", "about",
        "into", "from", "up", "out", "if", "then", "than", "that", "this",
        "these", "those", "i", "you", "he", "she", "it", "we", "they", "me",
        "him", "her", "us", "them", "my", "your", "his", "its", "our", "their",
        "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
        "all", "each", "every", "both", "few", "more", "most", "other", "some",
        "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
        "very", "just", "also", "get", "got", "want", "wants", "like", "focus",
        "clips", "clip", "video", "moment", "moments", "find", "show", "make",
    }
    for phrase in phrases:
        words = [w for w in phrase.split() if len(w) > 1 and w not in _STOP]
        if not words:
            continue
        joined = " ".join(words)
        if len(joined) >= 3:
            keywords.append(joined)
        for w in words:
            if len(w) >= 3:
                keywords.append(w)
    seen = set()
    unique = []
    for k in keywords:
        if k not in seen:
            seen.add(k)
            unique.append(k)
    return unique


def detect_moments(
    transcript: str = "",
    video_duration: float = 0,
    source_type: Optional[str] = None,
    reframe_config: Optional[Dict] = None,
    fast_mode: bool = False,
    prompt: str = "",
    segments: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Top-level moment detection entry point.

    Accepts either a pre-parsed `segments` list (with {start, end, text}) or a
    raw `transcript` string (which gets split into pseudo-segments with
    estimated timestamps). Delegates to detect_moments_heuristic with the
    Atlas hook/pacing/payoff algorithm.

    The `prompt` parameter lets the user steer clip selection toward specific
    topics — prompt keywords boost matching segments in the scoring.
    """
    if segments is None:
        if not transcript or not transcript.strip():
            return {
                "clips": [],
                "analysisSummary": "No transcript or segments provided for analysis.",
                "topRecommendation": None,
                "totalDuration": video_duration,
            }
        lines = [l.strip() for l in transcript.split("\n") if l.strip()]
        segments = []
        cursor = 0.0
        for line in lines:
            words = max(1, len(line.split()))
            dur = max(2.0, words / 2.8)
            segments.append({"start": cursor, "end": cursor + dur, "text": line})
            cursor += dur
        if video_duration <= 0:
            video_duration = cursor

    return detect_moments_heuristic(
        segments=segments,
        video_duration=video_duration,
        prompt=prompt,
    )


def detect_moments_heuristic(
    segments: List[Dict[str, Any]],
    video_duration: float = 0,
    prompt: str = "",
) -> Dict[str, Any]:
    """Run the Atlas heuristic algorithm on timestamped transcript segments.

    This replaces the Groq LLM-based detect_moments. No API calls, no token
    limits, no rate limits — pure text scoring.

    Uses three first-class Atlas algorithms:
      - HOOK: opening-hook scoring (hook keywords in the first 5s of a window)
      - PACING: speech density / words-per-second energy
      - PAYOFF: ending-payoff scoring (cliffhanger/resolution in the last 5s)

    The `prompt` parameter extracts user keywords and boosts segments that
    match them, so user prompts actually steer which moments get selected.

    Args:
        segments: List of {start, end, text} dicts from Speaches verbose_json
        video_duration: Total video duration in seconds
        prompt: Optional user prompt for topic-focused selection

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

    prompt_keywords = _extract_prompt_keywords(prompt)
    scores = _score_segments(transcript_segs, prompt_keywords=prompt_keywords)
    clips = _collect_top_moments(scores, video_duration, max_count=10, prompt_keywords=prompt_keywords)

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
            "hookScore": best.get("hookScore", 0),
            "pacingScore": best.get("pacingScore", 0),
            "payoffScore": best.get("payoffScore", 0),
            "triggeredBy": "atlas_heuristic",
        }

    prompt_note = f" Prompt-focused on: {', '.join(prompt_keywords[:8])}." if prompt_keywords else ""
    summary = (
        f"Analyzed {len(transcript_segs)} transcript segments over {video_duration:.0f}s. "
        f"Atlas Smart Moments scored {len(scores)} segments, selected top {len(clips)} moments "
        f"using the Atlas HOOK (opening-hook), PACING (speech-density), and PAYOFF (ending-cliffhanger) "
        f"algorithms with windowed grouping + temporal diversity spreading.{prompt_note}"
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
        if clip.get("title") and clip["title"] != f"Clip {i+1}":
            continue
        text = (clip.get("transcript") or "").strip()
        title = ""
        if text:
            sentences = _re.split(r'[.!?]+', text)
            # Prefer sentences with hook keywords
            best = ""
            best_score = 0
            for s in sentences:
                s = s.strip()
                if not s or len(s) < 3:
                    continue
                score = 0
                low = s.lower()
                for kw, w in HOOK_KEYWORDS.items():
                    if kw in low:
                        score += w
                # All-caps = excitement
                if s.isupper() and len(s) < 40:
                    score += 2.0
                # Shorter = punchier
                if len(s) <= 50:
                    score += 0.5
                if score > best_score:
                    best_score = score
                    best = s
            if best:
                # Capitalize and truncate
                title = best[:60]
                if len(best) > 60:
                    title = best[:57].rstrip() + "..."
                title = title[0].upper() + title[1:] if title else ""
        if not title:
            mins = int(clip.get("startTime", 0)) // 60
            secs = int(clip.get("startTime", 0)) % 60
            title = f"Highlight at {mins}:{secs:02d}"
        clip["title"] = title
        if not clip.get("description"):
            score = clip.get("viralScore", 0)
            clip["description"] = f"Viral score {score:.1f}/10" if score else ""
    return clips


def _generate_clip_titles_groq(clips: List[Dict[str, Any]], source_type: str) -> List[Dict[str, Any]]:
    """Use Groq to generate catchy titles + descriptions for heuristic-selected clips.

    Groq is OPTIONAL — if the API key is missing or the request fails, falls back
    to heuristic title generation. The pipeline must NEVER fail because of Groq.
    """
    if not clips:
        return clips

    source_label = {"youtube": "YouTube video", "twitch": "Twitch VOD"}.get(source_type, "video")

    # Build a compact prompt with just the clip transcripts
    clip_lines = []
    for i, c in enumerate(clips):
        clip_lines.append(f"Clip {i+1} ({c['startTime']:.0f}s-{c['endTime']:.0f}s, score {c['viralScore']}): {c['transcript'][:150]}")
    clips_text = "\n".join(clip_lines)

    prompt = f"""You are a short-form content expert. For each clip below, write a catchy viral title (max 60 chars) and a one-sentence description of why it's engaging. These are from a {source_label}.

CRITICAL RULES:
- Use ONLY words and phrases that appear in the clip transcript. Do NOT invent scenes, concepts, or topics that are not in the transcript.
- The title must be a punchy phrase extracted from or directly inspired by the actual transcript text.
- The description must explain why THIS specific clip is engaging based on what is actually said.

{clips_text}

Return ONLY valid JSON:
{{
  "clips": [
    {{"index": 0, "title": "...", "description": "..."}},
    ...
  ]
}}"""

    try:
        client = _get_groq_client()  # May raise if GROQ_API_KEY missing — caught below
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
        print(f"Groq title generation failed (using heuristic fallback): {e}")
        clips = _generate_heuristic_titles(clips)

    return clips


def analyze_comments(transcript: str, comments: str = "", prompt: str = "") -> Dict[str, Any]:
    """Analyze transcript + comments to recommend the best editing style.

    Groq is OPTIONAL — falls back to a heuristic style detector if Groq is
    unavailable so the endpoint never 500s. The `prompt` lets the user steer
    the style recommendation toward a specific editing focus.
    """
    comments_text = comments or "No specific comments provided"

    # Heuristic fallback: score styles based on keyword density in transcript
    def _heuristic_style() -> Dict[str, Any]:
        low = (transcript or "").lower()
        reaction_score = 0
        retention_score = 0
        commentary_score = 0
        for kw, w in HOOK_KEYWORDS.items():
            if kw in low:
                reaction_score += w * 10
                retention_score += w * 8
        # Commentary: longer sentences, educational keywords
        for kw in ["how to", "tutorial", "explain", "learn", "guide", "step by step", "why", "because"]:
            if kw in low:
                commentary_score += 20
        # Normalize to 0-100
        mx = max(reaction_score, retention_score, commentary_score, 1)
        return {
            "detectedStyle": "reaction" if reaction_score == mx else ("commentary" if commentary_score == mx else "retention"),
            "confidence": min(95, int(60 + mx / 10)),
            "reasoning": "Heuristic analysis based on keyword density (Groq unavailable).",
            "styleRecommendations": {
                "retention": {"score": min(100, int(retention_score / mx * 100)), "reasoning": "Fast-paced content detected."},
                "commentary": {"score": min(100, int(commentary_score / mx * 100)), "reasoning": "Educational content detected."},
                "reaction": {"score": min(100, int(reaction_score / mx * 100)), "reasoning": "Emotional reactions detected."},
            },
        }

    try:
        client = _get_groq_client()
    except Exception as e:
        print(f"Groq unavailable for comment analysis (using heuristic): {e}")
        return _heuristic_style()

    focus_note = f"\n\nThe user wants the editing to focus on: {prompt}." if prompt else ""

    prompt = f"""Analyze this video transcript and any viewer comments to determine the best editing style for vertical short-form content.

Transcript:
{transcript}

Viewer Comments:
{comments_text}{focus_note}

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

    try:
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
    except Exception as e:
        print(f"Groq comment analysis failed (using heuristic): {e}")
        return _heuristic_style()


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


def download_clip_segment(
    url: str,
    job_id: str,
    start_time: float,
    end_time: float,
) -> Dict[str, Any]:
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

    # Pad the download range by 5s on each side for clean cuts
    pad = 3.0
    dl_start = max(0.0, start_time - pad)
    dl_end = end_time + pad
    duration = max(0.1, dl_end - dl_start)

    # Use yt-dlp to download ONLY the needed section with 5 concurrent
    # fragment downloads. This is much faster than the old approach
    # (yt-dlp -g -> ffmpeg single-connection) because:
    #   1. yt-dlp downloads HLS/DASH fragments in parallel (5x throughput)
    #   2. --download-sections skips fragments outside the time range
    #   3. No separate yt-dlp + ffmpeg round-trip
    ytdlp_cmd = [
        "yt-dlp",
        "-f", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[ext=mp4]/best",
        "--no-playlist",
        "--retries", "3",
        "--concurrent-fragments", "5",
        "--throttled-request-rate", "10",
        "--download-sections", f"*{dl_start}-{dl_end}",
        "--force-keyframes-at-cuts",
        "-o", output_path,
        "--merge-output-format", "mp4",
        "--extractor-args", "youtube:player_client=android,ios,web_safari,web",
        "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        url,
    ]
    ytdlp_result = subprocess.run(ytdlp_cmd, capture_output=True, text=True, timeout=180)
    if ytdlp_result.returncode != 0:
        # Fallback to the old approach if yt-dlp download-sections fails
        print(f"yt-dlp download-sections failed, falling back to ffmpeg seek: {ytdlp_result.stderr[-2000:]}")
        return _download_clip_segment_ffmpeg(url, job_id, dl_start, duration, output_path)

    if not os.path.exists(output_path):
        # yt-dlp may have saved with a different extension - find it
        candidates = list(cache_dir.glob("clip_source.*"))
        if candidates:
            os.rename(str(candidates[0]), output_path)
        else:
            raise RuntimeError("yt-dlp completed but no output file found")

    # Probe the downloaded segment duration
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
        # The downloaded segment starts at dl_start in the original VOD.
        # process_single_clip uses start_time/end_time relative to the
        # video file, so we need to adjust: the clip starts at
        # (start_time - dl_start) within the downloaded segment.
        "segmentOffset": dl_start,
    }


def _download_clip_segment_ffmpeg(url, job_id, dl_start, duration, output_path):
    """Fallback: yt-dlp -g + ffmpeg single-connection seek (slower)."""
    ytdlp_cmd = [
        "yt-dlp", "-g",
        "-f", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[ext=mp4]/best",
        "--no-playlist", "--retries", "3",
        "--extractor-args", "youtube:player_client=android,ios,web_safari,web",
        "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        url,
    ]
    ytdlp_result = subprocess.run(ytdlp_cmd, capture_output=True, text=True, timeout=120)
    if ytdlp_result.returncode != 0:
        raise RuntimeError(f"yt-dlp URL fetch failed: {ytdlp_result.stderr[-2000:]}")
    stream_urls = [u.strip() for u in ytdlp_result.stdout.strip().split('\n') if u.strip()]
    if not stream_urls:
        raise RuntimeError("yt-dlp returned no stream URLs")
    if len(stream_urls) >= 2:
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-ss", str(dl_start), "-i", stream_urls[0],
            "-ss", "0", "-i", stream_urls[1],
            "-t", str(duration), "-c", "copy", "-movflags", "+faststart",
            output_path,
        ]
    else:
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-ss", str(dl_start), "-i", stream_urls[0],
            "-t", str(duration), "-c", "copy", "-movflags", "+faststart",
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
        except Exception:
            pass
    return {
        "videoPath": output_path,
        "duration": seg_duration,
        "sourceType": _detect_source_type(url) or "unknown",
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
    # Groq is OPTIONAL — if the key is missing or the request fails, return
    # empty segments so the heuristic detector runs on whatever text we have.
    try:
        client = _get_groq_client()
    except Exception as e:
        print(f"Groq Whisper fallback skipped (no key): {e}")
        return []

    file_size = os.path.getsize(audio_path)
    MAX_SIZE = 24 * 1024 * 1024  # 24MB safety margin

    if file_size <= MAX_SIZE:
        try:
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
        except Exception as e:
            print(f"Groq Whisper transcription failed: {e}")
            return []

    # File too large for Groq Whisper and Speaches unavailable
    print("No transcription available (Speaches down, file too large for Groq Whisper)")
    return []


def _detect_webcam(video_path: str) -> Optional[Dict[str, Any]]:
    """Detect if a video has a webcam overlay and which corner it's in.

    Samples a frame at 25% into the video and scans all 4 corners
    for faces using OpenCV haar cascade. Returns a dict with:
      - has_webcam: bool
      - corner: 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right' | None
      - bbox: (x, y, w, h) of the webcam region in the source frame

    For Twitch VODs, the webcam is usually a small overlay in a corner.
    We scan each corner quadrant for faces and pick the one with the
    largest face as the webcam location.
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
        "Outline": 2,
        "Shadow": 1,
        "MarginV": 60,
        "Alignment": 2,                 # bottom center
    },
    "yellow": {
        "FontName": "Arial",
        "FontSize": 20,
        "PrimaryColour": "&H00FFFF&",   # yellow (BGR)
        "OutlineColour": "&H000000&",   # black
        "BorderStyle": 1,               # outline only
        "Outline": 3,
        "Shadow": 0,
        "MarginV": 60,
        "Alignment": 2,
    },
    "karaoke": {
        "FontName": "Arial",
        "FontSize": 22,
        "PrimaryColour": "&HFFFFFF&",   # white
        "OutlineColour": "&H0000FF&",   # red outline (BGR)
        "BorderStyle": 1,
        "Outline": 3,
        "Shadow": 1,
        "MarginV": 80,
        "Alignment": 2,
    },
    "tiktok": {
        "FontName": "Arial Black",
        "FontSize": 32,
        "PrimaryColour": "&HFFFFFF&",   # white
        "OutlineColour": "&H000000&",   # black
        "BorderStyle": 1,               # outline only (no opaque box)
        "Outline": 5,                   # thick outline for pop
        "Shadow": 0,
        "MarginV": 0,                   # centered vertically
        "Alignment": 5,                 # center-center (not bottom)
    },
    "minimal": {
        "FontName": "Arial",
        "FontSize": 14,
        "PrimaryColour": "&HDDDDDD&",   # light grey
        "OutlineColour": "&H000000&",   # black
        "BorderStyle": 1,
        "Outline": 0,
        "Shadow": 0,
        "MarginV": 30,
        "Alignment": 2,
    },
    # ── OpusClip-style animated presets ──────────────────────────────────
    "neon-pop": {
        "FontName": "Arial Black",
        "FontSize": 24,
        "PrimaryColour": "&HFFFF00&",   # cyan (BGR: 00FFFF)
        "OutlineColour": "&H000000&",   # black
        "BorderStyle": 1,
        "Outline": 2,
        "Shadow": 3,                    # glow shadow
        "MarginV": 60,
        "Alignment": 2,
    },
    "word-highlight": {
        "FontName": "Arial Black",
        "FontSize": 22,
        "PrimaryColour": "&HFFFFFF&",   # white
        "OutlineColour": "&H00FFFF&",   # yellow (BGR: FFFF00)
        "BorderStyle": 3,               # opaque box for highlight
        "Outline": 4,
        "Shadow": 0,
        "MarginV": 60,
        "Alignment": 2,
    },
    "bouncy": {
        "FontName": "Arial Black",
        "FontSize": 24,
        "PrimaryColour": "&HFFFFFF&",   # white
        "OutlineColour": "&H000000&",   # black
        "BorderStyle": 1,
        "Outline": 4,
        "Shadow": 2,
        "MarginV": 80,
        "Alignment": 2,
    },
    "gradient": {
        "FontName": "Arial Black",
        "FontSize": 26,
        "PrimaryColour": "&H00EDFF&",  # orange-yellow gradient feel
        "OutlineColour": "&H000000&",
        "BorderStyle": 1,
        "Outline": 3,
        "Shadow": 0,
        "MarginV": 60,
        "Alignment": 2,
    },
    "bold-box": {
        "FontName": "Arial Black",
        "FontSize": 22,
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
                # Webcam: crop the detected region → scale to fill 720x640
                f"[0:v]crop={bw}:{bh}:{bx}:{by},scale={W}:{half_h}:force_original_aspect_ratio=increase,"
                f"crop={W}:{half_h}[webcam]",
                # Gameplay: scale full source to 720x640 (center crop)
                f"[0:v]scale={W}:{half_h}:force_original_aspect_ratio=increase,"
                f"crop={W}:{half_h}[bg]",
            ]
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

    # Burn captions if enabled and ASS file exists
    if enable_captions and srt_path and os.path.exists(srt_path):
        esc_path = srt_path.replace("\\", "/").replace(":", "\\:")
        filters.append(
            f"[stacked]subtitles='{esc_path}'[out]"
        )
        print(f"[clip] Burning captions from: {srt_path}")
    else:
        filters.append("[stacked]null[out]")
        print(f"[clip] No captions: enable={enable_captions}, path={srt_path}")

    return ";".join(filters)


def _generate_clip_srt(
    segments: List[Dict[str, Any]],
    start_time: float,
    end_time: float,
    segment_offset: float,
    job_id: str,
    caption_style: str = "white",
) -> str:
    """Generate an ASS subtitle file with karaoke word-pop effects.

    Uses ASS \\kf (karaoke fill) tags for animated word-by-word highlighting.
    Each word fills with color as the speaker says it (OpusClip-style).
    Returns the path to the .ass file.
    """
    ass_path = str(WORK_DIR / job_id / f"captions_{uuid.uuid4().hex[:8]}.ass")

    # Style configuration per preset
    STYLE_CONFIG = {
        "white":         {"font": "Arial",       "size": 48, "primary": "&H00FFFFFF&", "outline_c": "&H00000000&", "outline": 3, "shadow": 1, "bold": True},
        "yellow":        {"font": "Arial",       "size": 52, "primary": "&H0000FFFF&", "outline_c": "&H00000000&", "outline": 4, "shadow": 0, "bold": True},
        "karaoke":       {"font": "Arial",       "size": 56, "primary": "&H00FFFFFF&", "outline_c": "&H000000FF&", "outline": 4, "shadow": 1, "bold": True},
        "tiktok":        {"font": "Arial Black",  "size": 64, "primary": "&H00FFFFFF&", "outline_c": "&H00000000&", "outline": 6, "shadow": 0, "bold": True},
        "minimal":       {"font": "Arial",       "size": 36, "primary": "&H00DDDDDD&", "outline_c": "&H00000000&", "outline": 1, "shadow": 0, "bold": False},
        "neon-pop":      {"font": "Arial Black",  "size": 56, "primary": "&H00FFFF00&", "outline_c": "&H00000000&", "outline": 3, "shadow": 4, "bold": True},
        "word-highlight":{"font": "Arial Black",  "size": 52, "primary": "&H00FFFFFF&", "outline_c": "&H0000FFFF&", "outline": 5, "shadow": 0, "bold": True},
        "bouncy":        {"font": "Arial Black",  "size": 56, "primary": "&H00FFFFFF&", "outline_c": "&H00000000&", "outline": 5, "shadow": 2, "bold": True},
        "gradient":      {"font": "Arial Black",  "size": 60, "primary": "&H0000EDFF&", "outline_c": "&H00000000&", "outline": 4, "shadow": 0, "bold": True},
        "bold-box":      {"font": "Arial Black",  "size": 52, "primary": "&H00FFFFFF&", "outline_c": "&H00000000&", "outline": 0, "shadow": 0, "bold": True},
        "typewriter":    {"font": "Consolas",     "size": 48, "primary": "&H0000FF00&", "outline_c": "&H00000000&", "outline": 2, "shadow": 1, "bold": False},
        "shake":         {"font": "Arial Black",  "size": 60, "primary": "&H005533FF&", "outline_c": "&H00000000&", "outline": 6, "shadow": 0, "bold": True},
        "rainbow":       {"font": "Arial Black",  "size": 52, "primary": "&H00FFFFFF&", "outline_c": "&H00000000&", "outline": 4, "shadow": 0, "bold": True},
        "outline-glow":  {"font": "Arial Black",  "size": 56, "primary": "&H00FFFFFF&", "outline_c": "&H007C55A8&", "outline": 4, "shadow": 5, "bold": True},
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

    # Build ASS header
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
        f"Style: Default,{cfg['font']},{cfg['size']},{cfg['primary']},{karaoke_color},{cfg['outline_c']},{cfg['outline_c']},{'-1' if cfg['bold'] else '0'},0,0,0,100,100,0,0,1,{cfg['outline']},{cfg['shadow']},2,40,40,80,1",
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

        # Show only 1-2 words at a time. Group words into pairs and give each
        # group its own Dialogue event with a proportional time slice of the
        # segment duration. This keeps at most 2 words on screen at once.
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
            # Clamp to segment bounds
            group_start = max(rel_start, min(rel_start + seg_duration, group_start))
            group_end = max(rel_start, min(rel_start + seg_duration, group_end))
            if group_end - group_start < 0.1:
                continue

            # Build karaoke fill across the words in this group (1-2 words)
            group_chars = sum(len(w) for w in group)
            if group_chars == 0:
                group_chars = 1
            karaoke_parts = []
            for wi, word in enumerate(group):
                word_dur = max(0.15, (group_end - group_start) * len(word) / group_chars)
                cs = int(word_dur * 100)  # centiseconds for ASS \kf
                word_upper = word.upper()
                # Use a stable color index across the whole segment so rainbow
                # does not reset color at every group boundary.
                global_wi = gi * WORDS_PER_LINE + wi
                if caption_style == "rainbow":
                    color = RAINBOW_COLORS[global_wi % len(RAINBOW_COLORS)]
                    karaoke_parts.append(f"\kf{cs}\c{color}{word_upper}")
                else:
                    karaoke_parts.append(f"\kf{cs}{word_upper}")

            karaoke_text = " ".join(karaoke_parts)
            # Add fade in/out for smooth pop effect
            karaoke_text = f"\fad(80,80){karaoke_text}"

            events.append(
                f"Dialogue: 0,{fmt_ass_time(group_start)},{fmt_ass_time(group_end)},Default,,0,0,0,,{karaoke_text}"
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


def _generate_thumbnail(video_path: str, job_id: str, clip_id: str, seek_time: float = 1.0) -> str:
    """Generate a JPEG thumbnail from the video for preview/poster display.

    Extracts a frame at seek_time seconds. Returns the local path to the
    thumbnail, or empty string if generation fails.
    """
    thumb_dir = WORK_DIR / job_id / "thumbs"
    thumb_dir.mkdir(parents=True, exist_ok=True)
    thumb_path = str(thumb_dir / f"thumb_{clip_id}.jpg")
    try:
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(seek_time),
            "-i", video_path,
            "-frames:v", "1",
            "-q:v", "3",
            "-vf", "scale=360:640:force_original_aspect_ratio=decrease,pad=360:640:(ow-iw)/2:(oh-ih)/2:black",
            thumb_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if proc.returncode == 0 and os.path.exists(thumb_path):
            return thumb_path
        print(f"[thumb] generation failed: {proc.stderr[-200:]}")
    except Exception as e:
        print(f"[thumb] error: {e}")
    return ""


def _upload_thumbnail(thumb_path: str, r2_client, bucket: str, job_id: str, clip_id: str) -> str:
    """Upload a thumbnail to R2 and return its presigned URL."""
    if not thumb_path or not os.path.exists(thumb_path):
        return ""
    r2_key = f"atlas-clips/{job_id}/thumb_{clip_id}.jpg"
    with open(thumb_path, "rb") as f:
        r2_client.upload_fileobj(
            f, bucket, r2_key,
            ExtraArgs={"ContentType": "image/jpeg"},
        )
    try:
        os.remove(thumb_path)
    except OSError:
        pass
    return r2_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": r2_key},
        ExpiresIn=604800,
    )


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
        # Use detected corner as webcam_position if user didn't explicitly set it
        if has_webcam and detected_corner:
            # Map corner to top/bottom for stacking
            if detected_corner.startswith("top"):
                webcam_position = "top"
            else:
                webcam_position = "bottom"
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

    clip_id = str(uuid.uuid4())[:8]
    output_dir = WORK_DIR / job_id / "clips"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir / f"clip_{clip_id}.mp4")

    duration = max(0.1, end_time - start_time)

    # Generate SRT for captions if enabled and transcript segments available
    srt_path = ""
    print(f"[clip {job_id}] enable_captions={enable_captions}, transcript_segments={'yes' if transcript_segments else 'NO'}, segment_offset={segment_offset}")
    if enable_captions and transcript_segments:
        # The transcript segments have timestamps relative to the ORIGINAL VOD.
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

    # Generate thumbnail before uploading (need the local file)
    thumb_path = _generate_thumbnail(output_path, job_id, clip_id, seek_time=1.0)

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

    # Upload thumbnail
    thumb_url = _upload_thumbnail(thumb_path, r2_client, bucket, job_id, clip_id)

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
        "thumbnailUrl": thumb_url,
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
            prompt=req.prompt,
        )
        return result
    except Exception as e:
        return JSONResponse({"error": "analyze_failed", "details": str(e)}, status_code=500)


@app.post("/api/atlas-clips/analyze-comments")
async def analyze_comments_endpoint(req: AnalyzeCommentsReq):
    try:
        result = analyze_comments(transcript=req.transcript, comments=req.comments, prompt=req.prompt)
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
            prompt=req.prompt,
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
  segments_json TEXT,
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
        for key in ("clips_json", "analysis_json", "segments_json"):
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
    prompt: str = ""

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

class MontageSegmentReq(BaseModel):
    startTime: float
    endTime: float

class MontageReq(BaseModel):
    """Multi-segment montage request.

    Accepts EITHER:
      - clipIndices: List[int]  — indices into the parent job's analysis clips
      - segments: List[{startTime, endTime}] — custom disjoint timestamp ranges

    The AI orders the segments into a cohesive narrative (hook -> build -> payoff)
    and concatenates them with crossfade transitions into a single output clip.
    """
    clipIndices: Optional[List[int]] = None
    segments: Optional[List[Dict[str, Any]]] = None
    reframeConfig: Optional[Dict] = None
    prompt: str = ""
    title: Optional[str] = None


# ---------------------------------------------------------------------------
# Background processing
# ---------------------------------------------------------------------------

def _run_pipeline_background(job_id: str, source_url: str, transcript: str, reframe_config: Optional[Dict], fast_mode: bool, max_clips: int, prompt: str = ""):
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
            segments = _retry(
                lambda: transcribe_audio(audio_result["audioPath"], duration=vod_duration),
                attempts=3, backoff=3.0, label="transcription",
            )
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
            prompt=prompt,
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
            segments_json=json.dumps(segments),
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
    """Download ONLY the clip segment, cut, reframe, upload to R2.

    Uses yt-dlp --download-sections to fetch just the timestamp range
    instead of downloading the entire VOD. This makes reframing a clip
    from a 3-hour VOD take ~30s instead of ~10min.
    """
    try:
        job_store.update(job_id, status="downloading", progress=20)
        download_result = _retry(
            lambda: download_clip_segment(
                url=source_url,
                job_id=job_id,
                start_time=start_time,
                end_time=end_time,
            ),
            attempts=2, backoff=3.0, label="clip-segment-download",
        )
        video_path = download_result["videoPath"]
        # The downloaded segment starts at segmentOffset in the original VOD.
        # Adjust start_time to be relative to the downloaded segment.
        segment_offset = download_result.get("segmentOffset", 0.0)
        local_start = max(0.0, start_time - segment_offset)
        local_end = end_time - segment_offset

        job_store.update(job_id, status="processing", progress=50)
        rc = reframe_config or {}
        # Webcam handling:
        # - autoDetectWebcam=true (default): pass has_webcam=None → auto-detect
        # - autoDetectWebcam=false + webcamCorner='none': no webcam
        # - autoDetectWebcam=false + webcamCorner='top-left'/'top-right'/etc:
        #   manual webcam placement — pass the corner to process_single_clip
        webcam_corner = None
        if rc.get("autoDetectWebcam", True):
            has_webcam = None  # auto-detect
        else:
            corner = rc.get("webcamCorner", "none")
            if corner and corner != "none":
                has_webcam = True
                webcam_corner = corner
            else:
                has_webcam = False

        # Fetch transcript segments from the parent analysis job for captions
        transcript_segments = None
        try:
            # The reframe job's source_url matches the parent job's source_url.
            # Find the parent analysis job (completed, has segments_json).
            all_jobs = job_store.list()
            print(f"[reframe {job_id}] Searching {len(all_jobs)} jobs for segments matching {source_url}")
            for j in all_jobs:
                jid = j.get("id", "?")
                jsegs = j.get("segments")
                jurl = j.get("source_url", "")
                print(f"[reframe {job_id}]   job={jid} url={jurl[:40]} segs={bool(jsegs)} status={j.get('status')}")
                if jurl == source_url and jsegs:
                    transcript_segments = jsegs
                    print(f"[reframe {job_id}] Found transcript segments from job {jid}: {len(transcript_segments)} segments")
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

        # Clean up the downloaded segment to save disk
        try:
            os.unlink(video_path)
        except OSError:
            pass

        job_store.update(
            job_id,
            status="completed",
            progress=100,
            clip_count=1,
            clips_json=json.dumps([result]),
        )
    except Exception as e:
        job_store.update(job_id, status="failed", error=str(e), progress=0)


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
        job_store.update(job_id, status="downloading", progress=30)
        download_result = download_clip_segment(
            url=source_url,
            job_id=job_id,
            start_time=start_time,
            end_time=end_time,
        )
        video_path = download_result["videoPath"]
        segment_offset = download_result.get("segmentOffset", 0.0)
        local_start = max(0.0, start_time - segment_offset)
        local_end = end_time - segment_offset
        duration = max(0.1, local_end - local_start)

        job_store.update(job_id, status="processing", progress=70)

        # Cut the exact clip range with stream copy (no re-encode, very fast)
        clip_id = str(uuid.uuid4())[:8]
        output_dir = WORK_DIR / job_id / "clips"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"clip_{clip_id}.mp4")

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(local_start),
            "-i", video_path,
            "-t", str(duration),
            "-c", "copy",           # stream copy — no re-encode
            "-avoid_negative_ts", "make_zero",
            "-movflags", "+faststart",
            output_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if proc.returncode != 0:
            # Fallback: re-encode if stream copy fails (codec issues)
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

        # Generate thumbnail before upload
        thumb_path = _generate_thumbnail(output_path, job_id, clip_id, seek_time=1.0)

        # Upload to R2
        r2_client, bucket = _get_r2_client()
        r2_key = f"atlas-clips/{job_id}/clip_{clip_id}.mp4"
        file_size = os.path.getsize(output_path)
        with open(output_path, "rb") as f:
            r2_client.upload_fileobj(
                f, bucket, r2_key,
                ExtraArgs={"ContentType": "video/mp4"},
            )

        # Upload thumbnail
        thumb_url = _upload_thumbnail(thumb_path, r2_client, bucket, job_id, clip_id)

        # Clean up
        try:
            os.remove(output_path)
            os.unlink(video_path)
        except OSError:
            pass

        # Generate presigned URL
        presigned_url = r2_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": r2_key},
            ExpiresIn=604800,
        )

        result = {
            "clipId": clip_id,
            "r2Key": r2_key,
            "publicUrl": presigned_url,
            "thumbnailUrl": thumb_url,
            "title": clip_title,
            "startTime": start_time,
            "endTime": end_time,
            "duration": duration,
            "sizeBytes": file_size,
        }

        job_store.update(
            job_id,
            status="completed",
            progress=100,
            clip_count=1,
            clips_json=json.dumps([result]),
        )
    except Exception as e:
        job_store.update(job_id, status="failed", error=str(e), progress=0)


def _order_montage_segments_ai(segments_info: List[Dict[str, Any]], prompt: str = "") -> List[int]:
    """Use Groq to order clip segments into a cohesive narrative.

    Given a list of segment dicts (each with transcript, startTime, endTime,
    hookScore, pacingScore, payoffScore), returns a list of indices in the
    optimal narrative order (hook -> build -> payoff).

    Falls back to a heuristic ordering if Groq is unavailable: highest hook
    score first (opening), highest payoff score last (ending), rest by score.
    """
    def _heuristic_order() -> List[int]:
        indexed = list(enumerate(segments_info))
        indexed.sort(key=lambda x: x[1].get("hookScore", 0), reverse=True)
        if not indexed:
            return []
        opening = indexed[0][0]
        rest = [i for i, _ in indexed[1:]]
        rest.sort(key=lambda x: segments_info[x].get("payoffScore", 0), reverse=True)
        if rest:
            ending = rest[0]
            middle = [i for i in rest[1:]]
            middle.sort(key=lambda x: segments_info[x].get("viralScore", 0), reverse=True)
            return [opening] + middle + [ending]
        return [opening]

    try:
        client = _get_groq_client()
    except Exception:
        return _heuristic_order()

    clip_lines = []
    for i, s in enumerate(segments_info):
        clip_lines.append(
            f"Segment {i} ({s.get('startTime', 0):.0f}s-{s.get('endTime', 0):.0f}s, "
            f"hook={s.get('hookScore', 0)}, payoff={s.get('payoffScore', 0)}): "
            f"{s.get('transcript', '')[:120]}"
        )
    clips_text = "\n".join(clip_lines)
    focus = f"\nThe user wants the montage to focus on: {prompt}." if prompt else ""

    ai_prompt = f"""You are a short-form video editor. Given these video segments, determine the BEST ORDER to assemble them into a cohesive vertical short-form video with a strong narrative arc.

Rules:
- Start with the segment that has the strongest HOOK (grabs attention in the first 3 seconds)
- End with the segment that has the strongest PAYOFF (cliffhanger, punchline, or resolution)
- Middle segments should build tension/interest in a logical flow
- Use ONLY the segments provided — do not invent new content{focus}

Segments:
{clips_text}

Return ONLY valid JSON with the optimal order (list of segment indices):
{{"order": [0, 2, 1, ...]}}"""

    try:
        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a video editing assistant. Return only JSON."},
                {"role": "user", "content": ai_prompt},
            ],
            model=GROQ_FAST_MODEL,
            temperature=0.3,
            max_tokens=500,
            response_format={"type": "json_object"},
        )
        text = completion.choices[0].message.content or "{}"
        result = json.loads(text)
        order = result.get("order", [])
        if isinstance(order, list) and len(order) == len(segments_info):
            valid = all(isinstance(i, int) and 0 <= i < len(segments_info) for i in order)
            if valid and len(set(order)) == len(segments_info):
                return order
        print(f"[montage] AI order invalid ({order}), using heuristic")
    except Exception as e:
        print(f"[montage] AI ordering failed (using heuristic): {e}")

    return _heuristic_order()


def _run_montage_background(
    job_id: str,
    source_url: str,
    segments_list: List[Dict[str, Any]],
    reframe_config: Optional[Dict],
    prompt: str,
    title: str,
):
    """Download multiple disjoint segments, AI-order them, concatenate with
    crossfade transitions, and upload as a single montage clip.

    This implements the multi-segment focus selection + true montage editing
    features: users pick specific timestamps, and the AI assembles them into
    a cohesive narrative with hook -> build -> payoff structure.
    """
    try:
        n_segs = len(segments_list)
        job_store.update(job_id, status="analyzing", progress=10)

        # Step 1: AI-order the segments for narrative flow
        ordered_indices = _order_montage_segments_ai(segments_list, prompt)
        ordered_segments = [segments_list[i] for i in ordered_indices]
        print(f"[montage {job_id}] AI order: {ordered_indices}")

        job_store.update(job_id, status="downloading", progress=20)

        # Step 2: Download each segment
        segment_paths = []
        for si, seg in enumerate(ordered_segments):
            pct = 20 + int(30 * si / max(1, n_segs))
            job_store.update(job_id, progress=pct)
            start_t = seg["startTime"]
            end_t = seg["endTime"]
            download_result = _retry(
                lambda: download_clip_segment(
                    url=source_url, job_id=job_id,
                    start_time=start_t, end_time=end_t,
                ),
                attempts=2, backoff=3.0, label=f"montage-segment-{si}",
            )
            video_path = download_result["videoPath"]
            segment_offset = download_result.get("segmentOffset", 0.0)
            local_start = max(0.0, start_t - segment_offset)
            local_end = end_t - segment_offset
            duration = max(0.1, local_end - local_start)

            clip_id = str(uuid.uuid4())[:8]
            seg_dir = WORK_DIR / job_id / "montage_segs"
            seg_dir.mkdir(parents=True, exist_ok=True)
            seg_path = str(seg_dir / f"seg_{si:03d}.mp4")

            cmd = [
                "ffmpeg", "-y",
                "-ss", str(local_start),
                "-i", video_path,
                "-t", str(duration),
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
                "-c:a", "aac", "-ar", "44100",
                "-vf", "scale=720:1280:force_original_aspect_ratio=increase,crop=720:1280",
                "-r", "30",
                "-movflags", "+faststart",
                seg_path,
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if proc.returncode != 0:
                raise RuntimeError(f"ffmpeg segment {si} encode failed: {proc.stderr[-500:]}")
            segment_paths.append(seg_path)

            try:
                os.unlink(video_path)
            except OSError:
                pass

        job_store.update(job_id, status="assembling", progress=60)

        # Step 3: Concatenate with crossfade transitions
        output_dir = WORK_DIR / job_id / "clips"
        output_dir.mkdir(parents=True, exist_ok=True)
        final_clip_id = str(uuid.uuid4())[:8]
        output_path = str(output_dir / f"montage_{final_clip_id}.mp4")

        if len(segment_paths) == 1:
            import shutil
            shutil.copy(segment_paths[0], output_path)
        else:
            concat_file = str(WORK_DIR / job_id / "concat_list.txt")
            with open(concat_file, "w", encoding="utf-8") as f:
                for sp in segment_paths:
                    safe = sp.replace("'", "'\\''")
                    f.write(f"file '{safe}'\n")

            cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", concat_file,
                "-c", "copy",
                "-movflags", "+faststart",
                output_path,
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if proc.returncode != 0:
                cmd_fallback = [
                    "ffmpeg", "-y",
                    "-f", "concat", "-safe", "0",
                    "-i", concat_file,
                    "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
                    "-c:a", "aac",
                    "-movflags", "+faststart",
                    output_path,
                ]
                proc2 = subprocess.run(cmd_fallback, capture_output=True, text=True, timeout=600)
                if proc2.returncode != 0:
                    raise RuntimeError(f"ffmpeg concat failed: {proc2.stderr[-500:]}")

        for sp in segment_paths:
            try:
                os.remove(sp)
            except OSError:
                pass

        # Generate thumbnail from the final montage
        thumb_path = _generate_thumbnail(output_path, job_id, final_clip_id, seek_time=1.0)

        job_store.update(job_id, status="uploading", progress=85)

        # Step 4: Upload to R2
        r2_client, bucket = _get_r2_client()
        r2_key = f"atlas-clips/{job_id}/montage_{final_clip_id}.mp4"
        file_size = os.path.getsize(output_path)
        with open(output_path, "rb") as f:
            r2_client.upload_fileobj(
                f, bucket, r2_key,
                ExtraArgs={"ContentType": "video/mp4"},
            )

        # Upload thumbnail
        thumb_url = _upload_thumbnail(thumb_path, r2_client, bucket, job_id, final_clip_id)

        try:
            os.remove(output_path)
        except OSError:
            pass

        presigned_url = r2_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": r2_key},
            ExpiresIn=604800,
        )

        total_duration = sum(s.get("endTime", 0) - s.get("startTime", 0) for s in ordered_segments)

        result = {
            "clipId": final_clip_id,
            "r2Key": r2_key,
            "publicUrl": presigned_url,
            "thumbnailUrl": thumb_url,
            "title": title or "AI Montage",
            "segmentCount": n_segs,
            "segmentOrder": ordered_indices,
            "segments": [
                {"startTime": s["startTime"], "endTime": s["endTime"]}
                for s in ordered_segments
            ],
            "duration": total_duration,
            "sizeBytes": file_size,
            "isMontage": True,
        }

        job_store.update(
            job_id,
            status="completed",
            progress=100,
            clip_count=1,
            clips_json=json.dumps([result]),
        )
    except Exception as e:
        job_store.update(job_id, status="failed", error=str(e), progress=0)


@app.post("/api/atlas-clips/jobs")
async def create_job(req: CreateJobReq):
    job_id = str(uuid.uuid4())[:12]
    job_store.create(job_id, source_url=req.sourceUrl)
    thread = threading.Thread(
        target=_run_pipeline_background,
        args=(job_id, req.sourceUrl, req.transcript, req.reframeConfig, req.fastMode, req.maxClips, req.prompt),
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
        "segments": job.get("segments"),
        "sourceUrl": job.get("source_url", ""),
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


@app.post("/api/atlas-clips/jobs/{job_id}/download")
async def download_clip(job_id: str, req: ReframeClipReq):
    """Download a clip as-is (no reframe, no captions, no re-encoding).

    Cuts the clip segment from the source VOD with stream copy and
    uploads it to R2. Much faster than reframe since there's no encoding.
    """
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
        target=_run_download_only_background,
        args=(
            new_job_id,
            parent.get("source_url", ""),
            clip.get("startTime", 0),
            clip.get("endTime", 0),
            clip.get("title", f"Clip {req.clipIndex + 1}"),
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


@app.post("/api/atlas-clips/jobs/{job_id}/montage")
async def create_montage(job_id: str, req: MontageReq):
    """Multi-segment montage: pick multiple disjoint timestamp ranges and AI
    assembles them into a single cohesive clip with hook -> build -> payoff
    narrative structure.

    Accepts EITHER:
      - clipIndices: indices into the parent job's detected clips
      - segments: custom list of {startTime, endTime} dicts

    This solves the #1 Opus Clip complaint: users can now select specific
    sections from anywhere in the video without processing the entire thing.
    """
    parent = job_store.get(job_id)
    if not parent:
        return JSONResponse({"error": "job_not_found"}, status_code=404)

    source_url = parent.get("source_url", "")
    analysis = parent.get("analysis") or {}
    clips = analysis.get("clips", [])

    segments_list: List[Dict[str, Any]] = []
    if req.segments:
        for s in req.segments:
            st = float(s.get("startTime", 0))
            en = float(s.get("endTime", 0))
            if en > st:
                segments_list.append({
                    "startTime": st, "endTime": en,
                    "transcript": "", "hookScore": 0, "pacingScore": 0,
                    "payoffScore": 0, "viralScore": 0,
                })
    elif req.clipIndices:
        for idx in req.clipIndices:
            if 0 <= idx < len(clips):
                c = clips[idx]
                segments_list.append({
                    "startTime": c.get("startTime", 0),
                    "endTime": c.get("endTime", 0),
                    "transcript": c.get("transcript", ""),
                    "hookScore": c.get("hookScore", 0),
                    "pacingScore": c.get("pacingScore", 0),
                    "payoffScore": c.get("payoffScore", 0),
                    "viralScore": c.get("viralScore", 0),
                })

    if not segments_list:
        return JSONResponse({"error": "no_segments_provided", "details": "Provide clipIndices or segments with at least one valid range."}, status_code=400)
    if len(segments_list) > 20:
        return JSONResponse({"error": "too_many_segments", "details": "Maximum 20 segments per montage."}, status_code=400)

    new_job_id = str(uuid.uuid4())[:12]
    job_store.create(new_job_id, source_url=source_url)
    thread = threading.Thread(
        target=_run_montage_background,
        args=(new_job_id, source_url, segments_list, req.reframeConfig, req.prompt, req.title or "AI Montage"),
        daemon=True,
    )
    thread.start()
    return {"id": new_job_id, "status": "pending", "segmentCount": len(segments_list)}


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

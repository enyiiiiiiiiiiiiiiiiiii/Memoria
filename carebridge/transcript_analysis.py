from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Any

from .config import ENABLE_GEMINI, GEMINI_API_KEY

POSITIVE_WORDS = {
    "good", "better", "happy", "fine", "okay", "calm", "peaceful", "thanks", "thankful", "glad"
}
NEGATIVE_WORDS = {
    "sad", "lost", "confused", "worried", "afraid", "angry", "upset", "tired", "lonely", "stress"
}
MEMORY_WORDS = {
    "forgot", "forget", "remember", "misplaced", "confused", "lost", "where", "when", "keys", "medicine"
}
SOCIAL_RISK_WORDS = {"alone", "nobody", "no one", "isolated", "empty", "boring"}
FILLERS = {"um", "uh", "erm", "ah", "like", "you know"}


def _safe_div(n: float, d: float) -> float:
    return n / d if d else 0.0


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", text.lower())


def analyze_transcript(text: str) -> dict[str, Any]:
    cleaned = (text or "").strip()
    tokens = _tokenize(cleaned)
    token_count = len(tokens)
    counts = Counter(tokens)
    repeated = sum(1 for token, count in counts.items() if count >= 3 and len(token) > 3)
    sentences = [s.strip() for s in re.split(r"[.!?]+", cleaned) if s.strip()]
    avg_sentence_len = _safe_div(token_count, len(sentences))

    positive_hits = sum(token in POSITIVE_WORDS for token in tokens)
    negative_hits = sum(token in NEGATIVE_WORDS for token in tokens)
    memory_hits = sum(token in MEMORY_WORDS for token in tokens)
    social_hits = sum(token in SOCIAL_RISK_WORDS for token in tokens)
    filler_hits = sum(token in FILLERS for token in tokens)
    pause_hits = cleaned.count("...") + cleaned.count("[pause]")

    coherence_risk = min(1.0, 0.15 * repeated + 0.04 * filler_hits + 0.08 * pause_hits)
    if avg_sentence_len < 4 and token_count > 8:
        coherence_risk += 0.15
    if avg_sentence_len > 30:
        coherence_risk += 0.12
    coherence_risk = min(coherence_risk, 1.0)

    memory_score = min(1.0, 0.18 * memory_hits + 0.10 * pause_hits)
    mood_score = min(1.0, 0.18 * negative_hits - 0.06 * positive_hits + 0.08 * social_hits)
    mood_score = max(0.0, mood_score)

    concern = min(1.0, 0.45 * memory_score + 0.25 * coherence_risk + 0.20 * mood_score + 0.10 * min(1.0, social_hits * 0.25))

    flags: list[str] = []
    if memory_hits >= 2:
        flags.append("Repeated memory-related expressions detected")
    if pause_hits >= 1 or filler_hits >= 3:
        flags.append("Possible hesitation or word-finding difficulty")
    if social_hits >= 1:
        flags.append("Possible loneliness or social withdrawal signal")
    if negative_hits > positive_hits:
        flags.append("Negative affect outweighs positive affect in transcript")
    if not flags:
        flags.append("No major high-risk transcript marker detected in this short sample")

    level = "Low" if concern < 0.35 else "Moderate" if concern < 0.65 else "High"
    alert = {
        "Low": "Maintain routine follow-up and encourage social engagement.",
        "Moderate": "Recommend caregiver follow-up and repeat check within 1 week.",
        "High": "Escalate to caregiver or clinician review soon; transcript warrants closer monitoring.",
    }[level]

    return {
        "summary": f"Transcript concern assessed as {level.lower()} from memory, coherence, and mood markers.",
        "memory_score": round(memory_score, 3),
        "coherence_score": round(1 - coherence_risk, 3),
        "mood_score": round(1 - mood_score, 3),
        "concern_score": round(concern, 3),
        "level": level,
        "flags": flags,
        "token_count": token_count,
        "avg_sentence_len": round(avg_sentence_len, 2),
        "alert": alert,
    }


def analyze_chat_sentiment(messages: list[str]) -> dict[str, Any]:
    joined = " ".join(messages)
    analysis = analyze_transcript(joined)
    engagement = max(0.0, min(1.0, math.log(len(joined) + 1, 10) / 3))
    analysis["engagement_score"] = round(engagement, 3)
    return analysis


def maybe_run_gemini(prompt: str) -> dict[str, Any] | None:
    if not ENABLE_GEMINI or not GEMINI_API_KEY:
        return None
    try:
        from google import genai  # type: ignore

        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=(
                "You are assisting in a dementia-support prototype. Return compact JSON with keys "
                "summary, memory_score, coherence_score, mood_score, concern_score, level, flags, alert. "
                "Do not diagnose. Analyse this transcript:\n" + prompt
            ),
        )
        text = response.text.strip()
        return json.loads(text)
    except Exception:
        return None

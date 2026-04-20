from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from . import db


def _split_tags(value: str | None) -> set[str]:
    if not value:
        return set()
    return {part.strip().lower() for part in value.split(",") if part.strip()}


def get_patient_profile(patient_user_id: int) -> dict[str, Any]:
    row = db.fetch_one(
        """
        SELECT u.id as user_id, u.full_name, ep.*
        FROM users u
        LEFT JOIN elder_profiles ep ON ep.user_id = u.id
        WHERE u.id = ?
        """,
        (patient_user_id,),
    )
    return dict(row) if row else {}


def suggest_activity_matches(patient_user_id: int) -> list[dict[str, Any]]:
    profile = get_patient_profile(patient_user_id)
    interests = _split_tags(profile.get("interests"))
    age = int(profile.get("age") or 70)
    activities = [dict(row) for row in db.fetch_all("SELECT * FROM activities ORDER BY starts_at ASC")]

    matches: list[dict[str, Any]] = []
    for activity in activities:
        tags = _split_tags(activity.get("interest_tags"))
        overlap = len(interests & tags)
        age_bonus = 1 if str(activity.get("age_group", "")).startswith("65") and age >= 65 else 0
        score = overlap * 2 + age_bonus
        activity["match_score"] = score
        activity["match_reason"] = ", ".join(sorted(interests & tags)) or "Good age-group fit"
        matches.append(activity)
    return sorted(matches, key=lambda item: item["match_score"], reverse=True)


def suggest_peer_matches(patient_user_id: int) -> list[dict[str, Any]]:
    patient = get_patient_profile(patient_user_id)
    patient_interests = _split_tags(patient.get("interests"))
    age = int(patient.get("age") or 70)

    rows = db.fetch_all(
        """
        SELECT u.id as user_id, u.full_name, ep.age, ep.interests, ep.language_pref
        FROM users u
        JOIN elder_profiles ep ON ep.user_id = u.id
        WHERE u.role = 'patient' AND u.id != ?
        """,
        (patient_user_id,),
    )
    matches: list[dict[str, Any]] = []
    for row in rows:
        candidate = dict(row)
        candidate_interests = _split_tags(candidate.get("interests"))
        overlap = len(patient_interests & candidate_interests)
        age_gap = abs((candidate.get("age") or age) - age)
        score = overlap * 3 + max(0, 2 - age_gap / 10)
        candidate["match_score"] = round(score, 2)
        candidate["match_reason"] = ", ".join(sorted(patient_interests & candidate_interests)) or "Similar age band"
        matches.append(candidate)
    return sorted(matches, key=lambda item: item["match_score"], reverse=True)


def participants_needing_follow_up() -> list[dict[str, Any]]:
    threshold = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")
    rows = db.fetch_all(
        """
        SELECT u.id, u.full_name, ep.age, ep.last_risk_level, ep.last_risk_score,
               MAX(i.created_at) AS last_interaction,
               AVG(COALESCE(i.concern_score, 0)) AS mean_concern
        FROM users u
        JOIN elder_profiles ep ON ep.user_id = u.id
        LEFT JOIN interactions i ON i.patient_user_id = u.id
        WHERE u.role = 'patient'
        GROUP BY u.id, u.full_name, ep.age, ep.last_risk_level, ep.last_risk_score
        ORDER BY ep.last_risk_score DESC, mean_concern DESC
        """
    )
    flagged = []
    for row in rows:
        item = dict(row)
        item["needs_follow_up"] = (
            not item.get("last_interaction")
            or item["last_interaction"] < threshold
            or float(item.get("mean_concern") or 0) >= 0.5
            or float(item.get("last_risk_score") or 0) >= 0.6
        )
        if item["needs_follow_up"]:
            flagged.append(item)
    return flagged

from __future__ import annotations

import json

from . import auth, db
from .config import DEMO_USERS


EDUCATION_VIDEOS = [
    {
        "title": "Spotting early signs of dementia",
        "description": "A quick primer on memory, mood, and functional changes that should prompt closer attention.",
        "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "category": "Signs and symptoms",
    },
    {
        "title": "How to respond during agitation and sundowning",
        "description": "Practical de-escalation steps for evening restlessness and confusion.",
        "url": "https://www.youtube.com/watch?v=ysz5S6PUM-U",
        "category": "Caregiving skills",
    },
    {
        "title": "Medication and home safety checklist",
        "description": "Reducing medication mix-ups, fall risk, and wandering hazards at home.",
        "url": "https://www.youtube.com/watch?v=jNQXAC9IVRw",
        "category": "Safety",
    },
]

QUIZ_QUESTIONS = [
    {
        "question": "Which change is most appropriate to flag as a possible early dementia sign?",
        "option_a": "Occasionally forgetting a name but recalling it later",
        "option_b": "Repeated confusion about familiar routes and appointments",
        "option_c": "Preferring spicy food less often",
        "option_d": "Taking a daytime nap after a busy week",
        "correct_option": "B",
        "rationale": "Persistent disorientation in familiar routines is more concerning than normal minor forgetfulness.",
        "category": "Signs",
    },
    {
        "question": "A patient becomes agitated in the evening. What is the best first response?",
        "option_a": "Correct them immediately and raise your voice",
        "option_b": "Offer reassurance, reduce stimulation, and use simple prompts",
        "option_c": "Ignore them completely",
        "option_d": "Test them with more questions",
        "correct_option": "B",
        "rationale": "Low-stimulation reassurance is a safer first response during sundowning or confusion.",
        "category": "Caregiving",
    },
    {
        "question": "Why should patients not see all confidential clinical notes by default?",
        "option_a": "Because caregivers own the account",
        "option_b": "Because unclear risk language can create unnecessary stress without context",
        "option_c": "Because doctors dislike transparency",
        "option_d": "Because notes are never useful to patients",
        "correct_option": "B",
        "rationale": "The app uses role-based views to reduce harm from raw clinical wording while keeping support information visible.",
        "category": "Ethics",
    },
]

ACTIVITIES = [
    {
        "title": "Heartland Mahjong Memory Hour",
        "description": "Mahjong-themed memory and matching activities with volunteers. Low-pressure and social.",
        "category": "Cognitive",
        "age_group": "65-79",
        "interest_tags": "mahjong,games,hawker culture",
        "location": "Tanjong Pagar CC",
        "starts_at": "2026-04-10 10:00",
        "capacity": 20,
    },
    {
        "title": "Kopitiam Stories Circle",
        "description": "Reminiscence and storytelling with peers who enjoy food, local history, and conversation.",
        "category": "Social",
        "age_group": "65+",
        "interest_tags": "stories,hawker,conversation,history",
        "location": "Toa Payoh Library",
        "starts_at": "2026-04-11 14:00",
        "capacity": 18,
    },
    {
        "title": "Gentle Movement and Music",
        "description": "Light seated exercise with multilingual playlists and volunteer support.",
        "category": "Wellbeing",
        "age_group": "60+",
        "interest_tags": "music,movement,wellbeing",
        "location": "Bukit Merah Active Ageing Centre",
        "starts_at": "2026-04-13 09:30",
        "capacity": 25,
    },
]

SAMPLE_NOTES = [
    "Family reports increased forgetfulness over the past 3 months, especially around medication timing.",
    "Evening agitation appears worse on days with poor sleep. Consider calming routine before 7pm.",
]

SAMPLE_TRANSCRIPT = (
    "Hello auntie, how have you been? I forgot where I kept my keys again and I felt a bit lost yesterday. "
    "Sometimes I cannot remember whether I already ate lunch. But talking today makes me feel better."
)


def seed_if_needed() -> None:
    if db.fetch_one("SELECT id FROM users LIMIT 1"):
        return

    user_ids: dict[str, int] = {}
    for user in DEMO_USERS:
        user_ids[user["role"]] = auth.register_user(
            username=user["username"],
            password=user["password"],
            full_name=user["full_name"],
            role=user["role"],
            email=user["email"],
        )

    patient_id = user_ids["patient"]
    caregiver_id = user_ids["caregiver"]
    volunteer_id = user_ids["volunteer"]
    doctor_id = user_ids["doctor"]

    db.upsert_elder_profile(
        patient_id,
        age=74,
        language_pref="English",
        dialect="Hokkien",
        interests="mahjong,hker food,old tv dramas,gardening,walking",
        conditions="hypertension,diabetes",
        mobility_level="Independent with occasional supervision",
        caregiver_notes="Lives with daughter. Most alert in the morning.",
        last_risk_score=0.61,
        last_risk_level="Moderate",
        last_risk_summary="Memory complaints and functional slips warrant closer monitoring.",
    )

    db.execute(
        "INSERT INTO care_links (caregiver_user_id, patient_user_id, link_type) VALUES (?, ?, ?)",
        (caregiver_id, patient_id, "daughter"),
    )
    db.execute(
        "INSERT INTO volunteer_assignments (volunteer_user_id, patient_user_id, active) VALUES (?, ?, 1)",
        (volunteer_id, patient_id),
    )

    for note in SAMPLE_NOTES:
        db.execute(
            "INSERT INTO confidential_notes (patient_user_id, author_user_id, note) VALUES (?, ?, ?)",
            (patient_id, doctor_id, note),
        )

    for item in EDUCATION_VIDEOS:
        db.execute(
            """
            INSERT INTO education_videos (title, description, url, category, target_role)
            VALUES (?, ?, ?, ?, 'caregiver')
            """,
            (item["title"], item["description"], item["url"], item["category"]),
        )

    for item in QUIZ_QUESTIONS:
        db.execute(
            """
            INSERT INTO quiz_questions
            (question, option_a, option_b, option_c, option_d, correct_option, rationale, category)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item["question"],
                item["option_a"],
                item["option_b"],
                item["option_c"],
                item["option_d"],
                item["correct_option"],
                item["rationale"],
                item["category"],
            ),
        )

    for activity in ACTIVITIES:
        db.execute(
            """
            INSERT INTO activities (title, description, category, age_group, interest_tags, organiser_user_id, location, starts_at, capacity)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                activity["title"],
                activity["description"],
                activity["category"],
                activity["age_group"],
                activity["interest_tags"],
                volunteer_id,
                activity["location"],
                activity["starts_at"],
                activity["capacity"],
            ),
        )

    db.execute(
        """
        INSERT INTO interactions (patient_user_id, actor_user_id, mode, content, concern_score, analysis_json)
        VALUES (?, ?, 'call', ?, ?, ?)
        """,
        (
            patient_id,
            volunteer_id,
            SAMPLE_TRANSCRIPT,
            0.63,
            json.dumps(
                {
                    "summary": "Transcript contains repeated forgetting and mild disorientation markers.",
                    "memory_score": 0.72,
                    "mood_score": 0.35,
                    "coherence_score": 0.58,
                    "alert": "Monitor lunch and medication routine this week.",
                }
            ),
        ),
    )

    db.execute(
        "INSERT INTO messages (sender_user_id, recipient_user_id, body) VALUES (?, ?, ?)",
        (volunteer_id, patient_id, "Good morning! Shall we do our short memory game at 3pm today?"),
    )
    db.execute(
        "INSERT INTO messages (sender_user_id, recipient_user_id, body) VALUES (?, ?, ?)",
        (patient_id, volunteer_id, "Yes, after lunch please."),
    )

    db.execute(
        "INSERT INTO alerts (patient_user_id, severity, title, body, source) VALUES (?, ?, ?, ?, ?)",
        (
            patient_id,
            "medium",
            "Routine slip detected",
            "Recent call transcript suggests repeated uncertainty about meals and keys.",
            "call_analysis",
        ),
    )

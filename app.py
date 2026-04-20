from __future__ import annotations

import json
import random
import time
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from carebridge import db
from carebridge.auth import authenticate
from carebridge.community import suggest_activity_matches, suggest_peer_matches
from carebridge.config import (
    ADDRESSO_DIR,
    APP_TAGLINE,
    APP_TITLE,
    EEG_DIR,
    EXPORT_DIR,
    IMAGING_DIR,
    LANGUAGE_LABELS,
    ROLE_LABELS,
    TABULAR_DIR,
)
from carebridge.games import (
    build_memory_board,
    evaluate_kopi_guess,
    new_kopi_round,
)
from carebridge.i18n import t
from carebridge.resources import SINGAPORE_RESOURCES
from carebridge.risk_engine import (
    load_addresso_metadata,
    load_eeg_metadata,
    load_imaging_metadata,
    load_tabular_metadata,
    load_training_metadata,
    predict_profile,
    train_addresso_bundle,
    train_and_save_model,
    train_eeg_bundle,
    train_imaging_bundle,
)
from carebridge.seed import seed_if_needed

st.set_page_config(page_title=APP_TITLE, page_icon="🧠", layout="wide")

SENTENCE_RECALL_PROMPTS = [
    "The elderly man watered the plants before taking the bus to the market.",
    "Every morning, she folds the newspaper and makes a cup of warm tea.",
    "The nurse reminded the patient to bring his glasses and medicine after lunch.",
    "On Sunday, the grandchildren visited and played board games in the living room.",
    "The caregiver wrote a shopping list with rice, eggs, milk, and fruit.",
    "After breakfast, the old woman opened the window and fed the small birds outside.",
    "The doctor checked her pulse before explaining the new medicine schedule to the family.",
    "At sunset, the neighbours sat downstairs and talked about the weather and rising prices.",
    "Before dinner, he placed the clean plates carefully on the table near the blue vase.",
    "The patient smiled when her daughter brought mangoes, bread, and a warm bowl of soup.",
    "Every Friday, the volunteer walks with Mr Tan around the park beside the community centre.",
    "She packed her umbrella and wallet before leaving the clinic for the taxi stand.",
    "The little boy helped his grandmother carry the laundry basket into the bright kitchen.",
    "After the rain stopped, they slowly crossed the road and entered the quiet pharmacy.",
    "The caregiver switched off the television so the patient could rest for a while.",
    "At the hawker centre, he ordered fish porridge and a cup of barley without ice.",
    "The family wrote important phone numbers on a card and kept it near the front door.",
    "Before the appointment, she brushed her hair and looked for her pink health booklet.",
    "The old man placed his keys beside the radio and forgot them a few minutes later.",
    "In the afternoon, the nurse changed the bandage and opened the curtains for more light.",
    "She counted the oranges twice before putting them into the basket beside the sink.",
    "The patient laughed softly when the children sang an old school song in the hallway.",
    "On Monday morning, he wore his grey shoes and waited downstairs for the shuttle bus.",
    "The caregiver warmed the rice, cut the fruit, and poured a glass of water.",
    "After reading the letter, she placed it carefully inside the wooden drawer near the bed.",
    "The pharmacist explained that the tablets should be taken after meals and before sleep.",
    "At the playground, the grandfather watched his grandson run past the swing and slide.",
    "The woman cleaned her glasses with a tissue before reading the menu on the wall.",
    "Every evening, they lock the gate, water the flowers, and turn on the porch light.",
    "The doctor asked the patient to raise her hand and repeat the date and month."
]

def ensure_reminders_table() -> None:
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS reminders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_user_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            description TEXT,
            due_date TEXT,
            status TEXT NOT NULL DEFAULT 'active',
            requires_approval INTEGER NOT NULL DEFAULT 0,
            is_approved INTEGER NOT NULL DEFAULT 1,
            created_by_user_id INTEGER NOT NULL,
            approved_by_user_id INTEGER,
            completed_by_user_id INTEGER,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            approved_at TEXT,
            completed_at TEXT
        )
        """
    )


def ensure_memoria_tables() -> None:
    ensure_reminders_table()
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS caregiver_journal (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_user_id INTEGER NOT NULL,
            caregiver_user_id INTEGER NOT NULL,
            entry TEXT NOT NULL,
            mood TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS community_groups (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            description TEXT,
            icon TEXT,
            active_members INTEGER DEFAULT 0
        )
        """
    )
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS community_memberships (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_user_id INTEGER NOT NULL,
            group_id INTEGER NOT NULL,
            joined_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(patient_user_id, group_id)
        )
        """
    )
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS community_group_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            group_id INTEGER NOT NULL,
            sender_user_id INTEGER NOT NULL,
            body TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    default_groups = [
        ("Gardening Club", "Share plant care tips and weekly garden photos.", "🌱", 12),
        ("Morning Kopi Talk", "A light morning chat group for familiar routines.", "☕", 8),
        ("Music Memories", "Listen to songs and talk about favourite memories.", "🎵", 15),
        ("Gentle Walks", "Neighbourhood walking reminders and encouragement.", "🚶", 10),
    ]
    for name, description, icon, active_members in default_groups:
        db.execute(
            """
            INSERT OR IGNORE INTO community_groups (name, description, icon, active_members)
            VALUES (?, ?, ?, ?)
            """,
            (name, description, icon, active_members),
        )


def bootstrap() -> None:
    db.init_db()
    seed_if_needed()
    ensure_memoria_tables()


bootstrap()


QUOTE_BANK = [
    "Small steps still move you forward.",
    "A familiar routine can make today feel lighter.",
    "You are supported, remembered, and never alone.",
    "One calm moment at a time.",
]

SENTENCE_DISTRACTOR_WORDS = [
    "orange", "clinic", "window", "radio", "garden", "blue", "ticket", "chair",
    "river", "blanket", "phone", "school", "market", "quiet", "family", "music",
]

LOCAL_COPY = {
    "emoji_game_caption": {
        "en": "Find matching emoji pairs. Tap the question marks to reveal two at a time.",
        "zh": "找出相同的表情配对。点击问号，每次翻开两个。",
        "ms": "Cari pasangan emoji yang sama. Tekan tanda soal untuk buka dua pada satu masa.",
        "ta": "ஒரே மாதிரியான எமோஜி ஜோடிகளை கண்டுபிடிக்கவும். கேள்விக்குறிகளைத் தட்டி, ஒரே நேரத்தில் இரண்டு திறக்கவும்。",
    },
    "drink_legend_title": {
        "en": "Colour legend",
        "zh": "颜色图例",
        "ms": "Petunjuk warna",
        "ta": "நிற விளக்கம்",
    },
    "drink_legend_desc": {
        "en": "Use the colours in the diagram to identify the drink correctly.",
        "zh": "根据图中的颜色来辨认正确的饮料。",
        "ms": "Gunakan warna dalam rajah untuk mengenal pasti minuman yang betul.",
        "ta": "வரைபடத்தில் உள்ள நிறங்களைப் பயன்படுத்தி சரியான பானத்தை கண்டறியவும்。",
    },
    "legend_black": {
        "en": "Black: coffee",
        "zh": "黑色：咖啡",
        "ms": "Hitam: kopi",
        "ta": "கருப்பு: காபி",
    },
    "legend_brown": {
        "en": "Brown: tea",
        "zh": "棕色：茶",
        "ms": "Perang: teh",
        "ta": "பழுப்பு: தேநீர்",
    },
    "legend_blue": {
        "en": "Blue: water",
        "zh": "蓝色：水",
        "ms": "Biru: air",
        "ta": "நீலம்: தண்ணீர்",
    },
    "legend_white_cubes": {
        "en": "White cubes: sugar",
        "zh": "白色方块：糖",
        "ms": "Kiub putih: gula",
        "ta": "வெள்ளைக் கட்டிகள்: சர்க்கரை",
    },
    "legend_beige": {
        "en": "Beige: condensed milk",
        "zh": "米色：炼奶",
        "ms": "Beige: susu pekat",
        "ta": "மங்கலான மஞ்சள்: கண்டென்ஸ்டு பால்",
    },
    "tap_to_reveal": {
        "en": "Tap to reveal",
        "zh": "点击翻开",
        "ms": "Tekan untuk buka",
        "ta": "திறக்க தட்டவும்",
    },
    "group_chat": {
        "en": "Group chat",
        "zh": "群聊",
        "ms": "Sembang kumpulan",
        "ta": "குழு உரையாடல்",
    },
    "open_group_chat": {
        "en": "Open chat",
        "zh": "打开群聊",
        "ms": "Buka sembang",
        "ta": "உரையாடலைத் திறக்கவும்",
    },
    "joined_group": {
        "en": "You joined this group.",
        "zh": "你已加入这个群组。",
        "ms": "Anda telah menyertai kumpulan ini.",
        "ta": "நீங்கள் இந்த குழுவில் இணைந்துள்ளீர்கள்.",
    },
    "say_hello_group": {
        "en": "Say hello to the group",
        "zh": "向群组打个招呼",
        "ms": "Ucap hello kepada kumpulan",
        "ta": "குழுவுக்கு வணக்கம் சொல்லுங்கள்",
    },
    "no_group_messages": {
        "en": "No messages yet. Start the conversation.",
        "zh": "还没有消息。开始聊天吧。",
        "ms": "Belum ada mesej. Mulakan perbualan.",
        "ta": "இன்னும் செய்திகள் இல்லை. உரையாடலை தொடங்குங்கள்。",
    },
}


def current_language() -> str:
    return st.session_state.get("language", "en")


def ui(key: str) -> str:
    lang = current_language()
    fallback = LOCAL_COPY.get(key, {}).get(lang) or LOCAL_COPY.get(key, {}).get("en")
    translated = t(key, lang)
    if translated == key and fallback:
        return fallback
    return translated


def inject_memoria_theme() -> None:
    st.markdown(
        """
        <style>
        .memoria-hero {
            padding: 2.5rem 3rem;
            border-radius: 34px;
            background: linear-gradient(135deg, #3bb273 0%, #2f8f83 100%);
            color: white;
            box-shadow: 0 16px 35px rgba(31, 115, 96, 0.18);
            margin-bottom: 1.5rem;
        }
        .memoria-hero h1 {
            font-size: 3rem;
            margin: 0.2rem 0 0.2rem 0;
            color: white;
        }
        .memoria-hero p {font-size: 1.25rem; margin: 0.15rem 0;}
        .memoria-card {
            background: white;
            border-radius: 24px;
            border: 1px solid #e9edf3;
            box-shadow: 0 8px 24px rgba(0,0,0,0.06);
            padding: 1.3rem 1.5rem;
            margin-bottom: 1rem;
        }
        .memoria-card-soft {
            background: #f3f7ff;
            border-radius: 24px;
            border: 1px solid #dbe6ff;
            padding: 1.3rem 1.5rem;
            margin-bottom: 1rem;
        }
        .memoria-community-card {
            display:flex;
            align-items:center;
            gap: 1rem;
            padding: 1rem 1.2rem;
            border-radius: 22px;
            border: 1px solid #e8e8e8;
            background: #ffffff;
            margin: 0.75rem 0;
        }
        .memoria-icon-bubble {
            height: 64px;
            width: 64px;
            display:flex;
            align-items:center;
            justify-content:center;
            border-radius: 50%;
            background: #e9fff1;
            font-size: 2rem;
        }
        .word-chip {
            display:inline-block;
            padding: 0.7rem 1rem;
            border-radius: 16px;
            background: #f1f5ff;
            border: 1px solid #dce6ff;
            margin: 0.25rem;
            font-weight: 700;
        }
        .emoji-grid div[data-testid="column"] > div:has(button) button {
            height: 150px !important;
            min-height: 150px !important;
            width: 100% !important;
            border-radius: 20px !important;
            font-size: 2.2rem !important;
            padding: 0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def greeting_key() -> str:
    hour = datetime.now().hour
    if hour < 12:
        return "good_morning"
    if hour < 18:
        return "good_afternoon"
    return "good_evening"


def get_community_groups(patient_user_id: int) -> list[dict[str, Any]]:
    rows = db.fetch_all(
        """
        SELECT cg.*, cm.id AS membership_id
        FROM community_groups cg
        LEFT JOIN community_memberships cm
          ON cm.group_id = cg.id AND cm.patient_user_id = ?
        ORDER BY cg.id ASC
        """,
        (patient_user_id,),
    )
    return [dict(row) for row in rows]


def join_community(patient_user_id: int, group_id: int) -> None:
    db.execute(
        "INSERT OR IGNORE INTO community_memberships (patient_user_id, group_id) VALUES (?, ?)",
        (patient_user_id, group_id),
    )


def get_group_messages(group_id: int) -> list[dict[str, Any]]:
    rows = db.fetch_all(
        """
        SELECT cgm.*, u.full_name AS sender_name
        FROM community_group_messages cgm
        JOIN users u ON u.id = cgm.sender_user_id
        WHERE cgm.group_id = ?
        ORDER BY cgm.created_at ASC
        """,
        (group_id,),
    )
    return [dict(row) for row in rows]


def send_group_message(group_id: int, sender_user_id: int, body: str) -> None:
    db.execute(
        "INSERT INTO community_group_messages (group_id, sender_user_id, body) VALUES (?, ?, ?)",
        (group_id, sender_user_id, body),
    )


def save_caregiver_journal(patient_user_id: int, caregiver_user_id: int, entry: str, mood: str) -> None:
    db.execute(
        """
        INSERT INTO caregiver_journal (patient_user_id, caregiver_user_id, entry, mood)
        VALUES (?, ?, ?, ?)
        """,
        (patient_user_id, caregiver_user_id, entry, mood),
    )


def get_caregiver_journal_entries(patient_user_id: int) -> list[dict[str, Any]]:
    rows = db.fetch_all(
        """
        SELECT cj.*, u.full_name AS caregiver_name
        FROM caregiver_journal cj
        JOIN users u ON u.id = cj.caregiver_user_id
        WHERE cj.patient_user_id = ?
        ORDER BY cj.created_at DESC
        """,
        (patient_user_id,),
    )
    return [dict(row) for row in rows]


def get_user(user_id: int) -> dict:
    row = db.fetch_one("SELECT * FROM users WHERE id = ?", (user_id,))
    return dict(row) if row else {}


def get_profile(user_id: int) -> dict:
    row = db.fetch_one("SELECT * FROM elder_profiles WHERE user_id = ?", (user_id,))
    return dict(row) if row else {}


def get_patients_for_caregiver(user_id: int) -> list[dict]:
    rows = db.fetch_all(
        """
        SELECT u.*, ep.age, ep.language_pref, ep.interests, ep.last_risk_score, ep.last_risk_level, ep.last_risk_summary
        FROM care_links cl
        JOIN users u ON u.id = cl.patient_user_id
        LEFT JOIN elder_profiles ep ON ep.user_id = u.id
        WHERE cl.caregiver_user_id = ?
        ORDER BY u.full_name
        """,
        (user_id,),
    )
    return [dict(row) for row in rows]


def get_primary_caregiver_for_patient(patient_user_id: int) -> dict | None:
    row = db.fetch_one(
        """
        SELECT u.*
        FROM care_links cl
        JOIN users u ON u.id = cl.caregiver_user_id
        WHERE cl.patient_user_id = ?
        ORDER BY u.full_name
        LIMIT 1
        """,
        (patient_user_id,),
    )
    return dict(row) if row else None


def get_all_patients() -> list[dict]:
    rows = db.fetch_all(
        """
        SELECT u.*, ep.age, ep.language_pref, ep.interests, ep.last_risk_score, ep.last_risk_level, ep.last_risk_summary
        FROM users u
        LEFT JOIN elder_profiles ep ON ep.user_id = u.id
        WHERE u.role = 'patient'
        ORDER BY u.full_name
        """
    )
    return [dict(row) for row in rows]


def get_alerts(patient_user_id: int) -> list[dict]:
    return [
        dict(row)
        for row in db.fetch_all(
            "SELECT * FROM alerts WHERE patient_user_id = ? ORDER BY created_at DESC",
            (patient_user_id,),
        )
    ]


def get_notes(patient_user_id: int) -> list[dict]:
    rows = db.fetch_all(
        """
        SELECT cn.*, u.full_name AS author_name, u.role AS author_role
        FROM confidential_notes cn
        JOIN users u ON u.id = cn.author_user_id
        WHERE cn.patient_user_id = ?
        ORDER BY cn.created_at DESC
        """,
        (patient_user_id,),
    )
    return [dict(row) for row in rows]


def get_messages(user_a: int, user_b: int) -> list[dict]:
    rows = db.fetch_all(
        """
        SELECT m.*, s.full_name AS sender_name
        FROM messages m
        JOIN users s ON s.id = m.sender_user_id
        WHERE (sender_user_id = ? AND recipient_user_id = ?)
           OR (sender_user_id = ? AND recipient_user_id = ?)
        ORDER BY created_at ASC
        """,
        (user_a, user_b, user_b, user_a),
    )
    return [dict(row) for row in rows]


def send_message(sender_id: int, recipient_id: int, body: str) -> None:
    db.execute(
        "INSERT INTO messages (sender_user_id, recipient_user_id, body) VALUES (?, ?, ?)",
        (sender_id, recipient_id, body),
    )


def get_reminders(patient_user_id: int, include_pending: bool = True) -> list[dict]:
    if include_pending:
        rows = db.fetch_all(
            """
            SELECT r.*, cu.full_name AS created_by_name, au.full_name AS approved_by_name, co.full_name AS completed_by_name
            FROM reminders r
            LEFT JOIN users cu ON cu.id = r.created_by_user_id
            LEFT JOIN users au ON au.id = r.approved_by_user_id
            LEFT JOIN users co ON co.id = r.completed_by_user_id
            WHERE r.patient_user_id = ?
            ORDER BY r.created_at DESC
            """,
            (patient_user_id,),
        )
    else:
        rows = db.fetch_all(
            """
            SELECT r.*, cu.full_name AS created_by_name, au.full_name AS approved_by_name, co.full_name AS completed_by_name
            FROM reminders r
            LEFT JOIN users cu ON cu.id = r.created_by_user_id
            LEFT JOIN users au ON au.id = r.approved_by_user_id
            LEFT JOIN users co ON co.id = r.completed_by_user_id
            WHERE r.patient_user_id = ? AND r.is_approved = 1
            ORDER BY r.created_at DESC
            """,
            (patient_user_id,),
        )
    return [dict(row) for row in rows]


def create_reminder(
    patient_user_id: int,
    creator_user_id: int,
    creator_role: str,
    title: str,
    description: str,
    due_date: str,
) -> None:
    requires_approval = 1 if creator_role == "caregiver" else 0
    is_approved = 0 if creator_role == "caregiver" else 1
    status = "pending_approval" if creator_role == "caregiver" else "active"
    approved_by = None if creator_role == "caregiver" else creator_user_id
    approved_at = None if creator_role == "caregiver" else datetime.now().isoformat(sep=" ", timespec="seconds")

    db.execute(
        """
        INSERT INTO reminders (
            patient_user_id, title, description, due_date, status,
            requires_approval, is_approved, created_by_user_id,
            approved_by_user_id, approved_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            patient_user_id,
            title,
            description,
            due_date,
            status,
            requires_approval,
            is_approved,
            creator_user_id,
            approved_by,
            approved_at,
        ),
    )


def approve_reminder(reminder_id: int, doctor_user_id: int) -> None:
    db.execute(
        """
        UPDATE reminders
        SET is_approved = 1,
            status = 'active',
            approved_by_user_id = ?,
            approved_at = CURRENT_TIMESTAMP
        WHERE id = ? AND is_approved = 0
        """,
        (doctor_user_id, reminder_id),
    )


def complete_reminder(reminder_id: int, actor_user_id: int) -> None:
    db.execute(
        """
        UPDATE reminders
        SET status = 'completed',
            completed_by_user_id = ?,
            completed_at = CURRENT_TIMESTAMP
        WHERE id = ? AND status != 'completed'
        """,
        (actor_user_id, reminder_id),
    )


def total_points(user_id: int) -> int:
    row = db.fetch_one(
        "SELECT COALESCE(SUM(points_awarded), 0) AS total_points FROM game_sessions WHERE user_id = ?",
        (user_id,),
    )
    return int(row["total_points"] if row else 0)


def reward_for_total_points(total_points: int) -> dict[str, Any] | None:
    rewards = [
        {"threshold": 600, "voucher_name": "$5 Community Cafe", "partner_name": "Local Kopitiam"},
        {"threshold": 1000, "voucher_name": "$8 Pharmacy Voucher", "partner_name": "Neighbourhood Pharmacy"},
        {"threshold": 1500, "voucher_name": "$12 Hawker Treat", "partner_name": "Heartland Hawker"},
    ]
    eligible = [item for item in rewards if total_points >= item["threshold"]]
    return eligible[-1] if eligible else None


def maybe_issue_reward(user_id: int) -> dict | None:
    reward = reward_for_total_points(total_points(user_id))
    if not reward:
        return None
    existing = db.fetch_one(
        "SELECT id FROM vouchers WHERE user_id = ? AND voucher_name = ?",
        (user_id, reward["voucher_name"]),
    )
    if existing:
        return None
    db.execute(
        "INSERT INTO vouchers (user_id, voucher_name, partner_name, points_cost) VALUES (?, ?, ?, ?)",
        (user_id, reward["voucher_name"], reward["partner_name"], reward["threshold"]),
    )
    return reward


def show_vouchers(user_id: int) -> None:
    vouchers = [
        dict(row)
        for row in db.fetch_all(
            "SELECT * FROM vouchers WHERE user_id = ? ORDER BY issued_at DESC", (user_id,)
        )
    ]
    if not vouchers:
        st.info("No community-style demo vouchers issued yet. Earn more points through the games.")
        return
    st.dataframe(
        pd.DataFrame(vouchers)[["voucher_name", "partner_name", "status", "issued_at"]],
        use_container_width=True,
    )


def persist_game_session(
    user_id: int,
    game_name: str,
    raw_score: float,
    full_marks: float,
    duration_seconds: int,
    breakdown: dict[str, Any],
    interpretation: str,
) -> dict[str, Any]:
    percent_score = round((raw_score / full_marks) * 100, 1) if full_marks > 0 else 0.0
    points_awarded = int(round(raw_score))
    payload = {
        "raw_score": raw_score,
        "full_marks": full_marks,
        "percent_score": percent_score,
        "breakdown": breakdown,
        "interpretation": interpretation,
    }
    db.execute(
        """
        INSERT INTO game_sessions (user_id, game_name, score, points_awarded, duration_seconds, metadata_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            game_name,
            raw_score,
            points_awarded,
            duration_seconds,
            json.dumps(payload),
        ),
    )
    reward = maybe_issue_reward(user_id)
    return {
        "raw_score": raw_score,
        "full_marks": full_marks,
        "percent_score": percent_score,
        "points_awarded": points_awarded,
        "reward": reward,
    }


def parse_game_metadata(metadata_json: str | None) -> dict[str, Any]:
    if not metadata_json:
        return {}
    try:
        return json.loads(metadata_json)
    except Exception:
        return {}


def get_game_sessions(user_id: int) -> list[dict]:
    rows = db.fetch_all(
        "SELECT * FROM game_sessions WHERE user_id = ? ORDER BY id DESC",
        (user_id,),
    )
    out = []
    for row in rows:
        item = dict(row)
        item["parsed_meta"] = parse_game_metadata(item.get("metadata_json"))
        out.append(item)
    return out


def get_game_performance_summary(user_id: int) -> dict[str, Any]:
    sessions = get_game_sessions(user_id)
    if not sessions:
        return {
            "overall_avg_percent": None,
            "by_game": {},
            "memory_avg_percent": None,
            "latest_breakdowns": [],
            "interpretation": "No game data yet.",
        }

    by_game: dict[str, list[float]] = {}
    latest_breakdowns: list[dict[str, Any]] = []

    for session in sessions:
        meta = session["parsed_meta"]
        percent = meta.get("percent_score")
        if percent is None:
            continue
        by_game.setdefault(session["game_name"], []).append(float(percent))
        if len(latest_breakdowns) < 6:
            latest_breakdowns.append(
                {
                    "game": session["game_name"],
                    "percent_score": percent,
                    "interpretation": meta.get("interpretation", ""),
                    "breakdown": meta.get("breakdown", {}),
                    "played_at": session.get("played_at") or session.get("created_at") or "",
                }
            )

    game_avgs = {k: round(sum(v) / len(v), 1) for k, v in by_game.items() if v}
    all_scores = [score for scores in by_game.values() for score in scores]
    overall_avg = round(sum(all_scores) / len(all_scores), 1) if all_scores else None

    memory_games = ["emoji_memory_match", "sentence_recall"]
    memory_scores = [game_avgs[g] for g in memory_games if g in game_avgs]
    memory_avg = round(sum(memory_scores) / len(memory_scores), 1) if memory_scores else None

    interpretation = "Game performance is currently limited."
    if overall_avg is not None:
        if overall_avg >= 85:
            interpretation = "Game performance is strong overall."
        elif overall_avg >= 65:
            interpretation = "Game performance is fair with some room for monitoring."
        else:
            interpretation = "Game performance is weaker and may warrant closer monitoring."

    return {
        "overall_avg_percent": overall_avg,
        "by_game": game_avgs,
        "memory_avg_percent": memory_avg,
        "latest_breakdowns": latest_breakdowns,
        "interpretation": interpretation,
    }


def get_reminder_adherence_summary(patient_user_id: int) -> dict[str, Any]:
    reminders = get_reminders(patient_user_id, include_pending=True)
    approved = [r for r in reminders if int(r.get("is_approved", 0)) == 1]
    completed = [r for r in approved if r.get("status") == "completed"]
    adherence = round((len(completed) / len(approved)) * 100, 1) if approved else None
    return {
        "total": len(reminders),
        "approved": len(approved),
        "completed": len(completed),
        "adherence_percent": adherence,
    }


def monitoring_overlay(patient_user_id: int, base_result: dict[str, Any]) -> dict[str, Any]:
    game_summary = get_game_performance_summary(patient_user_id)
    reminder_summary = get_reminder_adherence_summary(patient_user_id)

    delta = 0.0
    notes = []
    extra_factors = []

    memory_avg = game_summary.get("memory_avg_percent")
    overall_avg = game_summary.get("overall_avg_percent")
    adherence = reminder_summary.get("adherence_percent")

    if memory_avg is not None:
        if memory_avg < 55:
            delta += 0.10
            notes.append("low memory-game performance")
            extra_factors.append({"label": "Lower memory-game scores", "direction": "risk_up", "weight": "medium"})
        elif memory_avg >= 85:
            delta -= 0.03
            notes.append("strong memory-game performance")
            extra_factors.append({"label": "Strong recent memory-game scores", "direction": "protective", "weight": "low"})

    if overall_avg is not None and overall_avg < 60:
        delta += 0.05
        notes.append("low overall cognitive-game performance")
        extra_factors.append({"label": "Lower overall game scores", "direction": "risk_up", "weight": "low"})

    if adherence is not None:
        if adherence < 50:
            delta += 0.06
            notes.append("low reminder completion")
            extra_factors.append({"label": "Lower reminder adherence", "direction": "risk_up", "weight": "low"})
        elif adherence >= 80:
            delta -= 0.02
            notes.append("good reminder completion")
            extra_factors.append({"label": "Good reminder adherence", "direction": "protective", "weight": "low"})

    composite_score = max(0.0, min(float(base_result["score"]) + delta, 0.99))
    level = "Low" if composite_score < 0.4 else "Moderate" if composite_score < 0.7 else "High"
    summary = base_result["summary"]

    if notes:
        summary += " Monitoring overlay: " + ", ".join(notes) + "."

    return {
        "score": round(composite_score, 3),
        "level": level,
        "summary": summary,
        "factors": base_result["factors"] + extra_factors,
        "engine": f"{base_result['engine']}+monitoring",
        "overlay_delta": round(delta, 3),
        "game_summary": game_summary,
        "reminder_summary": reminder_summary,
    }


def render_login() -> None:
    inject_memoria_theme()
    st.title(APP_TITLE)
    st.caption(APP_TAGLINE)
    st.warning("Due to GitHub storage constraints, the listed raw EEG/MRI/Behavioural datasets are excluded from the repository, though their use is still supported by the app.")
    st.info("Demo accounts: doctor_demo / Doctor@123, caregiver_demo / Caregiver@123, patient_demo / Patient@123")

    language = st.selectbox(
        t("language", st.session_state.get("language", "en")),
        list(LANGUAGE_LABELS.keys()),
        index=list(LANGUAGE_LABELS.keys()).index(st.session_state.get("language", "en")),
        format_func=lambda code: LANGUAGE_LABELS[code],
    )
    st.session_state["language"] = language

    with st.form("login_form"):
        username = st.text_input(t("username", language))
        password = st.text_input(t("password", language), type="password")
        submitted = st.form_submit_button(t("login", language))
        if submitted:
            user = authenticate(username, password)
            if user:
                st.session_state["user_id"] = user["id"]
                st.session_state["language"] = language
                st.success(f"{t('welcome', language)}, {user['full_name']}")
                st.rerun()
            else:
                st.error(t("invalid_login", language))

def logout() -> None:
    for key in [
        "user_id",
        "game_state",
        "kopi_game_state",
        "sentence_game_state",
    ]:
        st.session_state.pop(key, None)
    st.rerun()


def render_sidebar(user: dict) -> None:
    with st.sidebar:
        st.subheader(APP_TITLE)
        st.caption(user["full_name"])
        st.caption(f"{ui('role')}: {ROLE_LABELS.get(user['role'], user['role'])}")
        language = st.selectbox(
            ui("language"),
            list(LANGUAGE_LABELS.keys()),
            index=list(LANGUAGE_LABELS.keys()).index(st.session_state.get("language", "en")),
            format_func=lambda code: LANGUAGE_LABELS[code],
        )
        st.session_state["language"] = language
        st.button(t("logout", language), on_click=logout, use_container_width=True)

def save_prediction(patient_user_id: int, actor_user_id: int, result: dict) -> None:
    db.execute(
        """
        INSERT INTO model_predictions (patient_user_id, actor_user_id, score, level, summary, factors_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            patient_user_id,
            actor_user_id,
            result["score"],
            result["level"],
            result["summary"],
            json.dumps(result["factors"]),
        ),
    )
    db.upsert_elder_profile(
        patient_user_id,
        last_risk_score=result["score"],
        last_risk_level=result["level"],
        last_risk_summary=result["summary"],
    )
    if result["score"] >= 0.65:
        db.execute(
            "INSERT INTO alerts (patient_user_id, severity, title, body, source) VALUES (?, ?, ?, ?, ?)",
            (
                patient_user_id,
                "high" if result["score"] >= 0.8 else "medium",
                "Updated dementia concern score",
                result["summary"],
                "risk_model",
            ),
        )


def render_metadata_block(title: str, meta: dict) -> None:
    st.markdown(f"### {title}")
    if not meta:
        st.info("No training metadata found yet.")
        return

    summary_keys = [k for k in ["modality", "best_model", "best_auc", "row_count", "target_column", "mode", "training_mode"] if k in meta]
    if summary_keys:
        summary_df = pd.DataFrame([{k: meta.get(k) for k in summary_keys}])
        st.dataframe(summary_df, use_container_width=True)

    chart_paths = meta.get("chart_paths", {})
    if chart_paths:
        for label, path in chart_paths.items():
            if path and Path(path).exists():
                st.write(f"**{label.replace('_', ' ').title()}**")
                st.image(path, use_container_width=True)

    if meta.get("candidates"):
        rows = []
        for model_name, metrics in meta["candidates"].items():
            rows.append(
                {
                    "model": model_name,
                    "auc": metrics.get("auc"),
                    "accuracy": metrics.get("accuracy"),
                }
            )
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    with st.expander("Raw metadata JSON"):
        st.json(meta)


def render_patient_summary(patient: dict) -> None:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(ui("risk_score"), f"{float(patient.get('last_risk_score') or 0):.2f}")
    with col2:
        st.metric(ui("risk_level"), patient.get("last_risk_level") or "Unknown")
    with col3:
        st.metric(ui("points_earned"), total_points(patient["id"]))
    if patient.get("last_risk_summary"):
        st.info(patient["last_risk_summary"])

def render_alerts(patient_user_id: int) -> None:
    alerts = get_alerts(patient_user_id)
    if not alerts:
        st.success(ui("no_current_alerts"))
        return
    for alert in alerts:
        severity = alert["severity"].lower()
        if severity == "high":
            st.error(f"{alert['title']}: {alert['body']}")
        elif severity == "medium":
            st.warning(f"{alert['title']}: {alert['body']}")
        else:
            st.info(f"{alert['title']}: {alert['body']}")

def render_messages_panel(current_user: dict, counterpart: dict) -> None:
    thread = get_messages(current_user["id"], counterpart["id"])
    if not thread:
        st.caption("No messages yet.")
    else:
        for message in thread:
            speaker = "You" if message["sender_user_id"] == current_user["id"] else message["sender_name"]
            with st.chat_message("assistant" if speaker != "You" else "user"):
                st.markdown(f"**{speaker}**  ")
                st.write(message["body"])
                st.caption(message["created_at"])
    body = st.chat_input(f"Message {counterpart['full_name']}")
    if body:
        send_message(current_user["id"], counterpart["id"], body)
        st.rerun()


def render_education_quiz(user: dict) -> None:
    st.subheader(t("education", st.session_state["language"]))
    videos = [dict(row) for row in db.fetch_all("SELECT * FROM education_videos ORDER BY id ASC")]
    for video in videos:
        with st.expander(video["title"]):
            st.write(video["description"])
            if video.get("url"):
                st.video(video["url"])

    questions = [dict(row) for row in db.fetch_all("SELECT * FROM quiz_questions ORDER BY id ASC")]
    with st.form("education_quiz"):
        answers = {}
        for question in questions:
            options = {
                "A": question["option_a"],
                "B": question["option_b"],
                "C": question["option_c"],
                "D": question["option_d"],
            }
            answers[str(question["id"])] = st.radio(
                question["question"],
                list(options.keys()),
                format_func=lambda key, options=options: f"{key}. {options[key]}",
                horizontal=False,
                key=f"quiz_{question['id']}",
            )
        submitted = st.form_submit_button("Submit quiz")
        if submitted:
            score = sum(1 for question in questions if answers[str(question["id"])] == question["correct_option"])
            db.execute(
                "INSERT INTO quiz_attempts (user_id, score, total, answers_json) VALUES (?, ?, ?, ?)",
                (user["id"], score, len(questions), json.dumps(answers)),
            )
            st.success(f"Score: {score}/{len(questions)}")
            for question in questions:
                st.caption(f"{question['question']} — rationale: {question['rationale']}")


def render_reminders_panel(viewer: dict, patient: dict, patient_view: bool = False) -> None:
    if not patient_view and viewer["role"] == "doctor":
        st.subheader(ui("create_reminder"))
        with st.form(f"create_reminder_{viewer['id']}_{patient['id']}"):
            title = st.text_input(ui("reminder_title"))
            description = st.text_area(ui("reminder_details"))
            due_date = st.text_input(ui("due_date"), placeholder="e.g. Today 6pm or 2026-04-30 18:00")
            submitted = st.form_submit_button(ui("save_reminder"))
            if submitted:
                if not title.strip():
                    st.error("Reminder title is required.")
                else:
                    create_reminder(
                        patient_user_id=patient["id"],
                        creator_user_id=viewer["id"],
                        creator_role=viewer["role"],
                        title=title.strip(),
                        description=description.strip(),
                        due_date=due_date.strip(),
                    )
                    st.success("Reminder created and made active.")
                    st.rerun()
    elif not patient_view and viewer["role"] == "caregiver":
        st.info(ui("doctor_only_reminders"))

    reminders = get_reminders(patient["id"], include_pending=not patient_view)
    if patient_view:
        reminders = [r for r in reminders if int(r.get("is_approved", 0)) == 1 and r.get("status") != "completed"]

    if not reminders:
        st.info(ui("no_reminders"))
        return

    for reminder in reminders:
        status = reminder.get("status", "active")
        with st.expander(f"{reminder['title']} — {status}"):
            st.write(reminder.get("description") or "No details provided.")
            if reminder.get("due_date"):
                st.caption(f"Due: {reminder['due_date']}")
            st.caption(f"{ui('created_by')}: {reminder.get('created_by_name', 'Unknown')} | {reminder.get('created_at', '')}")

            if not patient_view:
                if viewer["role"] == "doctor" and int(reminder.get("is_approved", 0)) == 0:
                    if st.button("Approve reminder", key=f"approve_reminder_{reminder['id']}"):
                        approve_reminder(reminder["id"], viewer["id"])
                        st.success("Reminder approved.")
                        st.rerun()

                if viewer["role"] in {"doctor", "caregiver"} and reminder.get("status") != "completed" and int(reminder.get("is_approved", 0)) == 1:
                    if st.button(ui("mark_complete"), key=f"complete_reminder_{reminder['id']}"):
                        complete_reminder(reminder["id"], viewer["id"])
                        st.success("Reminder marked complete.")
                        st.rerun()

            if reminder.get("approved_by_name"):
                st.caption(f"{ui('approved_by')}: {reminder['approved_by_name']}")
            if reminder.get("completed_by_name"):
                st.caption(f"{ui('completed_by')}: {reminder['completed_by_name']}")

def render_game_performance_panel(patient_user_id: int) -> None:
    summary = get_game_performance_summary(patient_user_id)
    reminder_summary = get_reminder_adherence_summary(patient_user_id)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Overall game average", "-" if summary["overall_avg_percent"] is None else f"{summary['overall_avg_percent']}%")
    with col2:
        st.metric("Memory-game average", "-" if summary["memory_avg_percent"] is None else f"{summary['memory_avg_percent']}%")
    with col3:
        st.metric("Reminder adherence", "-" if reminder_summary["adherence_percent"] is None else f"{reminder_summary['adherence_percent']}%")

    st.caption(summary["interpretation"])

    if summary["by_game"]:
        game_rows = [{"game": k, "average_percent": v} for k, v in summary["by_game"].items()]
        st.dataframe(pd.DataFrame(game_rows), use_container_width=True)

    if summary["latest_breakdowns"]:
        st.markdown("Latest game breakdowns")
        for item in summary["latest_breakdowns"]:
            with st.expander(f"{item['game']} — {item['percent_score']}%"):
                st.write(item["interpretation"])
                st.json(item["breakdown"])



def render_patient_home(user: dict) -> None:
    inject_memoria_theme()
    reminders = get_reminders(user["id"], include_pending=False)
    active_reminders = [r for r in reminders if r.get("status") != "completed"]
    quote = random.choice(QUOTE_BANK)
    greeting = ui(greeting_key())

    st.markdown(
        f"""
        <div class="memoria-hero">
            <p>☀️ {greeting.upper()}</p>
            <h1>{user['full_name']}!</h1>
            <p>{ui('beautiful_day')}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 1])
    with left:
        st.markdown(f"### 🕒 {ui('today_focus')}")
        if active_reminders:
            for reminder in active_reminders[:3]:
                icon = "💊" if "med" in reminder["title"].lower() else "✅"
                st.markdown(
                    f"""
                    <div class="memoria-card">
                        <div style="display:flex; gap:1rem; align-items:center;">
                            <div class="memoria-icon-bubble">{icon}</div>
                            <div>
                                <div style="color:#777; font-weight:800; text-transform:uppercase;">{ui('doctor_reminders')}</div>
                                <div style="font-size:1.35rem; font-weight:900;">{reminder['title']}</div>
                                <div style="color:#2f8f63; font-weight:800;">{reminder.get('due_date') or ''}</div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info(ui("no_reminders"))

    with right:
        st.markdown("### ✨ Memory quote")
        st.markdown(
            f"""
            <div class="memoria-card-soft">
                <h3 style="margin-top:0;">“{quote}”</h3>
                <p>{APP_TAGLINE}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.metric(ui("points_earned"), total_points(user["id"]))


def render_community_tab(user: dict) -> None:
    st.markdown(f"# {ui('community_support')}")
    st.markdown(f"### 📞 {ui('talk_volunteer')}")
    st.markdown(
        f"""
        <div class="memoria-card-soft">
            <div style="display:flex; gap:1.3rem; align-items:center;">
                <div class="memoria-icon-bubble">👋</div>
                <div>
                    <h2 style="margin-bottom:0.2rem; color:#332c87;">{ui('friendly_chat')}</h2>
                    <p style="font-size:1.1rem; color:#3d35b6;">{ui('friendly_chat_desc')}</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.button(f"☎️ {ui('call_now')}", use_container_width=True)

    groups = get_community_groups(user["id"])
    joined_groups = [g for g in groups if g.get("membership_id")]
    unjoined_groups = [g for g in groups if not g.get("membership_id")]

    st.markdown(f"### 👥 {ui('interest_groups')}")
    for group in unjoined_groups:
        c1, c2 = st.columns([4, 1])
        with c1:
            st.markdown(
                f"""
                <div class="memoria-community-card">
                    <div class="memoria-icon-bubble">{group.get('icon') or '💬'}</div>
                    <div>
                        <h3 style="margin:0;">{group['name']}</h3>
                        <p style="margin:0.25rem 0; color:#666;">{group.get('description') or ''}</p>
                        <p style="margin:0; color:#777;">{group.get('active_members', 0)} members active</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            if st.button(ui("join_chat"), key=f"join_group_{group['id']}", use_container_width=True):
                join_community(user["id"], group["id"])
                st.session_state["active_group_id"] = group["id"]
                st.success(ui("joined_group"))
                st.rerun()

    if joined_groups:
        st.markdown(f"### 💬 {ui('group_chat')}")
        joined_map = {f"{g.get('icon') or '💬'} {g['name']}": g for g in joined_groups}
        default_index = 0
        active_group_id = st.session_state.get("active_group_id")
        labels = list(joined_map.keys())
        if active_group_id:
            for idx, label in enumerate(labels):
                if joined_map[label]["id"] == active_group_id:
                    default_index = idx
                    break
        selected_label = st.selectbox(ui("group_chat"), labels, index=default_index)
        selected_group = joined_map[selected_label]
        st.session_state["active_group_id"] = selected_group["id"]

        st.markdown(
            f"""
            <div class="memoria-card">
                <div style="display:flex; align-items:center; gap:1rem;">
                    <div class="memoria-icon-bubble">{selected_group.get('icon') or '💬'}</div>
                    <div>
                        <h3 style="margin:0;">{selected_group['name']}</h3>
                        <p style="margin:0.25rem 0; color:#666;">{selected_group.get('description') or ''}</p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        messages = get_group_messages(selected_group["id"])
        if not messages:
            st.caption(ui("no_group_messages"))
        else:
            for message in messages:
                with st.chat_message("user" if message["sender_user_id"] == user["id"] else "assistant"):
                    speaker = "You" if message["sender_user_id"] == user["id"] else message["sender_name"]
                    st.markdown(f"**{speaker}**")
                    st.write(message["body"])
                    st.caption(message["created_at"])

        group_body = st.chat_input(ui("say_hello_group"), key=f"group_chat_input_{selected_group['id']}")
        if group_body:
            send_group_message(selected_group["id"], user["id"], group_body)
            st.rerun()



def render_caregiver_journal_panel(caregiver: dict, patient: dict) -> None:
    st.subheader(ui("caregiver_journal"))
    with st.expander(f"ℹ️ {ui('journal_guidance_title')}"):
        st.write(ui("journal_guidance"))
        st.markdown("""
        **Template:**
        - Change noticed:
        - Example:
        - When/how often:
        - Doctor action needed:
        """)

    with st.form(f"journal_form_{caregiver['id']}_{patient['id']}"):
        mood = st.selectbox("Observed condition", ["Stable", "Improved", "More confused", "More withdrawn", "Needs review"])
        entry = st.text_area(ui("journal_entry"), height=160, placeholder="Example: Mdm Tan seemed more forgetful after lunch today. She asked twice where her keys were within 20 minutes. No falls or medication refusal. Please advise if we should monitor this daily.")
        submitted = st.form_submit_button(ui("send_to_doctor"))
        if submitted:
            if not entry.strip():
                st.error("Journal entry is required.")
            else:
                save_caregiver_journal(patient["id"], caregiver["id"], entry.strip(), mood)
                st.success(ui("journal_saved"))
                st.rerun()

    entries = get_caregiver_journal_entries(patient["id"])
    st.markdown(f"### {ui('recent_journal_entries')}")
    if not entries:
        st.info("No journal entries yet.")
    for item in entries[:8]:
        with st.expander(f"{item['caregiver_name']} · {item.get('mood') or 'Update'} · {item['created_at']}"):
            st.write(item["entry"])

def render_patient_page(user: dict) -> None:
    patient = get_user(user["id"])
    patient.update(get_profile(user["id"]))
    st.title(ui("patient_dashboard"))

    tabs = st.tabs(
        [
            ui("home"),
            ui("game"),
            ui("reminders"),
            ui("community"),
            ui("messages"),
            ui("rewards"),
        ]
    )

    with tabs[0]:
        render_patient_home(user)

    with tabs[1]:
        render_game_panel(user)

    with tabs[2]:
        st.subheader(ui("doctor_reminders"))
        render_reminders_panel(user, user, patient_view=True)

    with tabs[3]:
        render_community_tab(user)

    with tabs[4]:
        caregiver = get_primary_caregiver_for_patient(user["id"])
        if caregiver:
            render_messages_panel(user, caregiver)
        else:
            st.info("No caregiver linked yet.")

    with tabs[5]:
        show_vouchers(user["id"])

def render_caregiver_page(user: dict) -> None:
    st.title(ui("caregiver_dashboard"))
    patients = get_patients_for_caregiver(user["id"])
    if not patients:
        st.warning("No linked patients found.")
        return

    patient_names = {f"{p['full_name']} ({p.get('last_risk_level', 'Unknown')})": p for p in patients}
    selected_label = st.selectbox("Linked patient", list(patient_names))
    patient = patient_names[selected_label]
    render_patient_summary(patient)

    tabs = st.tabs(
        [
            ui("overview"),
            ui("ai_screener"),
            ui("education"),
            ui("caregiver_journal"),
            ui("messages"),
            ui("activities"),
            ui("resources"),
        ]
    )

    with tabs[0]:
        render_alerts(patient["id"])
        st.markdown("### Community matches")
        matches = suggest_activity_matches(patient["id"])
        for item in matches[:5]:
            st.write(f"{item['title']} — {item['match_reason']} — {item['starts_at']}")
        st.markdown("### Confidential notes")
        for note in get_notes(patient["id"]):
            st.info(f"{note['note']}\n\n*{note['author_name']} · {note['created_at']}*")
        st.markdown("### Game and adherence summary")
        render_game_performance_panel(patient["id"])
        st.markdown(f"### {ui('recent_journal_entries')}")
        journal_entries = get_caregiver_journal_entries(patient["id"])
        if journal_entries:
            for item in journal_entries[:3]:
                st.info(f"{item['entry']}\n\n*{item['caregiver_name']} · {item.get('mood') or 'Update'} · {item['created_at']}*")
        else:
            st.caption("No caregiver journal entries yet.")

    with tabs[1]:
        render_risk_screener(user, patient)

    with tabs[2]:
        render_education_quiz(user)

    with tabs[3]:
        render_caregiver_journal_panel(user, patient)

    with tabs[4]:
        render_messages_panel(user, patient)

    with tabs[5]:
        render_activity_browser(user, registrant_user_id=patient["id"])

    with tabs[6]:
        for resource in SINGAPORE_RESOURCES:
            st.write(f"{resource['name']} — {resource['type']}")
            st.caption(resource["description"])

def render_doctor_page(user: dict) -> None:
    st.title(ui("clinical_dashboard"))
    patients = get_all_patients()
    if not patients:
        st.warning("No patient records found.")
        return
    patient_map = {p["full_name"]: p for p in patients}
    selected_name = st.selectbox("Patient", list(patient_map))
    patient = patient_map[selected_name]

    tabs = st.tabs(
        [
            ui("overview"),
            ui("clinical_notes"),
            ui("reminders"),
            ui("dataset_hub"),
            ui("predictions"),
        ]
    )

    with tabs[0]:
        render_patient_summary(patient)
        render_alerts(patient["id"])
        st.markdown("### Profile")
        st.dataframe(pd.DataFrame([patient]), use_container_width=True)
        st.markdown("### Game and adherence summary")
        render_game_performance_panel(patient["id"])

    with tabs[1]:
        st.markdown(f"### {ui('recent_journal_entries')}")
        journal_entries = get_caregiver_journal_entries(patient["id"])
        if journal_entries:
            for item in journal_entries:
                with st.expander(f"{item['caregiver_name']} · {item.get('mood') or 'Update'} · {item['created_at']}"):
                    st.write(item["entry"])
        else:
            st.info("No caregiver journal entries yet.")

        st.markdown("### Doctor confidential notes")
        notes = get_notes(patient["id"])
        for note in notes:
            st.info(f"{note['note']}\n\n*{note['author_name']} · {note['created_at']}*")
        new_note = st.text_area("Add confidential note")
        if st.button("Save note"):
            if new_note.strip():
                db.execute(
                    "INSERT INTO confidential_notes (patient_user_id, author_user_id, note) VALUES (?, ?, ?)",
                    (patient["id"], user["id"], new_note.strip()),
                )
                st.success("Note saved.")
                st.rerun()

    with tabs[2]:
        render_reminders_panel(user, patient, patient_view=False)

    with tabs[3]:
        st.caption("Doctor-only training and evaluation hub for tabular, behavioural-sheet, EEG, and imaging datasets.")
        hub_tabs = st.tabs(
            [
                "Tabular",
                "Behavioural Sheets",
                "EEG",
                "Imaging",
                "Results Dashboard",
            ]
        )

        with hub_tabs[0]:
            st.markdown("### Tabular model training")
            st.write(f"Default folder: `{TABULAR_DIR}`")
            uploaded = st.file_uploader("Upload tabular CSV for training", type=["csv"], key="doctor_tabular_upload")
            csv_path = None
            if uploaded is not None:
                destination = EXPORT_DIR / uploaded.name
                destination.write_bytes(uploaded.getbuffer())
                csv_path = destination

            default_csv_options = [p.name for p in TABULAR_DIR.glob("*.csv")]
            selected_default_csv = None
            if default_csv_options:
                selected_default_csv = st.selectbox("Or choose a CSV from data/raw/tabular", default_csv_options)

            if st.button("Train tabular model", use_container_width=True, key="train_tabular_btn"):
                with st.spinner("Training tabular model..."):
                    if csv_path is not None:
                        metadata = train_and_save_model(csv_path)
                    elif selected_default_csv:
                        metadata = train_and_save_model(TABULAR_DIR / selected_default_csv)
                    else:
                        st.error("Provide or select a CSV first.")
                        metadata = None
                if metadata:
                    st.success("Tabular model trained.")
                    render_metadata_block("Tabular Results", metadata)

            render_metadata_block("Current Tabular Metadata", load_tabular_metadata())

        with hub_tabs[1]:
            st.markdown("### Behavioural multi-sheet training")
            st.write(f"Expected folder: `{ADDRESSO_DIR}`")
            st.caption("Uses Activity.csv, Demographics.csv, Labels.csv, Physiology.csv, and Sleep.csv if present.")
            if st.button("Train behavioural-sheet bundle", use_container_width=True, key="train_addresso_btn"):
                with st.spinner("Training behavioural-sheet model..."):
                    try:
                        metadata = train_addresso_bundle(ADDRESSO_DIR)
                        st.success("Behavioural-sheet model completed.")
                        render_metadata_block("Behavioural-Sheet Results", metadata)
                    except Exception as exc:
                        st.error(str(exc))

            render_metadata_block("Current Behavioural-Sheet Metadata", load_addresso_metadata())

        with hub_tabs[2]:
            st.markdown("### EEG model training")
            st.write(f"Expected folder: `{EEG_DIR}`")
            st.caption("Attempts signal-feature extraction if EEG files are readable; otherwise falls back to exploratory analysis.")
            max_files = st.slider("Max EEG files to scan", 10, 100, 40, key="eeg_max_files")
            if st.button("Train EEG model", use_container_width=True, key="train_eeg_btn"):
                with st.spinner("Training EEG model..."):
                    try:
                        metadata = train_eeg_bundle(EEG_DIR, max_files=max_files)
                        st.success("EEG model completed.")
                        render_metadata_block("EEG Results", metadata)
                    except Exception as exc:
                        st.error(str(exc))

            render_metadata_block("Current EEG Metadata", load_eeg_metadata())

        with hub_tabs[3]:
            st.markdown("### Imaging model training")
            st.write(f"Expected folder: `{IMAGING_DIR}`")
            st.caption("Uses handcrafted image features from MildDemented, VeryMildDemented, ModerateDemented, and NonDemented folders.")
            max_per_class = st.slider("Max images per class", 50, 500, 250, 25, key="img_max_per_class")
            if st.button("Train imaging model", use_container_width=True, key="train_img_btn"):
                with st.spinner("Training imaging model..."):
                    try:
                        metadata = train_imaging_bundle(IMAGING_DIR, max_per_class=max_per_class)
                        st.success("Imaging model completed.")
                        render_metadata_block("Imaging Results", metadata)
                    except Exception as exc:
                        st.error(str(exc))

            render_metadata_block("Current Imaging Metadata", load_imaging_metadata())

        with hub_tabs[4]:
            render_metadata_block("Core Live Screener Model", load_training_metadata())
            render_metadata_block("Tabular", load_tabular_metadata())
            render_metadata_block("Behavioural Sheets", load_addresso_metadata())
            render_metadata_block("EEG", load_eeg_metadata())
            render_metadata_block("Imaging", load_imaging_metadata())

    with tabs[4]:
        st.subheader("Run updated prediction")
        render_risk_screener(user, patient)
        history = [
            dict(row)
            for row in db.fetch_all(
                "SELECT * FROM model_predictions WHERE patient_user_id = ? ORDER BY created_at DESC",
                (patient["id"],),
            )
        ]
        if history:
            frame = pd.DataFrame(history)
            st.dataframe(frame[["score", "level", "summary", "created_at"]], use_container_width=True)


def render_risk_screener(actor: dict, patient: dict) -> None:
    profile = get_profile(patient["id"])
    default_age = int(profile.get("age") or 74)
    with st.form(f"risk_form_{patient['id']}_{actor['id']}"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Age", 55, 95, default_age)
            mmse = st.slider("MMSE-like score", 0, 30, 24)
            sleep_quality = st.slider("Sleep quality", 0, 10, 6)
            physical_activity = st.slider("Physical activity", 0, 10, 5)
            adl = st.slider("Functional / ADL score", 0, 10, 7)
        with col2:
            memory = st.select_slider("Memory complaints", options=[0, 1], value=1)
            confusion = st.select_slider("Confusion episodes", options=[0, 1], value=0)
            disorientation = st.select_slider("Disorientation", options=[0, 1], value=0)
            forgetfulness = st.select_slider("Forgetfulness", options=[0, 1], value=1)
            behavioral = st.select_slider("Behavioral problems", options=[0, 1], value=0)
            hypertension = st.select_slider("Hypertension", options=[0, 1], value=1)
            diabetes = st.select_slider("Diabetes", options=[0, 1], value=1)
            depression = st.select_slider("Low mood / depression", options=[0, 1], value=0)
            difficulty_tasks = st.select_slider("Difficulty completing tasks", options=[0, 1], value=0)
        submitted = st.form_submit_button("Run AI screening")
        if submitted:
            profile_dict = {
                "Age": age,
                "MMSE": mmse,
                "SleepQuality": sleep_quality,
                "PhysicalActivity": physical_activity,
                "ADL": adl,
                "MemoryComplaints": memory,
                "Confusion": confusion,
                "Disorientation": disorientation,
                "Forgetfulness": forgetfulness,
                "BehavioralProblems": behavioral,
                "Hypertension": hypertension,
                "Diabetes": diabetes,
                "Depression": depression,
                "DifficultyCompletingTasks": difficulty_tasks,
            }
            base_result = predict_profile(profile_dict)
            result = monitoring_overlay(patient["id"], base_result)
            save_prediction(patient["id"], actor["id"], result)

            st.success(result["summary"])
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Composite score", f"{result['score']:.2f}")
            with col_b:
                st.metric("Risk level", result["level"])
            with col_c:
                st.metric("Monitoring overlay", f"{result['overlay_delta']:+.2f}")

            st.write("Key factors")
            st.dataframe(pd.DataFrame(result["factors"]), use_container_width=True)

            st.write("Monitoring context")
            overlay_info = pd.DataFrame(
                [
                    {
                        "memory_avg_percent": result["game_summary"].get("memory_avg_percent"),
                        "overall_game_avg_percent": result["game_summary"].get("overall_avg_percent"),
                        "reminder_adherence_percent": result["reminder_summary"].get("adherence_percent"),
                        "engine": result["engine"],
                    }
                ]
            )
            st.dataframe(overlay_info, use_container_width=True)


def render_activity_browser(user: dict, registrant_user_id: int) -> None:
    activities = [dict(row) for row in db.fetch_all("SELECT * FROM activities ORDER BY starts_at ASC")]
    for activity in activities:
        registered = db.fetch_one(
            "SELECT id FROM activity_registrations WHERE activity_id = ? AND user_id = ?",
            (activity["id"], registrant_user_id),
        )
        with st.expander(f"{activity['title']} — {activity['starts_at']}"):
            st.write(activity["description"])
            st.caption(f"{activity['category']} · {activity['location']} · tags: {activity['interest_tags']}")
            if registered:
                st.success("Registered")
            else:
                if st.button(f"Register for {activity['title']}", key=f"register_{activity['id']}_{registrant_user_id}"):
                    db.execute(
                        "INSERT OR IGNORE INTO activity_registrations (activity_id, user_id) VALUES (?, ?)",
                        (activity["id"], registrant_user_id),
                    )
                    st.rerun()


def render_sentence_recall_game(user: dict) -> None:
    st.subheader(ui("sentence_recall"))
    st.caption(ui("sentence_instruction"))

    if "sentence_game_state" not in st.session_state:
        st.session_state["sentence_game_state"] = {
            "sentence": None,
            "phase": "idle",
            "started_at": None,
            "selected": [],
            "word_bank": [],
            "used_indices": [],
        }

    state = st.session_state["sentence_game_state"]

    if state["phase"] == "idle":
        if st.button(ui("start_sentence"), key="start_sentence_recall", use_container_width=True):
            sentence = random.choice(SENTENCE_RECALL_PROMPTS)
            target_words = sentence.replace(".", "").replace(",", "").split()
            distractors = random.sample(SENTENCE_DISTRACTOR_WORDS, min(8, len(SENTENCE_DISTRACTOR_WORDS)))
            word_bank = target_words + distractors
            random.shuffle(word_bank)
            st.session_state["sentence_game_state"] = {
                "sentence": sentence,
                "phase": "showing",
                "started_at": time.time(),
                "selected": [],
                "word_bank": word_bank,
                "used_indices": [],
            }
            st.rerun()

    elif state["phase"] == "showing":
        placeholder = st.empty()
        sentence = state["sentence"]
        for remaining in range(10, 0, -1):
            placeholder.markdown(
                f"""
                <div style="padding: 18px; border:1px solid #dce6ff; border-radius:18px; background:#f7fbff; font-size:26px;">
                {sentence}
                <br><br>
                <span style="font-size:18px;">{ui('memorise_sentence')}. {ui('time_left')}: {remaining}s</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            time.sleep(1)
        st.session_state["sentence_game_state"]["phase"] = "building"
        st.rerun()

    elif state["phase"] == "building":
        sentence = state["sentence"]
        selected = state.get("selected", [])
        word_bank = state.get("word_bank", [])

        st.markdown(f"### {ui('build_sentence')}")
        st.markdown(f"**{ui('selected_words')}:**")
        if selected:
            st.markdown(" ".join(f"<span class='word-chip'>{w}</span>" for w in selected), unsafe_allow_html=True)
        else:
            st.caption("Choose words below.")

        cols = st.columns(3)
        used_indices = state.get("used_indices", [])
        if "used_indices" not in state:
            state["used_indices"] = []
            used_indices = []

        # Show every available word block exactly once, including distractors.
        # Disable a block once selected so the sentence can be completed in one pass
        # without duplicated chips appearing in the bank.
        for idx, word in enumerate(word_bank):
            with cols[idx % len(cols)]:
                disabled = idx in used_indices
                if st.button(word, key=f"sentence_word_{idx}_{word}", disabled=disabled, use_container_width=True):
                    state["selected"].append(word)
                    state["used_indices"].append(idx)
                    st.rerun()

        c1, c2 = st.columns(2)
        with c1:
            if st.button(ui("clear_selection"), key="clear_sentence_blocks", use_container_width=True):
                state["selected"] = []
                state["used_indices"] = []
                st.rerun()
        with c2:
            if st.button(ui("submit_sentence"), key="submit_sentence_blocks", use_container_width=True):
                target_words = sentence.replace(".", "").replace(",", "").lower().split()
                response_words = [w.lower() for w in selected]
                response_sentence = " ".join(response_words)
                target_sentence = " ".join(target_words)
                sequence_similarity = SequenceMatcher(None, target_sentence, response_sentence).ratio()
                correct_positions = sum(
                    1 for i, word in enumerate(response_words[: len(target_words)])
                    if i < len(target_words) and word == target_words[i]
                )
                order_accuracy = correct_positions / max(1, len(target_words))
                target_set = set(target_words)
                response_set = set(response_words)
                word_coverage = len(target_set & response_set) / max(1, len(target_set))
                penalty = max(0, len(response_words) - len(target_words)) * 0.03

                raw_score = round(max(0, min(100, (word_coverage * 45 + order_accuracy * 40 + sequence_similarity * 15) - penalty * 100)), 1)
                interpretation = (
                    "Strong sentence recall."
                    if raw_score >= 85
                    else "Fair sentence recall with some omissions."
                    if raw_score >= 65
                    else "Lower sentence recall performance."
                )

                result = persist_game_session(
                    user_id=user["id"],
                    game_name="sentence_recall",
                    raw_score=raw_score,
                    full_marks=100.0,
                    duration_seconds=20,
                    breakdown={
                        "target_sentence": sentence,
                        "selected_words": selected,
                        "sequence_similarity": round(sequence_similarity, 3),
                        "word_coverage": round(word_coverage, 3),
                        "order_accuracy": round(order_accuracy, 3),
                    },
                    interpretation=interpretation,
                )

                st.success(f"Score: {raw_score}/100 ({result['percent_score']}%)")
                st.caption(interpretation)
                if result["reward"]:
                    st.balloons()
                    st.success(f"New reward unlocked: {result['reward']['voucher_name']} from {result['reward']['partner_name']}")

                st.session_state["sentence_game_state"] = {
                    "sentence": None,
                    "phase": "idle",
                    "started_at": None,
                    "selected": [],
                    "word_bank": [],
                    "used_indices": [],
                }

def render_game_panel(user: dict) -> None:
    game_tabs = st.tabs([ui("emoji_match"), ui("drink_game"), ui("sentence_recall")])

    with game_tabs[0]:
        st.subheader(ui("emoji_match"))
        st.caption(ui("emoji_game_caption"))

        if "game_state" not in st.session_state:
            st.session_state["game_state"] = {
                "board": build_memory_board(seed=int(time.time())),
                "selected": [],
                "matched": [],
                "moves": 0,
                "score": 0,
                "started_at": time.time(),
                "last_feedback": ui("pick_two"),
            }

        state = st.session_state["game_state"]

        st.caption(ui("tap_to_reveal"))

        if st.button(ui("start_new_game"), key="emoji_new_game"):
            st.session_state["game_state"] = {
                "board": build_memory_board(seed=int(time.time())),
                "selected": [],
                "matched": [],
                "moves": 0,
                "score": 0,
                "started_at": time.time(),
                "last_feedback": ui("new_game_started"),
            }
            st.session_state.pop("game_saved", None)
            st.rerun()

        st.info(state["last_feedback"])
        board = state["board"]

        left_gap, board_area, right_gap = st.columns([1, 2.2, 1])

        with board_area:
            cols = st.columns(3)
            st.markdown('<div class="emoji-grid">', unsafe_allow_html=True)

            for idx, tile in enumerate(board):
                visible = idx in state["matched"] or idx in state["selected"]
                shown = tile if visible else "❔"

                with cols[idx % len(cols)]:
                    disabled = idx in state["matched"] or idx in state["selected"]

                    clicked = st.button(
                        shown,
                        key=f"emoji_tile_{idx}",
                        disabled=disabled,
                        use_container_width=True,
                    )

                    if clicked:
                        if idx not in state["selected"] and idx not in state["matched"]:
                            state["selected"].append(idx)

                            if len(state["selected"]) == 2:
                                a, b = state["selected"]
                                state["moves"] += 1

                                if board[a] == board[b]:
                                    state["matched"].extend([a, b])
                                    state["score"] += 25
                                    state["last_feedback"] = f"{ui('match_found')}: {board[a]}"
                                else:
                                    state["last_feedback"] = f"{ui('no_match')}. You revealed {board[a]} and {board[b]}."

                                state["selected"] = []

                            st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)
            progress = len(state["matched"]) / len(board)
            st.progress(progress)
            st.write(f"{ui('moves')}: {state['moves']} | {ui('match_score')}: {state['score']}")

            if len(state["matched"]) == len(board):
                duration = int(time.time() - state["started_at"])
                raw_score = max(0.0, min(100.0, 100 - max(0, (state["moves"] - 8) * 4) - min(30, duration / 6)))
                interpretation = (
                    "Strong visual memory and efficiency."
                    if raw_score >= 85
                    else "Fair visual memory with some inefficiency."
                    if raw_score >= 65
                    else "Lower visual memory-game performance."
                )

                already_saved = st.session_state.get("game_saved")
                st.success(f"{ui('completed_in')} {duration}s.")

                if not already_saved:
                    result = persist_game_session(
                        user_id=user["id"],
                        game_name="emoji_memory_match",
                        raw_score=round(raw_score, 1),
                        full_marks=100.0,
                        duration_seconds=duration,
                        breakdown={
                            "moves": state["moves"],
                            "match_points": state["score"],
                            "duration_seconds": duration,
                        },
                        interpretation=interpretation,
                    )
                    st.session_state["game_saved"] = True
                    st.write(f"Score: {result['raw_score']}/100 ({result['percent_score']}%)")
                    st.caption(interpretation)
                    if result["reward"]:
                        st.balloons()
                        st.success(f"New reward unlocked: {result['reward']['voucher_name']} from {result['reward']['partner_name']}")

                st.write(f"{ui('total_points')}: {total_points(user['id'])}")

    with game_tabs[1]:
        st.subheader(ui("drink_game"))
        st.caption(ui("drink_prompt_hint"))

        if "kopi_game_state" not in st.session_state:
            initial_state = new_kopi_round(seed=int(time.time()))
            initial_state["result"] = None
            initial_state["saved_result"] = None
            st.session_state["kopi_game_state"] = initial_state

        kstate = st.session_state["kopi_game_state"]
        st.markdown(f"## {ui('drink_prompt_title')}")
        st.image(kstate["image_path"], width=260)

        with st.expander(f"ℹ️ {ui('drink_legend_title')}"):
            st.write(ui("drink_legend_desc"))
            st.markdown(
                "\n".join(
                    [
                        f"- {ui('legend_black')}",
                        f"- {ui('legend_brown')}",
                        f"- {ui('legend_blue')}",
                        f"- {ui('legend_white_cubes')}",
                        f"- {ui('legend_beige')}",
                    ]
                )
            )

        already_answered = kstate.get("result") is not None

        option_cols = st.columns(2)
        for idx, option in enumerate(kstate["options"]):
            with option_cols[idx % 2]:
                clicked = st.button(
                    option["label"],
                    key=f"drink_option_{idx}_{option['name']}",
                    use_container_width=True,
                    disabled=already_answered,
                )

                if clicked and not already_answered:
                    eval_result = evaluate_kopi_guess(kstate["target_drink"], option["name"])
                    kstate["result"] = eval_result

                    result = persist_game_session(
                        user_id=user["id"],
                        game_name="kopi_drink_guess",
                        raw_score=float(eval_result["score"]),
                        full_marks=100.0,
                        duration_seconds=20,
                        breakdown={
                            "target": kstate["target_drink"],
                            "selected": eval_result["selected"],
                            "correct": eval_result["correct"],
                        },
                        interpretation="Correct drink name recalled." if eval_result["correct"] else "Drink name recall was inaccurate.",
                    )
                    kstate["saved_result"] = result
                    st.rerun()

        if already_answered:
            eval_result = kstate["result"]
            saved_result = kstate.get("saved_result")

            if eval_result["correct"]:
                st.success(f"{ui('correct')}: {kstate['target_drink']}")
            else:
                st.error(f"{ui('not_quite')}. This was {kstate['target_drink']}.")

            if saved_result:
                st.write(f"Score: {saved_result['raw_score']}/100 ({saved_result['percent_score']}%)")
                if saved_result["reward"]:
                    st.balloons()
                    st.success(
                        f"New reward unlocked: {saved_result['reward']['voucher_name']} "
                        f"from {saved_result['reward']['partner_name']}"
                    )

        if st.button(ui("next_drink"), key="next_kopi_round", use_container_width=True):
            new_state = new_kopi_round(seed=int(time.time()))
            new_state["result"] = None
            new_state["saved_result"] = None
            st.session_state["kopi_game_state"] = new_state
            st.rerun()

        st.write(f"{ui('total_points')}: {total_points(user['id'])}")

    with game_tabs[2]:
        render_sentence_recall_game(user)

def main() -> None:
    if "language" not in st.session_state:
        st.session_state["language"] = "en"

    user_id = st.session_state.get("user_id")
    if not user_id:
        render_login()
        return

    user = get_user(user_id)
    if not user:
        logout()
        return

    render_sidebar(user)

    if user["role"] == "patient":
        render_patient_page(user)
    elif user["role"] in {"caregiver", "volunteer"}:
        render_caregiver_page(user)
    elif user["role"] == "doctor":
        render_doctor_page(user)
    else:
        st.error("Unknown role.")


if __name__ == "__main__":
    main()

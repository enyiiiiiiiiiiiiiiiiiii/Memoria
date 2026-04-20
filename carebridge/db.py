from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Iterable, Optional

from .config import DB_PATH


def _json_default(value):
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Type not serializable: {type(value)!r}")


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                full_name TEXT NOT NULL,
                role TEXT NOT NULL,
                email TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS elder_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER UNIQUE NOT NULL,
                age INTEGER,
                language_pref TEXT,
                dialect TEXT,
                interests TEXT,
                conditions TEXT,
                mobility_level TEXT,
                caregiver_notes TEXT,
                last_risk_score REAL DEFAULT 0,
                last_risk_level TEXT DEFAULT 'Unknown',
                last_risk_summary TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS care_links (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                caregiver_user_id INTEGER NOT NULL,
                patient_user_id INTEGER NOT NULL,
                link_type TEXT NOT NULL,
                FOREIGN KEY (caregiver_user_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY (patient_user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS volunteer_assignments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                volunteer_user_id INTEGER NOT NULL,
                patient_user_id INTEGER NOT NULL,
                active INTEGER DEFAULT 1,
                FOREIGN KEY (volunteer_user_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY (patient_user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS confidential_notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_user_id INTEGER NOT NULL,
                author_user_id INTEGER NOT NULL,
                note TEXT NOT NULL,
                visibility TEXT DEFAULT 'doctor_caregiver',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (patient_user_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY (author_user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS education_videos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT,
                url TEXT,
                category TEXT,
                target_role TEXT DEFAULT 'caregiver'
            );

            CREATE TABLE IF NOT EXISTS quiz_questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                option_a TEXT NOT NULL,
                option_b TEXT NOT NULL,
                option_c TEXT NOT NULL,
                option_d TEXT NOT NULL,
                correct_option TEXT NOT NULL,
                rationale TEXT,
                category TEXT
            );

            CREATE TABLE IF NOT EXISTS quiz_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                score INTEGER NOT NULL,
                total INTEGER NOT NULL,
                answers_json TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_user_id INTEGER NOT NULL,
                actor_user_id INTEGER NOT NULL,
                mode TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                concern_score REAL DEFAULT 0,
                analysis_json TEXT,
                FOREIGN KEY (patient_user_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY (actor_user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sender_user_id INTEGER NOT NULL,
                recipient_user_id INTEGER NOT NULL,
                body TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (sender_user_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY (recipient_user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS activities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT,
                category TEXT,
                age_group TEXT,
                interest_tags TEXT,
                organiser_user_id INTEGER,
                location TEXT,
                starts_at TEXT,
                capacity INTEGER DEFAULT 20,
                FOREIGN KEY (organiser_user_id) REFERENCES users(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS activity_registrations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                activity_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                status TEXT DEFAULT 'registered',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(activity_id, user_id),
                FOREIGN KEY (activity_id) REFERENCES activities(id) ON DELETE CASCADE,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS game_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                game_name TEXT NOT NULL,
                score INTEGER NOT NULL,
                points_awarded INTEGER DEFAULT 0,
                duration_seconds INTEGER DEFAULT 0,
                metadata_json TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS vouchers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                voucher_name TEXT NOT NULL,
                partner_name TEXT NOT NULL,
                points_cost INTEGER NOT NULL,
                status TEXT DEFAULT 'issued',
                issued_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS model_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_user_id INTEGER,
                actor_user_id INTEGER,
                score REAL NOT NULL,
                level TEXT NOT NULL,
                summary TEXT,
                factors_json TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (patient_user_id) REFERENCES users(id) ON DELETE SET NULL,
                FOREIGN KEY (actor_user_id) REFERENCES users(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_user_id INTEGER NOT NULL,
                severity TEXT NOT NULL,
                title TEXT NOT NULL,
                body TEXT NOT NULL,
                source TEXT NOT NULL,
                is_read INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (patient_user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            """
        )


def execute(query: str, params: tuple = ()) -> int:
    with get_conn() as conn:
        cur = conn.execute(query, params)
        return int(cur.lastrowid)


def executemany(query: str, params_seq: Iterable[tuple]) -> None:
    with get_conn() as conn:
        conn.executemany(query, list(params_seq))


def fetch_all(query: str, params: tuple = ()) -> list[sqlite3.Row]:
    with get_conn() as conn:
        cur = conn.execute(query, params)
        return cur.fetchall()


def fetch_one(query: str, params: tuple = ()) -> Optional[sqlite3.Row]:
    with get_conn() as conn:
        cur = conn.execute(query, params)
        return cur.fetchone()


def upsert_elder_profile(user_id: int, **fields) -> None:
    existing = fetch_one("SELECT id FROM elder_profiles WHERE user_id = ?", (user_id,))
    allowed = [
        "age",
        "language_pref",
        "dialect",
        "interests",
        "conditions",
        "mobility_level",
        "caregiver_notes",
        "last_risk_score",
        "last_risk_level",
        "last_risk_summary",
    ]
    payload = {k: v for k, v in fields.items() if k in allowed}
    if existing:
        if not payload:
            return
        assignments = ", ".join(f"{k} = ?" for k in payload)
        execute(
            f"UPDATE elder_profiles SET {assignments} WHERE user_id = ?",
            tuple(payload.values()) + (user_id,),
        )
    else:
        columns = ["user_id"] + list(payload)
        placeholders = ", ".join(["?"] * len(columns))
        execute(
            f"INSERT INTO elder_profiles ({', '.join(columns)}) VALUES ({placeholders})",
            (user_id, *payload.values()),
        )


def insert_json(query: str, params: tuple, payload: dict) -> int:
    return execute(query, params + (json.dumps(payload, default=_json_default),))

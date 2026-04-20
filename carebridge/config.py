from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
MODELS_DIR = DATA_DIR / "models"
EXPORT_DIR = DATA_DIR / "exports"
ARTIFACTS_DIR = EXPORT_DIR / "artifacts"
CHARTS_DIR = ARTIFACTS_DIR / "charts"
ASSETS_DIR = BASE_DIR / "assets"

DB_PATH = DATA_DIR / "carebridge.db"

# Existing core model
MODEL_PATH = MODELS_DIR / "dementia_risk_pipeline.joblib"
MODEL_META_PATH = MODELS_DIR / "dementia_risk_metrics.json"

# Modality-specific raw data roots
TABULAR_DIR = RAW_DIR / "tabular"
ADDRESSO_DIR = RAW_DIR / "addresso"
EEG_DIR = RAW_DIR / "eeg" / "ds004504"
IMAGING_DIR = RAW_DIR / "imaging" / "combined_images"

# Modality-specific model/meta outputs
TABULAR_MODEL_PATH = MODELS_DIR / "tabular_dementia_pipeline.joblib"
TABULAR_META_PATH = MODELS_DIR / "tabular_dementia_metrics.json"

ADDRESSO_MODEL_PATH = MODELS_DIR / "addresso_bundle_pipeline.joblib"
ADDRESSO_META_PATH = MODELS_DIR / "addresso_bundle_metrics.json"

EEG_MODEL_PATH = MODELS_DIR / "eeg_pipeline.joblib"
EEG_META_PATH = MODELS_DIR / "eeg_metrics.json"

IMAGING_MODEL_PATH = MODELS_DIR / "imaging_pipeline.joblib"
IMAGING_META_PATH = MODELS_DIR / "imaging_metrics.json"

APP_TITLE = "Memoria"
APP_TAGLINE = "Memory support, reminders, games, and community care in one place"
DEFAULT_LANGUAGE = os.getenv("CAREBRIDGE_DEFAULT_LANGUAGE", "en")
SECRET_KEY = os.getenv("CAREBRIDGE_SECRET_KEY", "carebridge-demo-secret")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
ENABLE_GEMINI = os.getenv("ENABLE_GEMINI", "0") == "1"

LANGUAGE_LABELS = {
    "en": "English",
    "zh": "中文",
    "ms": "Bahasa Melayu",
    "ta": "தமிழ்",
}

ROLE_LABELS = {
    "doctor": "Doctor",
    "caregiver": "Caregiver",
    "volunteer": "Volunteer",
    "patient": "Patient",
}

DEMO_USERS = [
    {
        "username": "doctor_demo",
        "password": "Doctor@123",
        "full_name": "Dr Grace Lim",
        "role": "doctor",
        "email": "doctor@example.com",
    },
    {
        "username": "caregiver_demo",
        "password": "Caregiver@123",
        "full_name": "Sarah Tan",
        "role": "caregiver",
        "email": "caregiver@example.com",
    },
    {
        "username": "volunteer_demo",
        "password": "Volunteer@123",
        "full_name": "Marcus Lee",
        "role": "volunteer",
        "email": "volunteer@example.com",
    },
    {
        "username": "patient_demo",
        "password": "Patient@123",
        "full_name": "Mdm Tan Mei Hua",
        "role": "patient",
        "email": "patient@example.com",
    },
]

for path in [
    DATA_DIR,
    RAW_DIR,
    MODELS_DIR,
    EXPORT_DIR,
    ARTIFACTS_DIR,
    CHARTS_DIR,
    ASSETS_DIR,
    TABULAR_DIR,
    ADDRESSO_DIR,
    EEG_DIR.parent,
    IMAGING_DIR.parent,
]:
    path.mkdir(parents=True, exist_ok=True)

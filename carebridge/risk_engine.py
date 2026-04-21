from __future__ import annotations

import json
import warnings
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.signal import welch
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    ADDRESSO_DIR,
    ADDRESSO_META_PATH,
    ADDRESSO_MODEL_PATH,
    CHARTS_DIR,
    EEG_DIR,
    EEG_META_PATH,
    EEG_MODEL_PATH,
    IMAGING_DIR,
    IMAGING_META_PATH,
    IMAGING_MODEL_PATH,
    MODEL_META_PATH,
    MODEL_PATH,
    TABULAR_META_PATH,
    TABULAR_MODEL_PATH,
)

warnings.filterwarnings("ignore", category=FutureWarning)

CANDIDATE_TARGETS = [
    "Diagnosis",
    "diagnosis",
    "target",
    "label",
    "Class",
    "Outcome",
    "Dementia",
    "dementia",
    "Group",
]

DROP_COLUMNS_EXACT = {
    "DoctorInCharge",
    "Subject ID",
    "MRI ID",
}

DROP_COLUMNS_LOWER = {
    "patientid",
    "id",
    "recordid",
}

CANONICAL_FEATURES = {
    "age": ["Age", "age"],
    "gender": ["Gender", "gender", "Sex", "sex"],
    "education": ["EducationLevel", "education", "education_level"],
    "bmi": ["BMI", "bmi"],
    "sleep_quality": ["SleepQuality", "sleep_quality", "sleep"],
    "physical_activity": ["PhysicalActivity", "physical_activity", "activity"],
    "depression": ["Depression", "depression"],
    "hypertension": ["Hypertension", "hypertension"],
    "diabetes": ["Diabetes", "diabetes"],
    "memory_complaints": ["MemoryComplaints", "memory_complaints"],
    "behavioral_problems": ["BehavioralProblems", "behavioral_problems"],
    "adl": ["ADL", "adl", "FunctionalAssessment", "functionalassessment"],
    "confusion": ["Confusion", "confusion"],
    "disorientation": ["Disorientation", "disorientation"],
    "difficulty_tasks": ["DifficultyCompletingTasks", "difficulty_tasks"],
    "forgetfulness": ["Forgetfulness", "forgetfulness"],
    "mmse": ["MMSE", "mmse"],
}


# -------------------------------------------------------------------
# Generic helpers
# -------------------------------------------------------------------
def _safe_slug(name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_chart_dir(modality: str) -> Path:
    chart_dir = CHARTS_DIR / _safe_slug(modality)
    chart_dir.mkdir(parents=True, exist_ok=True)
    return chart_dir


def _save_class_distribution_chart(y: pd.Series, chart_dir: Path, prefix: str) -> str:
    counts = pd.Series(y).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([str(i) for i in counts.index], counts.values)
    ax.set_title("Class Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    fig.tight_layout()
    path = chart_dir / f"{prefix}_class_distribution.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def _save_model_comparison_chart(report_payload: dict[str, Any], chart_dir: Path, prefix: str) -> str:
    names = list(report_payload.keys())
    aucs = [report_payload[k]["auc"] for k in names]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(names, aucs)
    ax.set_ylim(0, 1)
    ax.set_title("Model Comparison (ROC AUC)")
    ax.set_ylabel("ROC AUC")
    fig.tight_layout()
    path = chart_dir / f"{prefix}_model_comparison.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def _save_roc_curve(y_true: pd.Series, proba: np.ndarray, chart_dir: Path, prefix: str) -> str | None:
    if pd.Series(y_true).nunique() < 2:
        return None
    fpr, tpr, _ = roc_curve(y_true, proba)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, label="ROC")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    fig.tight_layout()
    path = chart_dir / f"{prefix}_roc_curve.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def _save_pr_curve(y_true: pd.Series, proba: np.ndarray, chart_dir: Path, prefix: str) -> str | None:
    if pd.Series(y_true).nunique() < 2:
        return None
    precision, recall, _ = precision_recall_curve(y_true, proba)
    ap = average_precision_score(y_true, proba)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(recall, precision, label=f"PR AUC={ap:.3f}")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    fig.tight_layout()
    path = chart_dir / f"{prefix}_pr_curve.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def _save_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, chart_dir: Path, prefix: str) -> str:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    path = chart_dir / f"{prefix}_confusion_matrix.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def _save_feature_importance_chart(model: Pipeline, X: pd.DataFrame, chart_dir: Path, prefix: str) -> str | None:
    try:
        estimator = model.named_steps["model"]
        if not hasattr(estimator, "feature_importances_"):
            return None
        preprocess = model.named_steps["preprocess"]
        try:
            names = list(preprocess.get_feature_names_out())
        except Exception:
            names = list(X.columns)
        importances = estimator.feature_importances_
        top_idx = np.argsort(importances)[::-1][:12]
        top_names = [names[i] if i < len(names) else f"f_{i}" for i in top_idx]
        top_vals = importances[top_idx]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(list(reversed(top_names)), list(reversed(top_vals)))
        ax.set_title("Top Feature Importances")
        fig.tight_layout()
        path = chart_dir / f"{prefix}_feature_importance.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        return str(path)
    except Exception:
        return None


def _save_pca_scatter(X: pd.DataFrame, labels: np.ndarray, chart_dir: Path, prefix: str, title: str) -> str:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="viridis", s=18)
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.colorbar(scatter, ax=ax)
    fig.tight_layout()
    path = chart_dir / f"{prefix}_pca_clusters.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def _detect_target(df: pd.DataFrame) -> str:
    for col in CANDIDATE_TARGETS:
        if col in df.columns:
            return col
    raise ValueError(
        f"Could not detect target column. Expected one of: {', '.join(CANDIDATE_TARGETS)}"
    )


def _coerce_binary_target(series: pd.Series, target_col: str) -> pd.Series:
    if target_col == "Group":
        mapped = series.astype(str).str.strip().map(
            {
                "Nondemented": 0,
                "Demented": 1,
                "Converted": 1,
            }
        )
        if mapped.isna().any():
            raise ValueError("Group column contains unsupported labels.")
        return mapped.astype(int)

    if series.dropna().isin([0, 1]).all():
        return series.astype(int)

    lowered = series.astype(str).str.strip().str.lower()
    mapping = {
        "yes": 1,
        "no": 0,
        "positive": 1,
        "negative": 0,
        "ad": 1,
        "control": 0,
        "dementia": 1,
        "healthy": 0,
        "true": 1,
        "false": 0,
        "converted": 1,
        "nondemented": 0,
        "demented": 1,
    }
    mapped = lowered.map(mapping)
    if mapped.isna().any():
        raise ValueError("Target column contains non-binary values the MVP pipeline cannot coerce safely.")
    return mapped.astype(int)


def _drop_irrelevant_columns(X: pd.DataFrame) -> pd.DataFrame:
    exact_drop = [c for c in X.columns if c in DROP_COLUMNS_EXACT]
    lower_drop = [c for c in X.columns if c.lower() in DROP_COLUMNS_LOWER]
    drop_cols = list(dict.fromkeys(exact_drop + lower_drop))
    if drop_cols:
        X = X.drop(columns=drop_cols)
    return X


def build_training_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, str]:
    target_col = _detect_target(df)
    y = _coerce_binary_target(df[target_col], target_col)
    X = df.drop(columns=[target_col]).copy()
    X = _drop_irrelevant_columns(X)
    return X, y, target_col


def _train_binary_tabular_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    model_path: Path,
    meta_path: Path,
    modality_name: str,
) -> dict[str, Any]:
    feature_columns = X.columns.tolist()
    numeric_cols = list(X.select_dtypes(include=["number", "bool"]).columns)
    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )

    candidates = {
        "logistic_regression": LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="liblinear",
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            random_state=42,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            n_jobs=-1,
        ),
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    best_name = None
    best_model = None
    best_auc = -1.0
    best_preds = None
    best_proba = None
    report_payload: dict[str, Any] = {}

    for name, estimator in candidates.items():
        pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        pr_auc = average_precision_score(y_test, proba)
        report_payload[name] = {
            "auc": round(float(auc), 4),
            "pr_auc": round(float(pr_auc), 4),
            "accuracy": round(float(accuracy_score(y_test, preds)), 4),
            "classification_report": classification_report(y_test, preds, output_dict=True),
        }
        if auc > best_auc:
            best_auc = float(auc)
            best_name = name
            best_model = pipe
            best_preds = preds
            best_proba = proba

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_path)

    chart_dir = _ensure_chart_dir(modality_name)
    chart_paths = {
        "class_distribution": _save_class_distribution_chart(y, chart_dir, modality_name),
        "model_comparison": _save_model_comparison_chart(report_payload, chart_dir, modality_name),
        "confusion_matrix": _save_confusion_matrix(y_test, best_preds, chart_dir, modality_name),
    }
    roc_path = _save_roc_curve(y_test, best_proba, chart_dir, modality_name)
    if roc_path:
        chart_paths["roc_curve"] = roc_path
    pr_path = _save_pr_curve(y_test, best_proba, chart_dir, modality_name)
    if pr_path:
        chart_paths["pr_curve"] = pr_path
    fi_path = _save_feature_importance_chart(best_model, X, chart_dir, modality_name)
    if fi_path:
        chart_paths["feature_importance"] = fi_path

    metadata = {
        "modality": modality_name,
        "feature_columns": feature_columns,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "best_model": best_name,
        "best_auc": round(best_auc, 4),
        "best_pr_auc": round(float(average_precision_score(y_test, best_proba)), 4),
        "candidates": report_payload,
        "row_count": int(len(X)),
        "chart_paths": chart_paths,
        "training_mode": "supervised",
    }
    _write_json(meta_path, metadata)
    return metadata


def _train_unsupervised_profile_dataset(
    X: pd.DataFrame,
    model_path: Path,
    meta_path: Path,
    modality_name: str,
) -> dict[str, Any]:
    X_num = X.select_dtypes(include=["number", "bool"]).copy()
    if X_num.empty:
        raise ValueError("No numeric features were available for exploratory profiling.")

    X_num = X_num.fillna(X_num.median(numeric_only=True)).fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_num)

    n_clusters = 2 if len(X_num) < 30 else 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    iso = IsolationForest(random_state=42, contamination=0.15)
    anomaly_labels = iso.fit_predict(X_scaled)
    anomaly_rate = float(np.mean(anomaly_labels == -1))
    sil = None
    if len(np.unique(cluster_labels)) > 1 and len(X_num) > n_clusters:
        sil = float(silhouette_score(X_scaled, cluster_labels))

    bundle = {
        "scaler": scaler,
        "kmeans": kmeans,
        "isolation_forest": iso,
        "feature_columns": list(X_num.columns),
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)

    chart_dir = _ensure_chart_dir(modality_name)
    pca_cluster_path = _save_pca_scatter(
        X_num,
        cluster_labels,
        chart_dir,
        modality_name,
        "Patient Behaviour Clusters",
    )
    pca_anomaly_path = _save_pca_scatter(
        X_num,
        (anomaly_labels == -1).astype(int),
        chart_dir,
        f"{modality_name}_anomaly",
        "Anomaly Pattern View",
    )

    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([str(i) for i in cluster_counts.index], cluster_counts.values)
    ax.set_title("Cluster Distribution")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Count")
    fig.tight_layout()
    cluster_path = chart_dir / f"{modality_name}_cluster_distribution.png"
    fig.savefig(cluster_path, dpi=140)
    plt.close(fig)

    metadata = {
        "modality": modality_name,
        "feature_columns": list(X_num.columns),
        "row_count": int(len(X_num)),
        "training_mode": "unsupervised_exploration",
        "n_clusters": int(n_clusters),
        "cluster_counts": {str(k): int(v) for k, v in cluster_counts.to_dict().items()},
        "anomaly_rate": round(anomaly_rate, 4),
        "silhouette_score": round(sil, 4) if sil is not None else None,
        "chart_paths": {
            "cluster_distribution": str(cluster_path),
            "pca_clusters": pca_cluster_path,
            "pca_anomalies": pca_anomaly_path,
        },
    }
    _write_json(meta_path, metadata)
    return metadata


# -------------------------------------------------------------------
# Core tabular model
# -------------------------------------------------------------------
def train_and_save_model(csv_path: str | Path) -> dict[str, Any]:
    df = pd.read_csv(csv_path)
    X, y, target_col = build_training_frame(df)
    metadata = _train_binary_tabular_dataset(
        X=X,
        y=y,
        model_path=MODEL_PATH,
        meta_path=MODEL_META_PATH,
        modality_name="tabular_core",
    )
    metadata["source_csv"] = str(csv_path)
    metadata["target_column"] = target_col
    _write_json(MODEL_META_PATH, metadata)
    joblib.dump(joblib.load(MODEL_PATH), TABULAR_MODEL_PATH)
    _write_json(TABULAR_META_PATH, metadata)
    return metadata


# -------------------------------------------------------------------
# Behavioural multi-sheet bundle
# -------------------------------------------------------------------
def _load_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


def _find_join_key(frames: list[pd.DataFrame]) -> str | None:
    candidates = [
        "patient_id",
        "participant_id",
        "Participant_ID",
        "subject_id",
        "Subject_ID",
        "subject",
        "Subject",
        "user_id",
        "User_ID",
        "pid",
        "ID",
        "id",
    ]
    for key in candidates:
        if all(key in frame.columns for frame in frames if frame is not None):
            return key
    return None


def _aggregate_sheet(df: pd.DataFrame, id_col: str, prefix: str) -> pd.DataFrame:
    if id_col not in df.columns:
        return pd.DataFrame()

    work = df.copy()

    if "date" in work.columns:
        work["date"] = pd.to_datetime(work["date"], errors="coerce")
        work[f"{prefix}_hour"] = work["date"].dt.hour
        work[f"{prefix}_dayofweek"] = work["date"].dt.dayofweek

    for col in work.columns:
        if col == id_col:
            continue
        if work[col].dtype == object:
            if col == "date":
                continue
            work[col] = work[col].astype(str)
            freq = work.groupby([id_col, col]).size().unstack(fill_value=0)
            freq.columns = [f"{prefix}_{col}_{c}_count" for c in freq.columns]
            freq = freq.reset_index()
            numeric = work.select_dtypes(include=["number", "bool"]).copy()
            if id_col not in numeric.columns:
                numeric[id_col] = work[id_col]
            grouped_num = numeric.groupby(id_col).agg(["mean", "std", "min", "max"]).reset_index()
            grouped_num.columns = [
                id_col if c[0] == id_col else f"{prefix}_{c[0]}_{c[1]}"
                for c in grouped_num.columns
            ]
            merged = grouped_num.merge(freq, on=id_col, how="left")
            return merged.fillna(0)

    numeric = work.select_dtypes(include=["number", "bool"]).copy()
    if id_col not in numeric.columns:
        numeric[id_col] = work[id_col]
    grouped = numeric.groupby(id_col).agg(["mean", "std", "min", "max"])
    grouped.columns = [f"{prefix}_{a}_{b}" for a, b in grouped.columns]
    grouped = grouped.reset_index()
    return grouped.fillna(0)


def _extract_patient_level_target(labels_df: pd.DataFrame, id_col: str) -> tuple[str | None, pd.DataFrame | None]:
    if labels_df is None or id_col not in labels_df.columns:
        return None, None

    for col in CANDIDATE_TARGETS:
        if col not in labels_df.columns:
            continue
        try:
            coerced = _coerce_binary_target(labels_df[col], col)
            temp = pd.DataFrame(
                {
                    id_col: labels_df[id_col].astype(str),
                    "__target__": coerced.astype(int),
                }
            )
            patient_target = temp.groupby(id_col, as_index=False)["__target__"].max()
            return col, patient_target
        except Exception:
            continue
    return None, None


def train_addresso_bundle(root_dir: str | Path = ADDRESSO_DIR) -> dict[str, Any]:
    root_dir = Path(root_dir)
    activity = _load_csv_if_exists(root_dir / "Activity.csv")
    demographics = _load_csv_if_exists(root_dir / "Demographics.csv")
    labels = _load_csv_if_exists(root_dir / "Labels.csv")
    physiology = _load_csv_if_exists(root_dir / "Physiology.csv")
    sleep = _load_csv_if_exists(root_dir / "Sleep.csv")

    if all(x is None for x in [activity, demographics, labels, physiology, sleep]):
        raise FileNotFoundError("No behavioural sheets were found in the configured folder.")

    available_frames = [f for f in [activity, demographics, labels, physiology, sleep] if f is not None]
    join_key = _find_join_key(available_frames)
    if join_key is None:
        raise ValueError("Could not determine a shared patient identifier across the behavioural sheets.")

    patient_ids = None
    for df in available_frames:
        ids = df[[join_key]].drop_duplicates()
        patient_ids = ids if patient_ids is None else patient_ids.merge(ids, on=join_key, how="outer")

    base = patient_ids.copy()

    if demographics is not None:
        demo = demographics.copy()
        demo[join_key] = demo[join_key].astype(str)
        base = base.merge(demo, on=join_key, how="left")

    if activity is not None:
        base = base.merge(_aggregate_sheet(activity, join_key, "activity"), on=join_key, how="left")
    if physiology is not None:
        base = base.merge(_aggregate_sheet(physiology, join_key, "physiology"), on=join_key, how="left")
    if sleep is not None:
        base = base.merge(_aggregate_sheet(sleep, join_key, "sleep"), on=join_key, how="left")
    supervised_target = None
    patient_target_df = None
    if labels is not None:
        base = base.merge(_aggregate_sheet(labels, join_key, "labels"), on=join_key, how="left")
        supervised_target, patient_target_df = _extract_patient_level_target(labels, join_key)
        if patient_target_df is not None:
            base[join_key] = base[join_key].astype(str)
            base = base.merge(patient_target_df, on=join_key, how="left")

    supervised_y = None
    if "__target__" in base.columns and base["__target__"].notna().sum() >= 2:
        keep = base["__target__"].notna()
        supervised_y = base.loc[keep, "__target__"].astype(int)
        base = base.loc[keep].copy()

    X = base.drop(columns=[join_key, "__target__"], errors="ignore").copy()

    chart_dir = _ensure_chart_dir("behavioural_bundle")
    missingness = X.isna().mean().sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(list(reversed(missingness.index.tolist())), list(reversed(missingness.values.tolist())))
    ax.set_title("Top Missingness Rates")
    ax.set_xlabel("Fraction Missing")
    fig.tight_layout()
    missing_path = chart_dir / "behavioural_bundle_missingness.png"
    fig.savefig(missing_path, dpi=140)
    plt.close(fig)

    if supervised_target is not None and supervised_y is not None:
        metadata = _train_binary_tabular_dataset(
            X=X.fillna(0),
            y=supervised_y,
            model_path=ADDRESSO_MODEL_PATH,
            meta_path=ADDRESSO_META_PATH,
            modality_name="behavioural_bundle",
        )
        metadata["target_column"] = supervised_target
        metadata["usable_labeled_rows"] = int(len(X))
    else:
        metadata = _train_unsupervised_profile_dataset(
            X=X,
            model_path=ADDRESSO_MODEL_PATH,
            meta_path=ADDRESSO_META_PATH,
            modality_name="behavioural_bundle",
        )

    metadata["source_root"] = str(root_dir)
    metadata["join_key"] = join_key
    metadata["available_files"] = {
        "Activity.csv": activity is not None,
        "Demographics.csv": demographics is not None,
        "Labels.csv": labels is not None,
        "Physiology.csv": physiology is not None,
        "Sleep.csv": sleep is not None,
    }
    metadata.setdefault("chart_paths", {})["missingness"] = str(missing_path)
    _write_json(ADDRESSO_META_PATH, metadata)
    return metadata


# -------------------------------------------------------------------
# EEG modelling
# -------------------------------------------------------------------
def _extract_eeg_features_from_file(file_path: Path) -> dict[str, float]:
    try:
        import mne

        if file_path.suffix.lower() == ".set":
            raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose="ERROR")
        elif file_path.suffix.lower() == ".edf":
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose="ERROR")
        elif file_path.suffix.lower() == ".bdf":
            raw = mne.io.read_raw_bdf(file_path, preload=True, verbose="ERROR")
        elif file_path.suffix.lower() == ".fif":
            raw = mne.io.read_raw_fif(file_path, preload=True, verbose="ERROR")
        else:
            return {}

        data = raw.get_data()
        sfreq = float(raw.info["sfreq"])
        if data.size == 0:
            return {}

        channel_mean = np.mean(data, axis=1)
        channel_std = np.std(data, axis=1)
        overall_mean = float(np.mean(channel_mean))
        overall_std = float(np.mean(channel_std))

        freqs, psd = welch(data, fs=sfreq, axis=1, nperseg=min(256, data.shape[1]))
        band_defs = {
            "delta": (1, 4),
            "theta": (4, 8),
            "alpha": (8, 13),
            "beta": (13, 30),
        }
        features = {
            "signal_mean": overall_mean,
            "signal_std": overall_std,
        }
        total_power = np.maximum(np.sum(psd, axis=1).mean(), 1e-9)

        for band, (lo, hi) in band_defs.items():
            mask = (freqs >= lo) & (freqs < hi)
            band_power = float(np.sum(psd[:, mask], axis=1).mean())
            features[f"{band}_power"] = band_power
            features[f"{band}_rel_power"] = band_power / total_power

        return features
    except Exception:
        return {}


def train_eeg_bundle(root_dir: str | Path = EEG_DIR, max_files: int = 40) -> dict[str, Any]:
    root_dir = Path(root_dir)

    # Allow either an extracted directory or a .zip upload path.
    # If a zip is provided, extract it to a temporary folder and recurse from there.
    if root_dir.is_file() and root_dir.suffix.lower() == ".zip":
        temp_dir = Path(tempfile.mkdtemp(prefix="memoria_eeg_"))
        with zipfile.ZipFile(root_dir, "r") as zip_ref:
            zip_ref.extractall(temp_dir)
        root_dir = temp_dir

    participants = None
    for candidate in list(root_dir.rglob("participants.tsv")) + list(root_dir.rglob("participants.csv")):
        if candidate.exists():
            participants = pd.read_csv(candidate, sep="\t" if candidate.suffix == ".tsv" else ",")
            break

    eeg_files = []
    for ext in ["*.set", "*.edf", "*.bdf", "*.fif"]:
        eeg_files.extend(root_dir.rglob(ext))
    eeg_files = sorted(eeg_files)[:max_files]

    # If files are unreadable or git-annex placeholders, we still do exploratory metadata mode
    if participants is not None:
        meta_df = participants.copy()
    else:
        # fallback manifest from file names only
        meta_df = pd.DataFrame({"file_name": [p.name for p in eeg_files]})

    target_col = None
    y = None
    for col in ["group", "Group", "diagnosis", "Diagnosis", "label", "Label", "status", "Status"]:
        if col in meta_df.columns:
            target_col = col
            try:
                lowered = meta_df[col].astype(str).str.strip().str.lower()
                mapping = {
                    "ad": 1,
                    "alzheimers": 1,
                    "alzheimer's disease": 1,
                    "dementia": 1,
                    "patient": 1,
                    "case": 1,
                    "control": 0,
                    "healthy": 0,
                    "hc": 0,
                    "normal": 0,
                }
                mapped = lowered.map(mapping)
                if mapped.notna().sum() >= max(2, int(len(meta_df) * 0.4)):
                    y = mapped.fillna(0).astype(int)
            except Exception:
                y = None
            break

    rows = []
    for file_path in eeg_files:
        feats = _extract_eeg_features_from_file(file_path)
        if feats:
            feats["file_name"] = file_path.name
            rows.append(feats)

    chart_dir = _ensure_chart_dir("eeg_signal")

    if rows and y is not None and len(rows) == len(y):
        eeg_df = pd.DataFrame(rows)
        X = eeg_df.drop(columns=["file_name"], errors="ignore")
        metadata = _train_binary_tabular_dataset(
            X=X.fillna(0),
            y=y.iloc[: len(X)],
            model_path=EEG_MODEL_PATH,
            meta_path=EEG_META_PATH,
            modality_name="eeg_signal",
        )
        metadata["mode"] = "signal_features_supervised"
    else:
        # fallback to exploratory profiling on whatever metadata / signal rows we have
        if rows:
            X = pd.DataFrame(rows).drop(columns=["file_name"], errors="ignore")
            metadata = _train_unsupervised_profile_dataset(
                X=X,
                model_path=EEG_MODEL_PATH,
                meta_path=EEG_META_PATH,
                modality_name="eeg_signal",
            )
            metadata["mode"] = "signal_features_unsupervised"
        else:
            work = meta_df.copy()
            # create numeric surrogates if only categorical metadata exists
            for col in work.columns:
                if work[col].dtype == object:
                    work[col] = work[col].astype("category").cat.codes
            metadata = _train_unsupervised_profile_dataset(
                X=work,
                model_path=EEG_MODEL_PATH,
                meta_path=EEG_META_PATH,
                modality_name="eeg_metadata",
            )
            metadata["mode"] = "metadata_unsupervised"

    metadata["eeg_files_found"] = len(eeg_files)
    metadata["participants_found"] = int(len(meta_df))

    # class/group summary if available
    if target_col is not None:
        counts = meta_df[target_col].astype(str).value_counts()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(counts.index.tolist(), counts.values.tolist())
        ax.set_title("EEG Label / Group Distribution")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        p = chart_dir / "eeg_group_distribution.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        metadata.setdefault("chart_paths", {})["group_distribution"] = str(p)

    _write_json(EEG_META_PATH, metadata)
    return metadata


# -------------------------------------------------------------------
# Imaging modelling
# -------------------------------------------------------------------
def _image_feature_vector(img_path: Path, size: tuple[int, int] = (64, 64)) -> dict[str, float]:
    img = Image.open(img_path).convert("L").resize(size)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    hist, _ = np.histogram(arr, bins=16, range=(0.0, 1.0), density=True)

    features = {
        "mean_intensity": float(arr.mean()),
        "std_intensity": float(arr.std()),
        "min_intensity": float(arr.min()),
        "max_intensity": float(arr.max()),
        "edge_proxy": float(np.mean(np.abs(np.diff(arr, axis=0))) + np.mean(np.abs(np.diff(arr, axis=1)))),
    }
    for i, val in enumerate(hist):
        features[f"hist_{i}"] = float(val)
    return features


def train_imaging_bundle(root_dir: str | Path = IMAGING_DIR, max_per_class: int = 250) -> dict[str, Any]:
    root_dir = Path(root_dir)
    class_dirs = [d for d in root_dir.iterdir() if d.is_dir()]
    if not class_dirs:
        raise FileNotFoundError("No class folders found in imaging root")

    rows = []
    class_counts = {}
    for class_dir in sorted(class_dirs):
        image_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            image_files.extend(class_dir.glob(ext))
        image_files = sorted(image_files)[:max_per_class]
        class_counts[class_dir.name] = len(image_files)
        for img_path in image_files:
            feats = _image_feature_vector(img_path)
            feats["label_name"] = class_dir.name
            rows.append(feats)

    if len(rows) < 8:
        raise ValueError("Not enough imaging samples found for training.")

    df = pd.DataFrame(rows)
    df["target"] = (df["label_name"].str.lower() != "nondemented").astype(int)
    y = df["target"]
    X = df.drop(columns=["label_name", "target"])

    metadata = _train_binary_tabular_dataset(
        X=X,
        y=y,
        model_path=IMAGING_MODEL_PATH,
        meta_path=IMAGING_META_PATH,
        modality_name="imaging_binary",
    )
    metadata["class_counts"] = class_counts
    metadata["binary_mapping"] = {
        "NonDemented": 0,
        "MildDemented": 1,
        "VeryMildDemented": 1,
        "ModerateDemented": 1,
    }

    chart_dir = _ensure_chart_dir("imaging_binary")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(class_counts.keys(), class_counts.values())
    ax.set_title("Imaging Samples per Class")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    p = chart_dir / "imaging_class_counts.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    metadata.setdefault("chart_paths", {})["class_counts"] = str(p)

    _write_json(IMAGING_META_PATH, metadata)
    return metadata


# -------------------------------------------------------------------
# Inference helpers
# -------------------------------------------------------------------
def _find_value(profile: dict[str, Any], keys: list[str], default: float = 0.0) -> Any:
    for key in keys:
        if key in profile and profile[key] not in (None, ""):
            return profile[key]
    lowered = {str(k).lower(): v for k, v in profile.items()}
    for key in keys:
        if key.lower() in lowered and lowered[key.lower()] not in (None, ""):
            return lowered[key.lower()]
    return default


def align_profile_to_training_columns(profile: dict[str, Any], feature_columns: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame([profile])
    for col in feature_columns:
        if col not in frame.columns:
            frame[col] = 0
    frame = frame[feature_columns]
    return frame


def heuristic_profile_score(profile: dict[str, Any]) -> dict[str, Any]:
    age = float(_find_value(profile, CANONICAL_FEATURES["age"], 70))
    mmse = float(_find_value(profile, CANONICAL_FEATURES["mmse"], 24))
    memory = float(_find_value(profile, CANONICAL_FEATURES["memory_complaints"], 0))
    confusion = float(_find_value(profile, CANONICAL_FEATURES["confusion"], 0))
    disorientation = float(_find_value(profile, CANONICAL_FEATURES["disorientation"], 0))
    forgetfulness = float(_find_value(profile, CANONICAL_FEATURES["forgetfulness"], 0))
    adl = float(_find_value(profile, CANONICAL_FEATURES["adl"], 7))
    hypertension = float(_find_value(profile, CANONICAL_FEATURES["hypertension"], 0))
    diabetes = float(_find_value(profile, CANONICAL_FEATURES["diabetes"], 0))
    depression = float(_find_value(profile, CANONICAL_FEATURES["depression"], 0))
    sleep_quality = float(_find_value(profile, CANONICAL_FEATURES["sleep_quality"], 6))
    physical_activity = float(_find_value(profile, CANONICAL_FEATURES["physical_activity"], 5))
    behavioral_problems = float(_find_value(profile, CANONICAL_FEATURES["behavioral_problems"], 0))
    difficulty_tasks = float(_find_value(profile, CANONICAL_FEATURES["difficulty_tasks"], 0))

    score = 0.0
    score += min(max((age - 60) / 30, 0), 1) * 0.14
    score += min(max((30 - mmse) / 20, 0), 1) * 0.22
    score += min(memory, 1) * 0.12
    score += min(confusion, 1) * 0.10
    score += min(disorientation, 1) * 0.10
    score += min(forgetfulness, 1) * 0.10
    score += min(behavioral_problems, 1) * 0.05
    score += min(difficulty_tasks, 1) * 0.07
    score += min(hypertension, 1) * 0.03
    score += min(diabetes, 1) * 0.03
    score += min(depression, 1) * 0.02
    score += min(max((7 - sleep_quality) / 7, 0), 1) * 0.01
    score += min(max((5 - physical_activity) / 5, 0), 1) * 0.01
    score += min(max((7 - adl) / 7, 0), 1) * 0.10
    score = max(0.0, min(score, 0.99))

    level = "Low"
    if score >= 0.7:
        level = "High"
    elif score >= 0.4:
        level = "Moderate"

    factors = explain_profile(profile)
    return {
        "score": round(score, 3),
        "level": level,
        "summary": summarize_risk(score, level, factors),
        "factors": factors,
        "engine": "heuristic",
    }


def explain_profile(profile: dict[str, Any]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []

    def add(condition: bool, label: str, direction: str, weight: str) -> None:
        if condition:
            evidence.append({"label": label, "direction": direction, "weight": weight})

    age = float(_find_value(profile, CANONICAL_FEATURES["age"], 70))
    mmse = float(_find_value(profile, CANONICAL_FEATURES["mmse"], 24))
    memory = float(_find_value(profile, CANONICAL_FEATURES["memory_complaints"], 0))
    confusion = float(_find_value(profile, CANONICAL_FEATURES["confusion"], 0))
    disorientation = float(_find_value(profile, CANONICAL_FEATURES["disorientation"], 0))
    forgetfulness = float(_find_value(profile, CANONICAL_FEATURES["forgetfulness"], 0))
    sleep_quality = float(_find_value(profile, CANONICAL_FEATURES["sleep_quality"], 6))
    physical_activity = float(_find_value(profile, CANONICAL_FEATURES["physical_activity"], 5))
    adl = float(_find_value(profile, CANONICAL_FEATURES["adl"], 7))

    add(age >= 75, "Older age band", "risk_up", "medium")
    add(mmse <= 23, "Lower cognitive screening score", "risk_up", "high")
    add(memory >= 1, "Memory complaints present", "risk_up", "high")
    add(confusion >= 1, "Confusion episodes reported", "risk_up", "high")
    add(disorientation >= 1, "Disorientation reported", "risk_up", "high")
    add(forgetfulness >= 1, "Repeated forgetfulness reported", "risk_up", "medium")
    add(adl <= 5, "Reduced daily functional performance", "risk_up", "high")
    add(sleep_quality >= 7, "Good sleep pattern", "protective", "low")
    add(physical_activity >= 6, "Higher physical activity", "protective", "low")
    return evidence[:6]


def summarize_risk(score: float, level: str, factors: list[dict[str, Any]]) -> str:
    if not factors:
        return f"Estimated dementia concern score: {score:.2f} ({level}). Collect more behavioural or screening data for a stronger profile."
    top = ", ".join(f["label"] for f in factors[:3])
    return f"Estimated dementia concern score: {score:.2f} ({level}). Key signals: {top}."


def predict_profile(profile: dict[str, Any]) -> dict[str, Any]:
    if MODEL_PATH.exists():
        try:
            model = joblib.load(MODEL_PATH)
            meta = load_training_metadata()
            feature_columns = meta.get("feature_columns", [])
            frame = (
                align_profile_to_training_columns(profile, feature_columns)
                if feature_columns
                else pd.DataFrame([profile])
            )
            score = float(model.predict_proba(frame)[0, 1])
            level = "Low" if score < 0.4 else "Moderate" if score < 0.7 else "High"
            factors = explain_profile(profile)
            return {
                "score": round(score, 3),
                "level": level,
                "summary": summarize_risk(score, level, factors),
                "factors": factors,
                "engine": "trained_model",
            }
        except Exception:
            pass
    return heuristic_profile_score(profile)


def load_training_metadata() -> dict[str, Any]:
    return load_json_if_exists(MODEL_META_PATH)


def load_tabular_metadata() -> dict[str, Any]:
    return load_json_if_exists(TABULAR_META_PATH)


def load_addresso_metadata() -> dict[str, Any]:
    return load_json_if_exists(ADDRESSO_META_PATH)


def load_eeg_metadata() -> dict[str, Any]:
    return load_json_if_exists(EEG_META_PATH)


def load_imaging_metadata() -> dict[str, Any]:
    return load_json_if_exists(IMAGING_META_PATH)

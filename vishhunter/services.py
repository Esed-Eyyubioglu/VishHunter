from __future__ import annotations

import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path

from fastapi import UploadFile
from sqlalchemy import func, select
from sqlalchemy.orm import Session, joinedload

from .config import settings
from .ml import pipeline
from .models import AuditLog, AudioFeatureSet, AudioRecord, Case, TextFeatureSet, Transcription, User
from .security import decrypt_text, encrypt_text, hash_password
from .system_settings import load_system_settings, save_system_settings


def ensure_directories() -> None:
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.artifact_dir.mkdir(parents=True, exist_ok=True)
    settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
    load_system_settings()


def bootstrap_defaults(db: Session) -> None:
    return None


def any_users_exist(db: Session) -> bool:
    return bool(db.scalar(select(func.count()).select_from(User)))


def create_initial_admin(db: Session, *, full_name: str, email: str, password: str) -> User:
    user = User(
        full_name=full_name,
        email=email,
        password_hash=hash_password(password),
        role="administrator",
        is_active=True,
    )
    db.add(user)
    db.flush()
    log_event(db, user, "system.setup", "Initial administrator account created")
    return user


def log_event(db: Session, user: User | None, action: str, details: str) -> None:
    db.add(AuditLog(user_id=user.id if user else None, action=action, details=details))


def next_case_number(db: Session) -> str:
    return f"VH-{datetime.utcnow():%Y%m%d}-{uuid.uuid4().hex[:6].upper()}"


def save_upload(file: UploadFile) -> tuple[str, int]:
    suffix = Path(file.filename or "upload.wav").suffix or ".wav"
    allowed_suffixes = {".wav", ".mp3", ".m4a", ".flac"}
    if suffix.lower() not in allowed_suffixes:
        raise ValueError("Unsupported file type. Upload WAV, MP3, M4A, or FLAC audio.")
    destination = settings.upload_dir / f"{uuid.uuid4()}{suffix}"
    content = file.file.read()
    destination.write_bytes(content)
    return str(destination), len(content)


def create_case_from_upload(
    db: Session,
    *,
    file: UploadFile,
    uploaded_by: User,
) -> Case:
    storage_path, file_size = save_upload(file)
    try:
        processed = pipeline.analyze(storage_path)
    except Exception:
        Path(storage_path).unlink(missing_ok=True)
        raise
    analyst = db.scalar(select(User).where(User.role == "analyst", User.is_active == True))
    case = Case(
        case_number=next_case_number(db),
        title=(file.filename or "Uploaded call recording").rsplit(".", 1)[0].replace("-", " ").title(),
        uploaded_by=uploaded_by.id,
        assigned_to=analyst.id if analyst else None,
        status="analyzed",
        risk_level=processed.risk_level,
        audio_score=processed.audio_score,
        text_score=processed.text_score,
        risk_score=processed.risk_score,
        confidence_score=processed.confidence,
        model_verdict=processed.verdict,
        indicators=processed.indicators,
        summary=processed.summary,
        review_status="pending",
    )
    db.add(case)
    db.flush()

    db.add(
        AudioRecord(
            case_id=case.id,
            original_filename=file.filename or "audio.wav",
            storage_path=storage_path,
            file_hash=processed.file_hash,
            format=(Path(file.filename or "audio.wav").suffix or ".wav").replace(".", "").lower(),
            mime_type=file.content_type or "audio/wav",
            duration_seconds=processed.duration_seconds,
            file_size_bytes=file_size,
        )
    )
    db.add(Transcription(case_id=case.id, encrypted_text=encrypt_text(processed.transcript), detected_language="en"))
    db.add(
        AudioFeatureSet(
            case_id=case.id,
            mfcc_vector=processed.mfcc_vector,
            spectral_contrast=processed.spectral_vector,
            zcr=processed.zcr,
            pitch_profile=processed.pitch_profile,
            anomalies=processed.anomalies,
        )
    )
    db.add(
        TextFeatureSet(
            case_id=case.id,
            embedding_vector=processed.embedding_vector,
            indicators=processed.indicators,
            suspicious_phrases=processed.suspicious_phrases,
        )
    )
    log_event(db, uploaded_by, "case.upload", f"{case.case_number} uploaded and analyzed")
    return case


def get_case_detail(db: Session, case_id: str) -> Case | None:
    return db.scalar(
        select(Case)
        .where(Case.id == case_id)
        .options(
            joinedload(Case.audio_record),
            joinedload(Case.audio_features),
            joinedload(Case.text_features),
            joinedload(Case.transcription),
            joinedload(Case.assigned_to_user),
            joinedload(Case.uploaded_by_user),
        )
    )


def dashboard_metrics(db: Session) -> dict:
    cases = db.scalars(select(Case)).all()
    total = len(cases)
    high = len([case for case in cases if case.risk_level == "high"])
    medium = len([case for case in cases if case.risk_level == "medium"])
    low = len([case for case in cases if case.risk_level == "low"])
    recent = len([case for case in cases if (datetime.utcnow() - case.created_at).days < 1])
    pending_reviews = len([case for case in cases if case.review_status == "pending"])
    avg_confidence = round((sum(case.confidence_score for case in cases) / total) * 100, 1) if total else 0
    avg_duration = round(sum((case.audio_record.duration_seconds for case in cases if case.audio_record), 0.0) / total, 2) if total else 0
    return {
        "total_cases": total,
        "high_risk_cases": high,
        "recent_analyses": recent,
        "avg_recording_length": f"{avg_duration or 0.0}s",
        "distribution": {"high": high, "medium": medium, "low": low},
        "avg_confidence": avg_confidence,
        "pending_reviews": pending_reviews,
    }


def weekly_case_trend(db: Session, days: int = 7) -> dict[str, list[int] | list[str]]:
    cases = db.scalars(select(Case)).all()
    today = datetime.utcnow().date()
    window = [today.fromordinal(today.toordinal() - offset) for offset in range(days - 1, -1, -1)]
    counts = Counter(case.created_at.date() for case in cases)
    return {
        "labels": [day.strftime("%a") for day in window],
        "values": [counts.get(day, 0) for day in window],
    }


def monthly_case_volume(db: Session, months: int = 6) -> dict[str, list[int] | list[str]]:
    cases = db.scalars(select(Case)).all()
    today = datetime.utcnow()
    month_points: list[tuple[int, int]] = []
    year = today.year
    month = today.month
    for _ in range(months):
        month_points.append((year, month))
        month -= 1
        if month == 0:
            month = 12
            year -= 1
    month_points.reverse()
    counts = Counter((case.created_at.year, case.created_at.month) for case in cases)
    labels = [datetime(year=year, month=month, day=1).strftime("%b") for year, month in month_points]
    values = [counts.get((year, month), 0) for year, month in month_points]
    return {"labels": labels, "values": values}


def case_rows(db: Session) -> list[Case]:
    return db.scalars(
        select(Case)
        .order_by(Case.created_at.desc())
        .options(joinedload(Case.assigned_to_user), joinedload(Case.audio_record))
    ).all()


def user_rows(db: Session) -> list[User]:
    return db.scalars(select(User).order_by(User.created_at.asc())).all()


def audit_rows(db: Session) -> list[AuditLog]:
    return db.scalars(select(AuditLog).options(joinedload(AuditLog.user)).order_by(AuditLog.created_at.desc())).all()


def decrypted_transcript(case: Case) -> str:
    if not case.transcription:
        return ""
    return decrypt_text(case.transcription.encrypted_text)


def current_system_settings() -> dict:
    return load_system_settings()


def update_system_settings(values: dict) -> dict:
    return save_system_settings(values)

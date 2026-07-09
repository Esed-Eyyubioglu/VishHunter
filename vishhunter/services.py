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
from .models import AuditLog, AudioFeatureSet, AudioRecord, Case, Notification, TextFeatureSet, Transcription, User
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


def active_admin_rows(db: Session) -> list[User]:
    return db.scalars(
        select(User)
        .where(User.role == "administrator", User.is_active == True)
        .order_by(User.full_name.asc())
    ).all()


def create_notification(
    db: Session,
    *,
    user: User,
    title: str,
    message: str,
    event_type: str,
    case: Case | None = None,
    severity: str = "info",
) -> Notification:
    notification = Notification(
        user_id=user.id,
        case_id=case.id if case else None,
        event_type=event_type,
        severity=severity,
        title=title,
        message=message,
    )
    db.add(notification)
    return notification


def create_notifications_for_users(
    db: Session,
    users: list[User],
    *,
    title: str,
    message: str,
    event_type: str,
    case: Case | None = None,
    severity: str = "info",
) -> None:
    seen: set[str] = set()
    for target in users:
        if not target or not target.is_active or target.id in seen:
            continue
        seen.add(target.id)
        create_notification(
            db,
            user=target,
            title=title,
            message=message,
            event_type=event_type,
            case=case,
            severity=severity,
        )


def case_notification_recipients(db: Session, case: Case, *, include_admins: bool = True) -> list[User]:
    recipients: list[User] = []
    if case.uploaded_by_user and case.uploaded_by_user.is_active:
        recipients.append(case.uploaded_by_user)
    if case.assigned_to_user and case.assigned_to_user.is_active:
        recipients.append(case.assigned_to_user)
    if include_admins:
        recipients.extend(active_admin_rows(db))
    return recipients


def notification_rows_for_user(db: Session, user: User, limit: int = 20) -> list[Notification]:
    return db.scalars(
        select(Notification)
        .where(Notification.user_id == user.id)
        .order_by(Notification.created_at.desc())
        .limit(limit)
    ).all()


def unread_notification_count(db: Session, user: User) -> int:
    return int(
        db.scalar(
            select(func.count())
            .select_from(Notification)
            .where(Notification.user_id == user.id, Notification.is_read == False)
        )
        or 0
    )


def mark_notifications_read(db: Session, user: User, notification_id: str | None = None) -> None:
    query = select(Notification).where(Notification.user_id == user.id, Notification.is_read == False)
    if notification_id:
        query = query.where(Notification.id == notification_id)
    for notification in db.scalars(query):
        notification.is_read = True


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
    assigned_to: User,
) -> Case:
    storage_path, file_size = save_upload(file)
    case = Case(
        case_number=next_case_number(db),
        title=(file.filename or "Uploaded call recording").rsplit(".", 1)[0].replace("-", " ").title(),
        uploaded_by=uploaded_by.id,
        assigned_to=assigned_to.id,
        status="queued",
        risk_level="pending",
        audio_score=0.0,
        text_score=0.0,
        risk_score=0.0,
        confidence_score=0.0,
        model_verdict="pending",
        indicators=["Analysis queued"],
        summary="The recording was uploaded successfully and is waiting for background analysis.",
        review_status="analysis_pending",
    )
    db.add(case)
    db.flush()

    db.add(
        AudioRecord(
            case_id=case.id,
            original_filename=file.filename or "audio.wav",
            storage_path=storage_path,
            file_hash="",
            format=(Path(file.filename or "audio.wav").suffix or ".wav").replace(".", "").lower(),
            mime_type=file.content_type or "audio/wav",
            duration_seconds=0.0,
            file_size_bytes=file_size,
        )
    )
    log_event(db, uploaded_by, "case.upload", f"{case.case_number} uploaded and assigned to {assigned_to.email}")
    create_notifications_for_users(
        db,
        [uploaded_by],
        title="Case uploaded",
        message=f"{case.case_number} was created and queued for background analysis.",
        event_type="case.upload",
        case=case,
        severity="success",
    )
    create_notifications_for_users(
        db,
        [assigned_to],
        title="New case assigned",
        message=f"{case.case_number} was assigned to you for validation after analysis completes.",
        event_type="case.assigned",
        case=case,
        severity="info",
    )
    create_notifications_for_users(
        db,
        active_admin_rows(db),
        title="New case uploaded",
        message=f"{case.case_number} was uploaded by {uploaded_by.full_name} and assigned to {assigned_to.full_name}.",
        event_type="case.upload",
        case=case,
        severity="info",
    )
    return case


def process_case_analysis(db: Session, case_id: str) -> None:
    case = get_case_detail(db, case_id)
    if not case or not case.audio_record:
        return

    case.status = "processing"
    case.summary = "Background analysis is running: transcription, acoustic extraction, linguistic scoring, and fusion."
    case.indicators = ["Analysis in progress"]
    log_event(db, None, "case.analysis.start", f"{case.case_number} background analysis started")
    create_notifications_for_users(
        db,
        case_notification_recipients(db, case),
        title="Analysis started",
        message=f"{case.case_number} is now processing in the background.",
        event_type="case.analysis.start",
        case=case,
        severity="info",
    )
    db.commit()

    try:
        audio_path = Path(case.audio_record.storage_path)
        if not audio_path.exists():
            raise FileNotFoundError("Stored audio file is missing.")
        processed = pipeline.analyze(str(audio_path))
        case.status = "analyzed"
        case.risk_level = processed.risk_level
        case.audio_score = processed.audio_score
        case.text_score = processed.text_score
        case.risk_score = processed.risk_score
        case.confidence_score = processed.confidence
        case.model_verdict = processed.verdict
        case.indicators = processed.indicators
        case.summary = processed.summary
        case.review_status = "pending"
        case.audio_record.file_hash = processed.file_hash
        case.audio_record.duration_seconds = processed.duration_seconds
        if case.transcription:
            case.transcription.encrypted_text = encrypt_text(processed.transcript)
            case.transcription.detected_language = "en"
        else:
            db.add(Transcription(case_id=case.id, encrypted_text=encrypt_text(processed.transcript), detected_language="en"))
        if case.audio_features:
            case.audio_features.mfcc_vector = processed.mfcc_vector
            case.audio_features.spectral_contrast = processed.spectral_vector
            case.audio_features.zcr = processed.zcr
            case.audio_features.pitch_profile = processed.pitch_profile
            case.audio_features.anomalies = processed.anomalies
        else:
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
        if case.text_features:
            case.text_features.embedding_vector = processed.embedding_vector
            case.text_features.indicators = processed.indicators
            case.text_features.suspicious_phrases = processed.suspicious_phrases
        else:
            db.add(
                TextFeatureSet(
                    case_id=case.id,
                    embedding_vector=processed.embedding_vector,
                    indicators=processed.indicators,
                    suspicious_phrases=processed.suspicious_phrases,
                )
            )
        log_event(db, None, "case.analysis.complete", f"{case.case_number} background analysis completed")
        create_notifications_for_users(
            db,
            case_notification_recipients(db, case),
            title="Analysis completed",
            message=f"{case.case_number} is ready for review with {case.risk_level.title()} risk.",
            event_type="case.analysis.complete",
            case=case,
            severity="success" if case.risk_level == "low" else "warning",
        )
        db.commit()
    except Exception as exc:
        db.rollback()
        failed_case = get_case_detail(db, case_id)
        if failed_case:
            failed_case.status = "failed"
            failed_case.risk_level = "pending"
            failed_case.model_verdict = "failed"
            failed_case.review_status = "analysis_failed"
            failed_case.indicators = ["Analysis failed"]
            failed_case.summary = f"Background analysis failed: {exc}"
            log_event(db, None, "case.analysis.failed", f"{failed_case.case_number} background analysis failed: {exc}")
            create_notifications_for_users(
                db,
                case_notification_recipients(db, failed_case),
                title="Analysis failed",
                message=f"{failed_case.case_number} could not be analyzed. Error: {exc}",
                event_type="case.analysis.failed",
                case=failed_case,
                severity="danger",
            )
            db.commit()


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


def active_analyst_rows(db: Session) -> list[User]:
    return db.scalars(
        select(User)
        .where(User.role == "analyst", User.is_active == True)
        .order_by(User.full_name.asc())
    ).all()


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


def reprocess_existing_cases(db: Session) -> int:
    cases = db.scalars(
        select(Case)
        .options(
            joinedload(Case.audio_record),
            joinedload(Case.audio_features),
            joinedload(Case.text_features),
            joinedload(Case.transcription),
        )
        .order_by(Case.created_at.asc())
    ).all()
    updated = 0
    for case in cases:
        if not case.audio_record or not case.audio_record.storage_path:
            continue
        audio_path = Path(case.audio_record.storage_path)
        if not audio_path.exists():
            continue
        processed = pipeline.analyze(str(audio_path))
        case.risk_level = processed.risk_level
        case.audio_score = processed.audio_score
        case.text_score = processed.text_score
        case.risk_score = processed.risk_score
        case.confidence_score = processed.confidence
        case.model_verdict = processed.verdict
        case.indicators = processed.indicators
        case.summary = processed.summary
        case.audio_record.file_hash = processed.file_hash
        case.audio_record.duration_seconds = processed.duration_seconds
        if case.transcription:
            case.transcription.encrypted_text = encrypt_text(processed.transcript)
        if case.audio_features:
            case.audio_features.mfcc_vector = processed.mfcc_vector
            case.audio_features.spectral_contrast = processed.spectral_vector
            case.audio_features.zcr = processed.zcr
            case.audio_features.pitch_profile = processed.pitch_profile
            case.audio_features.anomalies = processed.anomalies
        if case.text_features:
            case.text_features.embedding_vector = processed.embedding_vector
            case.text_features.indicators = processed.indicators
            case.text_features.suspicious_phrases = processed.suspicious_phrases
        updated += 1
    return updated

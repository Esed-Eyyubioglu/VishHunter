from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


def utc_now() -> datetime:
    return datetime.utcnow()


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    full_name: Mapped[str] = mapped_column(String(120))
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(32), index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)

    uploaded_cases: Mapped[list["Case"]] = relationship(
        back_populates="uploaded_by_user",
        foreign_keys="Case.uploaded_by",
    )
    assigned_cases: Mapped[list["Case"]] = relationship(
        back_populates="assigned_to_user",
        foreign_keys="Case.assigned_to",
    )
    audit_logs: Mapped[list["AuditLog"]] = relationship(back_populates="user")


class Case(Base):
    __tablename__ = "cases"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    case_number: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    title: Mapped[str] = mapped_column(String(200))
    uploaded_by: Mapped[str | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    assigned_to: Mapped[str | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="analyzed")
    risk_level: Mapped[str] = mapped_column(String(16), default="medium")
    audio_score: Mapped[float] = mapped_column(Float, default=0.0)
    text_score: Mapped[float] = mapped_column(Float, default=0.0)
    risk_score: Mapped[float] = mapped_column(Float, default=0.5)
    confidence_score: Mapped[float] = mapped_column(Float, default=0.5)
    model_verdict: Mapped[str] = mapped_column(String(16), default="safe")
    review_status: Mapped[str] = mapped_column(String(32), default="pending")
    review_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    indicators: Mapped[list[str]] = mapped_column(JSON, default=list)
    summary: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now, onupdate=utc_now)
    validated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    uploaded_by_user: Mapped[User | None] = relationship(back_populates="uploaded_cases", foreign_keys=[uploaded_by])
    assigned_to_user: Mapped[User | None] = relationship(back_populates="assigned_cases", foreign_keys=[assigned_to])
    audio_record: Mapped["AudioRecord | None"] = relationship(back_populates="case", uselist=False, cascade="all, delete-orphan")
    transcription: Mapped["Transcription | None"] = relationship(back_populates="case", uselist=False, cascade="all, delete-orphan")
    audio_features: Mapped["AudioFeatureSet | None"] = relationship(back_populates="case", uselist=False, cascade="all, delete-orphan")
    text_features: Mapped["TextFeatureSet | None"] = relationship(back_populates="case", uselist=False, cascade="all, delete-orphan")


class AudioRecord(Base):
    __tablename__ = "audio_records"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    case_id: Mapped[str] = mapped_column(ForeignKey("cases.id"), unique=True)
    original_filename: Mapped[str] = mapped_column(String(255))
    storage_path: Mapped[str] = mapped_column(String(500))
    file_hash: Mapped[str] = mapped_column(String(128), default="")
    format: Mapped[str] = mapped_column(String(32), default="")
    mime_type: Mapped[str] = mapped_column(String(120))
    duration_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    file_size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)

    case: Mapped[Case] = relationship(back_populates="audio_record")


class Transcription(Base):
    __tablename__ = "transcriptions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    case_id: Mapped[str] = mapped_column(ForeignKey("cases.id"), unique=True)
    encrypted_text: Mapped[str] = mapped_column(Text)
    detected_language: Mapped[str] = mapped_column(String(16), default="en")
    generated_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)

    case: Mapped[Case] = relationship(back_populates="transcription")


class AudioFeatureSet(Base):
    __tablename__ = "audio_feature_sets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    case_id: Mapped[str] = mapped_column(ForeignKey("cases.id"), unique=True)
    mfcc_vector: Mapped[list[float]] = mapped_column(JSON, default=list)
    spectral_contrast: Mapped[list[float]] = mapped_column(JSON, default=list)
    zcr: Mapped[float] = mapped_column(Float, default=0.0)
    pitch_profile: Mapped[list[float]] = mapped_column(JSON, default=list)
    anomalies: Mapped[list[str]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)

    case: Mapped[Case] = relationship(back_populates="audio_features")


class TextFeatureSet(Base):
    __tablename__ = "text_feature_sets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    case_id: Mapped[str] = mapped_column(ForeignKey("cases.id"), unique=True)
    embedding_vector: Mapped[list[float]] = mapped_column(JSON, default=list)
    indicators: Mapped[list[str]] = mapped_column(JSON, default=list)
    suspicious_phrases: Mapped[list[str]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)

    case: Mapped[Case] = relationship(back_populates="text_features")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    action: Mapped[str] = mapped_column(String(120))
    details: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)

    user: Mapped[User | None] = relationship(back_populates="audit_logs")

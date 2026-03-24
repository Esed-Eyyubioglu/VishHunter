from __future__ import annotations

import csv
import hashlib
import json
import threading
from dataclasses import dataclass
from pathlib import Path

import joblib
import librosa
import numpy as np
import soundfile as sf
import torch
from sklearn.base import clone
from faster_whisper import WhisperModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.svm import SVC
from transformers import AutoModel, AutoTokenizer
from xgboost import XGBClassifier

from .config import BASE_DIR, settings
from .system_settings import load_system_settings


TEXT_MODEL_NAME = "distilbert-base-uncased"
WHISPER_MODEL_NAME = "tiny.en"


@dataclass
class ProcessedAudio:
    transcript: str
    duration_seconds: float
    file_hash: str
    mfcc_vector: list[float]
    spectral_vector: list[float]
    zcr: float
    pitch_profile: list[float]
    anomalies: list[str]
    embedding_vector: list[float]
    indicators: list[str]
    suspicious_phrases: list[str]
    benign_markers: list[str]
    audio_score: float
    text_score: float
    risk_score: float
    confidence: float
    risk_level: str
    verdict: str
    summary: str


class HybridPipeline:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._tokenizer = None
        self._distilbert = None
        self._whisper = None
        self._audio_model = None
        self._text_model = None
        self._fusion_model = None
        self.artifact_dir = settings.artifact_dir
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.audio_model_path = self.artifact_dir / "audio_xgb.joblib"
        self.text_model_path = self.artifact_dir / "text_svm.joblib"
        self.fusion_model_path = self.artifact_dir / "fusion_logreg.joblib"
        self.metadata_path = self.artifact_dir / "training_metadata.json"

    def model_status(self) -> dict:
        conf = load_system_settings()
        return {
            "artifacts_present": all(
                path.exists()
                for path in (self.audio_model_path, self.text_model_path, self.fusion_model_path)
            ),
            "metadata": self._load_metadata(),
            "settings": conf,
        }

    def _load_metadata(self) -> dict:
        if not self.metadata_path.exists():
            return {}
        with open(self.metadata_path, "r", encoding="utf-8") as file:
            return json.load(file)

    def _save_metadata(self, data: dict) -> None:
        with open(self.metadata_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)

    def ensure_models(self) -> None:
        with self._lock:
            if self._audio_model and self._text_model and self._fusion_model:
                return
            if self.audio_model_path.exists() and self.text_model_path.exists() and self.fusion_model_path.exists():
                self._audio_model = joblib.load(self.audio_model_path)
                self._text_model = joblib.load(self.text_model_path)
                self._fusion_model = joblib.load(self.fusion_model_path)
                return
            self.train_from_public_datasets(force_retrain=True)

    def train_from_public_datasets(self, force_retrain: bool = False) -> dict:
        with self._lock:
            if (
                not force_retrain
                and self.audio_model_path.exists()
                and self.text_model_path.exists()
                and self.fusion_model_path.exists()
            ):
                if self._audio_model is None:
                    self._audio_model = joblib.load(self.audio_model_path)
                    self._text_model = joblib.load(self.text_model_path)
                    self._fusion_model = joblib.load(self.fusion_model_path)
                return self._load_metadata()

            self._ensure_text_encoder()
            examples = self._load_public_training_examples()
            if len(examples) < 20:
                raise RuntimeError("Not enough real public training examples were found.")

            audio_vectors = []
            text_vectors = []
            labels = []
            for audio_path, transcript, label in examples:
                feature_vector, _ = self.audio_features(audio_path)
                audio_vectors.append(feature_vector)
                text_vectors.append(self.encode_text(transcript))
                labels.append(label)

            audio_vectors_np = np.vstack(audio_vectors)
            text_vectors_np = np.vstack(text_vectors)
            labels_np = np.array(labels, dtype=np.int32)

            audio_model = XGBClassifier(
                n_estimators=80,
                max_depth=3,
                learning_rate=0.08,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                objective="binary:logistic",
                eval_metric="logloss",
            )
            text_model = SVC(kernel="linear", probability=True, random_state=42, class_weight="balanced")
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            audio_probs = cross_val_predict(clone(audio_model), audio_vectors_np, labels_np, cv=cv, method="predict_proba")[:, 1]
            text_probs = cross_val_predict(clone(text_model), text_vectors_np, labels_np, cv=cv, method="predict_proba")[:, 1]
            fusion_features = np.column_stack([audio_probs, text_probs, np.abs(audio_probs - text_probs)])

            audio_model.fit(audio_vectors_np, labels_np)
            text_model.fit(text_vectors_np, labels_np)

            fusion_model = LogisticRegression(random_state=42, max_iter=400, class_weight="balanced")
            fusion_model.fit(fusion_features, labels_np)

            self._audio_model = audio_model
            self._text_model = text_model
            self._fusion_model = fusion_model
            joblib.dump(audio_model, self.audio_model_path)
            joblib.dump(text_model, self.text_model_path)
            joblib.dump(fusion_model, self.fusion_model_path)

            metadata = {
                "trained_at": str(np.datetime64("now")),
                "sample_count": len(examples),
                "fraud_samples": int(labels_np.sum()),
                "benign_samples": int((labels_np == 0).sum()),
                "positive_dataset": load_system_settings()["positive_dataset_path"],
                "negative_dataset": load_system_settings()["negative_dataset_path"],
            }
            self._save_metadata(metadata)
            return metadata

    def _ensure_text_encoder(self) -> None:
        if self._tokenizer and self._distilbert:
            return
        self._tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME, cache_dir=str(settings.model_cache_dir))
        self._distilbert = AutoModel.from_pretrained(TEXT_MODEL_NAME, cache_dir=str(settings.model_cache_dir))
        self._distilbert.eval()
        torch.set_num_threads(1)

    def _ensure_whisper(self) -> None:
        if self._whisper is None:
            self._whisper = WhisperModel(
                WHISPER_MODEL_NAME,
                device="cpu",
                compute_type="int8",
                download_root=str(settings.model_cache_dir),
            )

    def encode_text(self, text: str) -> np.ndarray:
        self._ensure_text_encoder()
        tokens = self._tokenizer(text, truncation=True, padding=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            outputs = self._distilbert(**tokens)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
        return embedding.astype(np.float32)

    def transcribe(self, audio_path: str, transcript_override: str | None = None) -> str:
        if transcript_override and transcript_override.strip():
            return transcript_override.strip()
        self._ensure_whisper()
        segments, _ = self._whisper.transcribe(audio_path, vad_filter=True, beam_size=3)
        transcript = " ".join(segment.text.strip() for segment in segments).strip()
        return transcript or "No intelligible speech could be recovered from the audio."

    def audio_features(self, audio_path: str) -> tuple[np.ndarray, dict]:
        waveform, sr = librosa.load(audio_path, sr=16000, mono=True)
        duration = float(librosa.get_duration(y=waveform, sr=sr))
        mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=13)
        spectral = librosa.feature.spectral_contrast(y=waveform, sr=sr)
        zcr = float(librosa.feature.zero_crossing_rate(y=waveform).mean())
        pitches, magnitudes = librosa.piptrack(y=waveform, sr=sr)
        pitch_values = pitches[magnitudes > np.median(magnitudes)]
        pitch_profile = np.percentile(pitch_values, [10, 25, 50, 75, 90]).tolist() if pitch_values.size else [0.0] * 5
        rms = float(librosa.feature.rms(y=waveform).mean())
        centroid = float(librosa.feature.spectral_centroid(y=waveform, sr=sr).mean())

        flattened = np.concatenate(
            [
                mfcc.mean(axis=1),
                spectral.mean(axis=1),
                np.array([zcr, rms, centroid / 10000.0, duration / 120.0], dtype=np.float32),
            ]
        ).astype(np.float32)

        anomalies = []
        if zcr > 0.1:
            anomalies.append("Elevated zero-crossing rate")
        if rms < 0.03:
            anomalies.append("Low signal energy detected")
        if centroid > 3200:
            anomalies.append("High spectral centroid suggests bright or synthetic timbre")
        if not anomalies:
            anomalies.append("No strong acoustic anomaly detected")

        return flattened, {
            "duration": duration,
            "mfcc_vector": np.round(mfcc.mean(axis=1), 4).tolist(),
            "spectral_vector": np.round(spectral.mean(axis=1), 4).tolist(),
            "zcr": round(zcr, 4),
            "pitch_profile": [round(value, 2) for value in pitch_profile],
            "anomalies": anomalies,
        }

    def analyze(self, audio_path: str, transcript_override: str | None = None) -> ProcessedAudio:
        self.ensure_models()
        transcript = self.transcribe(audio_path, transcript_override=transcript_override)
        audio_vector, audio_meta = self.audio_features(audio_path)
        embedding = self.encode_text(transcript)

        raw_audio_score = float(self._audio_model.predict_proba(audio_vector.reshape(1, -1))[0, 1])
        raw_text_score = float(self._text_model.predict_proba(embedding.reshape(1, -1))[0, 1])

        current = load_system_settings()
        high_threshold = float(current["high_threshold"])
        medium_threshold = float(current["medium_threshold"])
        rule_score, indicators, suspicious_phrases, benign_markers = linguistic_signal_score(transcript)

        effective_audio_score = raw_audio_score
        effective_text_score = (raw_text_score * 0.55) + (rule_score * 0.45)
        if not suspicious_phrases:
            effective_audio_score = min(effective_audio_score, 0.45 if benign_markers else 0.6)
            effective_text_score = min(effective_text_score, 0.4 if benign_markers else effective_text_score)

        fusion_input = np.array(
            [[effective_audio_score, effective_text_score, abs(effective_audio_score - effective_text_score)]],
            dtype=np.float32,
        )
        model_risk_score = float(self._fusion_model.predict_proba(fusion_input)[0, 1])
        risk_score = (model_risk_score * 0.55) + (rule_score * 0.45)
        if not suspicious_phrases and len(benign_markers) >= 2:
            risk_score = min(risk_score, 0.24)
        elif not suspicious_phrases and benign_markers:
            risk_score = min(risk_score, 0.34)

        risk_level = "high" if risk_score >= high_threshold else "medium" if risk_score >= medium_threshold else "low"
        verdict = "fraud" if risk_score >= medium_threshold else "safe"
        confidence = max(rule_score, effective_audio_score, effective_text_score)

        if effective_text_score > effective_audio_score:
            indicators.append("Linguistic model contribution dominates the final score")
        else:
            indicators.append("Acoustic model contribution dominates the final score")
        indicators.extend(audio_meta["anomalies"][:2])
        if benign_markers:
            indicators.append("Benign service-call language detected")

        summary = (
            f"Hybrid fusion classified the call as {risk_level} risk with "
            f"{confidence:.0%} confidence using audio and linguistic signals."
        )

        file_hash = sha256_file(audio_path)
        return ProcessedAudio(
            transcript=transcript,
            duration_seconds=audio_meta["duration"],
            file_hash=file_hash,
            mfcc_vector=audio_meta["mfcc_vector"],
            spectral_vector=audio_meta["spectral_vector"],
            zcr=audio_meta["zcr"],
            pitch_profile=audio_meta["pitch_profile"],
            anomalies=audio_meta["anomalies"],
            embedding_vector=np.round(embedding[:24], 4).tolist(),
            indicators=indicators,
            suspicious_phrases=suspicious_phrases,
            benign_markers=benign_markers,
            audio_score=round(effective_audio_score, 4),
            text_score=round(effective_text_score, 4),
            risk_score=round(risk_score, 4),
            confidence=round(confidence, 4),
            risk_level=risk_level,
            verdict=verdict,
            summary=summary,
        )

    def _load_public_training_examples(self) -> list[tuple[str, str, int]]:
        current = load_system_settings()
        max_samples = int(current["max_training_samples"])
        positive_root = Path(current["positive_dataset_path"])
        negative_root = Path(current["negative_dataset_path"])

        fraud_examples = load_robocall_examples(positive_root, max_samples)
        benign_examples = load_harper_valley_examples(negative_root, max_samples, self.artifact_dir)
        return fraud_examples + benign_examples


def load_robocall_examples(dataset_root: Path, max_samples: int) -> list[tuple[str, str, int]]:
    metadata_path = dataset_root / "metadata.csv"
    if not metadata_path.exists():
        raise RuntimeError(f"Robocall dataset not found at {dataset_root}")

    rows: list[tuple[str, str, int]] = []
    with open(metadata_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row.get("language") != "en":
                continue
            audio_file = dataset_root / row["file_name"]
            transcript = (row.get("transcript") or "").strip()
            if audio_file.exists() and transcript:
                rows.append((str(audio_file), transcript, 1))
            if len(rows) >= max_samples:
                break
    return rows


def load_harper_valley_examples(dataset_root: Path, max_samples: int, artifact_dir: Path) -> list[tuple[str, str, int]]:
    agent_audio_root = dataset_root / "data" / "audio" / "agent"
    caller_audio_root = dataset_root / "data" / "audio" / "caller"
    transcript_root = dataset_root / "data" / "transcript"
    mixed_root = artifact_dir / "harper_valley_mix"
    mixed_root.mkdir(parents=True, exist_ok=True)
    if not agent_audio_root.exists() or not caller_audio_root.exists() or not transcript_root.exists():
        raise RuntimeError(f"Harper Valley dataset not found at {dataset_root}")

    rows: list[tuple[str, str, int]] = []
    for agent_audio_file in sorted(agent_audio_root.glob("*.wav")):
        sid = agent_audio_file.stem
        caller_audio_file = caller_audio_root / agent_audio_file.name
        transcript_file = transcript_root / f"{sid}.json"
        if not transcript_file.exists() or not caller_audio_file.exists():
            continue
        with open(transcript_file, "r", encoding="utf-8") as file:
            segments = json.load(file)
        transcript_parts = []
        for segment in segments:
            text = (segment.get("human_transcript") or "").strip()
            if text:
                transcript_parts.append(text)
        transcript = " ".join(transcript_parts).strip()
        if transcript:
            mixed_audio_file = mixed_root / f"{sid}.wav"
            if not mixed_audio_file.exists():
                build_mixed_call_audio(agent_audio_file, caller_audio_file, mixed_audio_file)
            rows.append((str(mixed_audio_file), transcript, 0))
        if len(rows) >= max_samples:
            break
    return rows


def build_mixed_call_audio(agent_audio_path: Path, caller_audio_path: Path, output_path: Path) -> None:
    agent_waveform, agent_sr = librosa.load(str(agent_audio_path), sr=16000, mono=True)
    caller_waveform, caller_sr = librosa.load(str(caller_audio_path), sr=16000, mono=True)
    if agent_sr != caller_sr:
        raise RuntimeError("Harper Valley audio sample rates are inconsistent.")
    max_length = max(len(agent_waveform), len(caller_waveform))
    agent_padded = np.pad(agent_waveform, (0, max_length - len(agent_waveform)))
    caller_padded = np.pad(caller_waveform, (0, max_length - len(caller_waveform)))
    mixed_waveform = ((agent_padded + caller_padded) / 2.0).astype(np.float32)
    sf.write(output_path, mixed_waveform, agent_sr)


def linguistic_signal_score(transcript: str) -> tuple[float, list[str], list[str], list[str]]:
    suspicious_phrases_map = {
        "urgent": 0.18,
        "immediately": 0.18,
        "otp": 0.28,
        "password": 0.28,
        "verification code": 0.26,
        "one-time password": 0.28,
        "security team": 0.22,
        "bank": 0.08,
        "account": 0.08,
        "verify your identity": 0.24,
        "transfer": 0.18,
        "wire": 0.18,
        "gift card": 0.3,
        "social security": 0.3,
        "suspend": 0.16,
        "locked": 0.12,
    }
    benign_phrases = [
        "thank you for calling",
        "how may i assist",
        "how can i help",
        "i'd like to order",
        "shipping address",
        "customer number",
        "credit card",
        "price",
        "order",
        "vehicle",
        "flowers",
        "delivery",
    ]
    lower = transcript.lower()
    suspicious = [phrase for phrase in suspicious_phrases_map if phrase in lower]
    benign_markers = [phrase for phrase in benign_phrases if phrase in lower]
    indicators = []
    if any(phrase in lower for phrase in ("urgent", "immediately")):
        indicators.append("Urgency pressure detected")
    if any(phrase in lower for phrase in ("bank", "security team")):
        indicators.append("Institution impersonation detected")
    if any(phrase in lower for phrase in ("otp", "password", "verification code", "one-time password", "verify your identity")):
        indicators.append("Credential harvesting language detected")
    if any(phrase in lower for phrase in ("transfer", "wire", "gift card", "social security")):
        indicators.append("Sensitive payment or identity extraction detected")
    if benign_markers and not indicators:
        indicators.append("Routine service-oriented conversation detected")
    elif not indicators:
        indicators.append("Low linguistic coercion markers")
    score = min(0.98, sum(suspicious_phrases_map[phrase] for phrase in suspicious))
    if suspicious and "bank" in suspicious and any(phrase in suspicious for phrase in ("otp", "password", "verification code", "one-time password")):
        score = min(0.98, score + 0.18)
    if benign_markers:
        score = max(0.03, score - (0.08 * min(len(benign_markers), 3)))
    if not suspicious:
        score = 0.08 if not benign_markers else 0.04
    return score, indicators, suspicious, benign_markers


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


pipeline = HybridPipeline()

from __future__ import annotations

import csv
import hashlib
import json
import re
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
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.svm import SVC
from transformers import AutoModel, AutoTokenizer
from xgboost import XGBClassifier

from .config import BASE_DIR, settings
from .system_settings import load_system_settings


TEXT_MODEL_NAME = "distilbert-base-uncased"
WHISPER_MODEL_NAME = "tiny.en"
KAGGLE_SCAM_TEXT_DATASET = "teeconnie/scam-and-non-scam-call-conversation-dataset"
KAGGLE_SCAM_LABEL_DATASET = "mealss/call-transcripts-scam-determinations"


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
        self.artifact_dir = settings.artifact_dir
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.audio_model_path = self.artifact_dir / "audio_xgb.joblib"
        self.text_model_path = self.artifact_dir / "text_svm.joblib"
        self.metadata_path = self.artifact_dir / "training_metadata.json"

    def model_status(self) -> dict:
        conf = load_system_settings()
        return {
            "artifacts_present": all(
                path.exists()
                for path in (self.audio_model_path, self.text_model_path)
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
            if self._audio_model and self._text_model:
                return
            if self.audio_model_path.exists() and self.text_model_path.exists():
                self._audio_model = joblib.load(self.audio_model_path)
                self._text_model = joblib.load(self.text_model_path)
                return
            self.train_from_public_datasets(force_retrain=True)

    def train_from_public_datasets(self, force_retrain: bool = False) -> dict:
        with self._lock:
            if (
                not force_retrain
                and self.audio_model_path.exists()
                and self.text_model_path.exists()
            ):
                if self._audio_model is None:
                    self._audio_model = joblib.load(self.audio_model_path)
                    self._text_model = joblib.load(self.text_model_path)
                return self._load_metadata()

            self._ensure_text_encoder()
            audio_examples = self._load_public_training_examples()
            text_only_examples = self._load_public_text_examples()
            if len(audio_examples) < 20:
                raise RuntimeError("Not enough real public training examples were found.")
            if len(text_only_examples) < 20:
                raise RuntimeError("Not enough transcript-only training examples were found.")

            audio_vectors = []
            audio_text_vectors = []
            audio_labels = []
            for audio_path, transcript, label in audio_examples:
                feature_vector, _ = self.audio_features(audio_path)
                audio_vectors.append(feature_vector)
                audio_text_vectors.append(self.encode_text(transcript))
                audio_labels.append(label)

            combined_text_examples = [(transcript, label) for _, transcript, label in audio_examples] + text_only_examples
            combined_text_vectors = [self.encode_text(transcript) for transcript, _ in combined_text_examples]
            combined_text_labels = [label for _, label in combined_text_examples]

            audio_vectors_np = np.vstack(audio_vectors)
            audio_text_vectors_np = np.vstack(audio_text_vectors)
            audio_labels_np = np.array(audio_labels, dtype=np.int32)
            combined_text_vectors_np = np.vstack(combined_text_vectors)
            combined_text_labels_np = np.array(combined_text_labels, dtype=np.int32)

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
            audio_cv = build_stratified_cv(audio_labels_np)
            text_cv = build_stratified_cv(combined_text_labels_np)
            audio_probs = cross_val_predict(
                clone(audio_model),
                audio_vectors_np,
                audio_labels_np,
                cv=audio_cv,
                method="predict_proba",
            )[:, 1]
            text_probs_audio = cross_val_predict(
                clone(text_model),
                audio_text_vectors_np,
                audio_labels_np,
                cv=audio_cv,
                method="predict_proba",
            )[:, 1]
            text_probs_combined = cross_val_predict(
                clone(text_model),
                combined_text_vectors_np,
                combined_text_labels_np,
                cv=text_cv,
                method="predict_proba",
            )[:, 1]
            weighted_fusion_probs = []
            for audio_prob, text_prob, (_, transcript, _) in zip(audio_probs, text_probs_audio, audio_examples):
                rule_score, _, suspicious_phrases, benign_markers = linguistic_signal_score(transcript)
                effective_audio_score = adjust_audio_score(audio_prob, suspicious_phrases, benign_markers)
                weighted_fusion_probs.append(weighted_fusion_score(effective_audio_score, text_prob, rule_score))
            weighted_fusion_probs_np = np.array(weighted_fusion_probs, dtype=np.float32)

            audio_model.fit(audio_vectors_np, audio_labels_np)
            text_model.fit(combined_text_vectors_np, combined_text_labels_np)

            self._audio_model = audio_model
            self._text_model = text_model
            joblib.dump(audio_model, self.audio_model_path)
            joblib.dump(text_model, self.text_model_path)

            metadata = {
                "trained_at": str(np.datetime64("now")),
                "sample_count": len(audio_examples) + len(text_only_examples),
                "audio_sample_count": len(audio_examples),
                "text_only_sample_count": len(text_only_examples),
                "fraud_samples": int(combined_text_labels_np.sum()),
                "benign_samples": int((combined_text_labels_np == 0).sum()),
                "audio_fraud_samples": int(audio_labels_np.sum()),
                "audio_benign_samples": int((audio_labels_np == 0).sum()),
                "positive_dataset": load_system_settings()["positive_dataset_path"],
                "negative_dataset": load_system_settings()["negative_dataset_path"],
                "text_datasets": [KAGGLE_SCAM_TEXT_DATASET, KAGGLE_SCAM_LABEL_DATASET],
                "dataset_scope": {
                    "positive_domain": "public robocall audio plus scam transcript corpora",
                    "negative_domain": "public customer-service and banking conversations plus non-scam transcript corpora",
                    "coverage_warning": "The model is broader than before, but public datasets still do not fully cover every real-world call type, accent, language variant, or vishing tactic.",
                },
                "validation_metrics": {
                    "audio_branch": evaluate_probabilities(audio_labels_np, audio_probs, threshold=0.5),
                    "text_branch": evaluate_probabilities(combined_text_labels_np, text_probs_combined, threshold=0.5),
                    "fusion_branch": evaluate_probabilities(audio_labels_np, weighted_fusion_probs_np, threshold=0.5),
                },
                "fusion_method": {
                    "type": "weighted_fusion_scoring",
                    "audio_weight": 0.30,
                    "text_weight": 0.35,
                    "rule_weight": 0.35,
                    "description": "Deterministic weighted scoring function combining XGBoost audio, SVM text, and linguistic rule scores.",
                },
                "recommended_thresholds": recommend_thresholds(audio_labels_np, weighted_fusion_probs_np),
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

        effective_audio_score = adjust_audio_score(raw_audio_score, suspicious_phrases, benign_markers)
        effective_text_score = raw_text_score
        if not suspicious_phrases:
            effective_text_score = min(raw_text_score, 0.4 if benign_markers else raw_text_score)

        risk_score = weighted_fusion_score(effective_audio_score, effective_text_score, rule_score)
        if not suspicious_phrases and len(benign_markers) >= 2 and raw_audio_score < 0.8:
            risk_score = min(risk_score, 0.24)
        elif not suspicious_phrases and benign_markers and raw_audio_score < 0.8:
            risk_score = min(risk_score, 0.34)

        risk_level = "high" if risk_score >= high_threshold else "medium" if risk_score >= medium_threshold else "low"
        verdict = "fraud" if risk_score >= medium_threshold else "safe"
        confidence = max(risk_score, 1.0 - risk_score)

        if effective_text_score > effective_audio_score:
            indicators.append("Linguistic model contribution dominates the final score")
        else:
            indicators.append("Acoustic model contribution dominates the final score")
        indicators.extend(audio_meta["anomalies"][:2])
        if benign_markers:
            indicators.append("Benign service-call language detected")

        summary = (
            f"Weighted fusion scoring classified the call as {risk_level} risk with "
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

    def _load_public_text_examples(self) -> list[tuple[str, int]]:
        current = load_system_settings()
        max_samples = int(current["max_training_samples"])
        scam_examples, benign_examples = load_kaggle_scambait_examples(max_samples=max_samples)
        better_examples = load_better30_examples(max_samples=max_samples)
        return scam_examples + benign_examples + better_examples


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


def load_kaggle_scambait_examples(max_samples: int) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    dataset_root = ensure_kaggle_dataset(KAGGLE_SCAM_TEXT_DATASET)
    scam_path = dataset_root / "English_Scam.txt"
    non_scam_path = dataset_root / "English_NonScam.txt"
    if not scam_path.exists() or not non_scam_path.exists():
        raise RuntimeError(f"Kaggle text dataset not found at {dataset_root}")

    scam_examples = []
    for line in read_non_empty_lines(scam_path):
        cleaned = re.sub(r"^\d+\.\s*", "", line).strip()
        if cleaned:
            scam_examples.append((cleaned, 1))
        if len(scam_examples) >= max_samples * 2:
            break

    benign_examples = []
    for line in read_non_empty_lines(non_scam_path):
        if line:
            benign_examples.append((line, 0))
        if len(benign_examples) >= max_samples * 2:
            break
    return scam_examples, benign_examples


def load_better30_examples(max_samples: int) -> list[tuple[str, int]]:
    dataset_root = ensure_kaggle_dataset(KAGGLE_SCAM_LABEL_DATASET)
    csv_path = dataset_root / "BETTER30.csv"
    if not csv_path.exists():
        raise RuntimeError(f"BETTER30 dataset not found at {dataset_root}")

    examples: list[tuple[str, int]] = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            text = (row.get("TEXT") or "").strip()
            label = normalize_better30_label(row.get("LABEL") or "")
            if text and label is not None:
                examples.append((text, label))
            if len(examples) >= max_samples * 3:
                break
    return examples


def normalize_better30_label(raw_label: str) -> int | None:
    label = raw_label.strip().strip('"').lower()
    if not label:
        return None
    positive_markers = (
        "scam",
        "suspicious",
        "potential_scam",
        "highly_suspicious",
        "slightly_suspicious",
        "urgency",
        "dangerous",
        "dismissing official protocols",
    )
    negative_markers = (
        "neutral",
        "legitimate",
        "standard_opening",
        "polite_ending",
        "adhering to protocols",
        "emphasizing security and compliance",
    )
    if any(marker in label for marker in positive_markers):
        return 1
    if any(marker in label for marker in negative_markers):
        return 0
    return None


def ensure_kaggle_dataset(dataset_slug: str) -> Path:
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError("kagglehub is required to download the Kaggle transcript datasets.") from exc
    return Path(kagglehub.dataset_download(dataset_slug))


def read_non_empty_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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
        "voucher": 0.28,
        "bitcoin": 0.32,
        "crypto": 0.28,
        "social security": 0.3,
        "suspend": 0.16,
        "locked": 0.12,
        "remote access": 0.32,
        "anydesk": 0.35,
        "teamviewer": 0.35,
        "refund": 0.14,
        "technical support": 0.24,
        "tech support": 0.24,
        "windows support": 0.28,
        "microsoft certified technicians": 0.34,
        "computer technical department": 0.32,
        "security check-up": 0.18,
        "warning messages": 0.16,
        "virus": 0.2,
        "computer slow": 0.16,
        "stay on the line": 0.18,
        "do not tell anyone": 0.28,
        "processing fee": 0.22,
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
    if any(phrase in lower for phrase in ("technical support", "tech support", "windows support", "microsoft certified technicians", "computer technical department", "virus")):
        indicators.append("Tech-support scam language detected")
    if benign_markers and not indicators:
        indicators.append("Routine service-oriented conversation detected")
    elif not indicators:
        indicators.append("Low linguistic coercion markers")
    score = min(0.98, sum(suspicious_phrases_map[phrase] for phrase in suspicious))
    if suspicious and "bank" in suspicious and any(phrase in suspicious for phrase in ("otp", "password", "verification code", "one-time password")):
        score = min(0.98, score + 0.18)
    if any(phrase in suspicious for phrase in ("remote access", "anydesk", "teamviewer")):
        score = min(0.98, score + 0.12)
    if any(phrase in suspicious for phrase in ("windows support", "microsoft certified technicians", "computer technical department")):
        score = min(0.98, score + 0.16)
    if benign_markers:
        score = max(0.03, score - (0.08 * min(len(benign_markers), 3)))
    if not suspicious:
        score = 0.08 if not benign_markers else 0.04
    return score, indicators, suspicious, benign_markers


def adjust_audio_score(raw_audio_score: float, suspicious_phrases: list[str], benign_markers: list[str]) -> float:
    effective_audio_score = float(np.clip(raw_audio_score, 0.0, 1.0))
    if not suspicious_phrases:
        if benign_markers and raw_audio_score < 0.8:
            effective_audio_score = min(effective_audio_score, 0.45)
        elif raw_audio_score < 0.85:
            effective_audio_score = min(effective_audio_score, 0.6)
    return effective_audio_score


def weighted_fusion_score(audio_score: float, text_score: float, rule_score: float) -> float:
    """Combine branch outputs with fixed, transparent prototype weights."""
    audio_component = 0.30 * float(np.clip(audio_score, 0.0, 1.0))
    text_component = 0.35 * float(np.clip(text_score, 0.0, 1.0))
    rule_component = 0.35 * float(np.clip(rule_score, 0.0, 1.0))
    return float(np.clip(audio_component + text_component + rule_component, 0.0, 0.98))


def build_stratified_cv(labels: np.ndarray) -> StratifiedKFold:
    _, counts = np.unique(labels, return_counts=True)
    min_count = int(counts.min()) if len(counts) else 2
    n_splits = max(2, min(5, min_count))
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)


def evaluate_probabilities(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "auc": round(float(roc_auc_score(labels, probabilities)), 4),
        "precision": round(float(precision_score(labels, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(labels, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(labels, predictions, zero_division=0)), 4),
        "false_positive_rate": round(float(fp / (fp + tn)) if (fp + tn) else 0.0, 4),
        "threshold": round(float(threshold), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def recommend_thresholds(labels: np.ndarray, fusion_probabilities: np.ndarray) -> dict:
    benign = fusion_probabilities[labels == 0]
    medium_threshold = max(0.45, float(np.quantile(benign, 0.95)))
    high_threshold = max(0.72, medium_threshold + 0.12, float(np.quantile(benign, 0.99)))
    high_threshold = min(high_threshold, 0.95)
    medium_threshold = min(medium_threshold, high_threshold - 0.05)
    return {
        "medium": round(medium_threshold, 4),
        "high": round(high_threshold, 4),
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


pipeline = HybridPipeline()

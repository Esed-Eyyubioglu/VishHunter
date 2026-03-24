from __future__ import annotations

import json
from pathlib import Path

from .config import BASE_DIR, settings


SETTINGS_PATH = settings.artifact_dir / "system_settings.json"


DEFAULT_SETTINGS = {
    "high_threshold": 0.72,
    "medium_threshold": 0.45,
    "max_training_samples": 120,
    "positive_dataset_path": str((BASE_DIR / "external-data" / "robocall-audio-dataset").resolve()),
    "negative_dataset_path": str((BASE_DIR / "external-data" / "harper-valley").resolve()),
}


def load_system_settings() -> dict:
    settings.artifact_dir.mkdir(parents=True, exist_ok=True)
    if not SETTINGS_PATH.exists():
        save_system_settings(DEFAULT_SETTINGS.copy())
    with open(SETTINGS_PATH, "r", encoding="utf-8") as file:
        current = json.load(file)
    merged = DEFAULT_SETTINGS.copy()
    merged.update(current)
    return merged


def save_system_settings(values: dict) -> dict:
    settings.artifact_dir.mkdir(parents=True, exist_ok=True)
    merged = DEFAULT_SETTINGS.copy()
    merged.update(values)
    with open(SETTINGS_PATH, "w", encoding="utf-8") as file:
        json.dump(merged, file, indent=2)
    return merged

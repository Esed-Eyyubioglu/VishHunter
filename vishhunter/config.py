from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    app_name: str = "VishHunter"
    secret_key: str = os.getenv("VISHHUNTER_SECRET_KEY", "change-this-in-production")
    encryption_key: str = os.getenv(
        "VISHHUNTER_ENCRYPTION_KEY",
        "vishhunter-demo-encryption-key-change-me",
    )
    algorithm: str = "HS256"
    access_token_expire_minutes: int = int(os.getenv("VISHHUNTER_TOKEN_TTL_MINUTES", "480"))
    database_url: str = os.getenv(
        "DATABASE_URL",
        f"sqlite:///{(BASE_DIR / 'vishhunter.db').as_posix()}",
    )
    upload_dir: Path = BASE_DIR / "uploads"
    artifact_dir: Path = BASE_DIR / "artifacts"
    model_cache_dir: Path = BASE_DIR / "model_cache"
    static_dir: Path = BASE_DIR / "static"
    template_dir: Path = BASE_DIR / "templates"


settings = Settings()

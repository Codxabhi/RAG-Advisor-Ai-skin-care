from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str
    chat_model: str
    embedding_model: str
    knowledge_path: Path
    eval_path: Path

    @property
    def has_key(self) -> bool:
        key = self.api_key.strip()
        return bool(key) and key not in {"sk-your-key", "changeme"}


def get_settings() -> Settings:
    return Settings(
        api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
        chat_model=os.getenv("CHAT_MODEL", "gpt-4o-mini"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        knowledge_path=ROOT / "data" / "knowledge.md",
        eval_path=ROOT / "data" / "eval.json",
    )

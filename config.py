from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    groq_api_key: str
    groq_model: str
    chroma_path: Path
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 4


def get_settings() -> Settings:
    return Settings(
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        groq_model=os.getenv("GROQ_MODEL", "qwen/qwen3-32b"),
        chroma_path=Path(os.getenv("CHROMA_PATH", "./chroma_data")),
    )

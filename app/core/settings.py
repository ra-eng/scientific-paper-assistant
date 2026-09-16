from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    gemini_api_key: str
    gemini_model: str = "gemini-2.0-flash"
    embedding_model: str = "all-MiniLM-L6-v2"
    chroma_host: str = "chromadb"
    chroma_port: int = 8000
    database_url: str = "sqlite+aiosqlite:///./data/threads.db"
    log_level: str = "INFO"


settings = Settings()

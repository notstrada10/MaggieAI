"""Configurazione centrale via Pydantic Settings.

Tutte le variabili d'ambiente del sistema sono dichiarate qui.
Vedi `.env.example` per i valori di default e la documentazione.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Postgres ---------------------------------------------------
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "maggieai"
    postgres_user: str = "maggieai"
    postgres_password: str = "maggieai_dev_only_change_me"

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def postgres_dsn_async(self) -> str:
        return (
            f"postgresql+psycopg_async://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # --- Servizi ---------------------------------------------------
    morphology_url: str = "http://localhost:8002"
    inference_local_url: str = "http://host.docker.internal:8001"
    inference_local_model: str = "mlx-community/Qwen2.5-14B-Instruct-4bit"
    gateway_port: int = 8000
    morphology_port: int = 8002
    inference_local_port: int = 8001

    # --- Anthropic / Claude ----------------------------------------
    anthropic_api_key: str = Field(default="", description="Lascia vuoto per disabilitare Claude")
    claude_model: str = "claude-sonnet-4-6"

    # --- Embedding -------------------------------------------------
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024

    # --- Logging / observability -----------------------------------
    log_level: str = "INFO"
    langsmith_tracing: bool = False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton — la cache lru evita di rileggere `.env` a ogni chiamata."""
    return Settings()

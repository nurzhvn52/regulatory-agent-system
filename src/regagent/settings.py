"""Application configuration loaded from environment variables."""

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated runtime settings.

    Secrets are never stored in agent specifications or experiment output.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="REGAGENT_",
        extra="ignore",
    )

    app_name: str = "Regulatory Agent System"
    environment: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"
    database_url: str = (
        "postgresql+psycopg://regagent:regagent@localhost:5432/regagent"
    )
    embedding_model: str = "BAAI/bge-m3"
    llm_provider: str = "mock"
    llm_model: str | None = None
    llm_api_key: SecretStr | None = None
    llm_base_url: str | None = None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return one immutable-by-convention settings object per process."""

    return Settings()


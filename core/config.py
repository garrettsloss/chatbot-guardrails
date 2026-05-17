from __future__ import annotations

from functools import lru_cache
from typing import Any

from pydantic import ConfigDict, Field, validator
from pydantic_settings import BaseSettings

class AppConfig(BaseSettings):
    api_key: str = Field(..., env="API_KEY")
    openai_model: str = Field("gpt-4.1", env="OPENAI_MODEL")
    embedding_model: str = Field("text-embedding-3-large", env="EMBEDDING_MODEL")
    vector_db_url: str = Field(..., env="VECTOR_DB_URL")
    moderation_thresholds: dict[str, float] = Field(default_factory=lambda: {"block": 0.8, "review": 0.5})
    rate_limits: dict[str, int] = Field(default_factory=lambda: {"user_per_minute": 60, "ip_per_minute": 120})
    enabled_modules: list[str] = Field(default_factory=lambda: ["security.auth", "guardrails.input_filter", "guardrails.policy_engine"])
    log_level: str = Field("INFO", env="LOG_LEVEL")
    audit_log_file: str = Field("logs/audit.log", env="AUDIT_LOG_FILE")
    redis_url: str | None = Field(default=None, env="REDIS_URL")
    environment: str = Field("development", env="ENVIRONMENT")
    azure_openai_endpoint: str
    azure_openai_api_key: str
    azure_openai_api_version: str = "2025-03-01-preview"

    model_config = ConfigDict(env_file=".env", extra="forbid")

    @validator("moderation_thresholds")
    def validate_thresholds(cls, value: dict[str, Any]) -> dict[str, Any]:
        if "block" not in value or "review" not in value:
            raise ValueError("moderation_thresholds must define block and review thresholds")
        return value

    @validator("rate_limits")
    def validate_rate_limits(cls, value: dict[str, Any]) -> dict[str, Any]:
        if any(v < 0 for v in value.values()):
            raise ValueError("rate limits must be non-negative")
        return value


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    return AppConfig()


def reload_config() -> AppConfig:
    get_config.cache_clear()
    return get_config()

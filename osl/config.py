"""Application configuration via Pydantic settings.

All settings are read from environment variables prefixed ``OSL_`` (or a local
``.env`` file). Secrets use ``SecretStr`` so they never render in logs or
reprs. ``get_settings()`` is the cached accessor the rest of the app should use.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed configuration for Options Strategy Lab.

    Field names map to ``OSL_<UPPER_NAME>`` environment variables.
    """

    model_config = SettingsConfigDict(
        env_prefix="OSL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Schwab Trader API ---
    schwab_app_key: SecretStr = SecretStr("")
    schwab_app_secret: SecretStr = SecretStr("")
    schwab_callback_url: str = "https://127.0.0.1"
    schwab_token_path: Path = Path(".secrets/schwab_token.json")

    # --- Reference data ---
    fred_api_key: SecretStr | None = None

    # --- Storage roots ---
    data_root: Path = Path("./data")
    snapshot_root: Path = Path("./data/snapshots")

    # --- Risk-free curve ---
    risk_free_curve_source: Literal["fred", "flat"] = "flat"
    risk_free_flat_rate: float = 0.045

    # --- Provider / analytics defaults ---
    default_provider: Literal["schwab", "yfinance"] = "schwab"
    iv_lookback_days: int = Field(default=252, gt=0)

    # Gate v2 / experimental features.
    enable_experimental: bool = False

    @property
    def raw_data_root(self) -> Path:
        """Directory for tee'd raw provider payloads."""
        return self.data_root / "raw"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-wide cached :class:`Settings` instance."""
    return Settings()

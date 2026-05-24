"""Application configuration via Pydantic settings.

All settings are read from environment variables prefixed ``OSL_`` (or a local
``.env`` file). Secrets use ``SecretStr`` so they never render in logs or
reprs. ``get_settings()`` is the cached accessor the rest of the app should use.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


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

    # --- Additional providers (read-only chains) ---
    tradier_token: SecretStr | None = None
    marketdata_token: SecretStr | None = None

    # --- Single-user app lock (sha256 hex of the password; unset = no lock) ---
    app_password_hash: SecretStr | None = None

    # --- Storage roots ---
    data_root: Path = Path("./data")
    snapshot_root: Path = Path("./data/snapshots")

    # --- Risk-free curve ---
    risk_free_curve_source: Literal["fred", "flat"] = "flat"
    risk_free_flat_rate: float = 0.045

    # --- Provider / analytics defaults ---
    default_provider: Literal["schwab", "yfinance"] = "schwab"
    iv_lookback_days: int = Field(default=252, gt=0)

    # Symbols the snapshot worker captures (comma-separated in the env).
    watchlist: Annotated[list[str], NoDecode] = ["SPY", "QQQ", "IWM"]

    # Gate v2 / experimental features.
    enable_experimental: bool = False

    @field_validator("watchlist", mode="before")
    @classmethod
    def _parse_watchlist(cls, value: object) -> object:
        if isinstance(value, str):
            return [s.strip().upper() for s in value.split(",") if s.strip()]
        return value

    @property
    def raw_data_root(self) -> Path:
        """Directory for tee'd raw provider payloads."""
        return self.data_root / "raw"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-wide cached :class:`Settings` instance."""
    return Settings()

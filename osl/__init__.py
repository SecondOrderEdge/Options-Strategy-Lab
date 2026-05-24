"""Options Strategy Lab — a strictly layered options analytics library.

The package is organized so that all business logic lives under ``osl.*`` and
the Streamlit UI is a thin shell on top. M0 ships the data layer: a
provider-agnostic interface, Schwab + yfinance adapters, caching, and a
Parquet snapshot store.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]

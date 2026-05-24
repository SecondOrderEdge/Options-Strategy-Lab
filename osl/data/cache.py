"""Disk-backed memoization for provider calls.

Wraps :mod:`diskcache` with a small typed decorator. The cache key is derived
from the function's qualified name plus its arguments, so distinct arguments
get distinct entries and repeated identical calls within the TTL are served
from disk. Quotes use a short TTL, chains a slightly longer one.
"""

from __future__ import annotations

import functools
import hashlib
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any, ParamSpec, TypeVar, cast

import diskcache

P = ParamSpec("P")
R = TypeVar("R")

# TTL presets (seconds) matching the blueprint's caching policy.
QUOTE_TTL = 15
CHAIN_TTL_OPEN = 60
CHAIN_TTL_CLOSED = 3600


def make_cache(path: str | Path) -> diskcache.Cache:
    """Create (or open) a diskcache at ``path``."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return diskcache.Cache(str(path))


def _key(func_name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    """Stable hash key for a call. ``self``-like unhashables fall back to repr."""
    try:
        payload = pickle.dumps((func_name, args, sorted(kwargs.items())))
    except (pickle.PicklingError, TypeError):
        payload = repr((func_name, args, sorted(kwargs.items()))).encode()
    return hashlib.sha256(payload).hexdigest()


def disk_cached(
    cache: diskcache.Cache,
    *,
    ttl: int,
    key_prefix: str | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator memoizing a function's result in ``cache`` for ``ttl`` seconds.

    The wrapped function gains a ``cache_clear()`` attribute for tests.
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        prefix = key_prefix or func.__qualname__

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            key = _key(prefix, args, kwargs)
            sentinel = object()
            cached = cache.get(key, default=sentinel)
            if cached is not sentinel:
                return cast(R, cached)
            result = func(*args, **kwargs)
            cache.set(key, result, expire=ttl)
            return result

        def cache_clear() -> None:
            cache.clear()

        wrapper.cache_clear = cache_clear  # type: ignore[attr-defined]
        return wrapper

    return decorator

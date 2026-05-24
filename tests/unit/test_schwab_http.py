"""Rate-limit backoff: exponential delay + jitter on HTTP 429."""

from __future__ import annotations

import random
from typing import Any

import pytest

from osl.data.schwab import SchwabAPIError, request_with_backoff


class FakeResponse:
    def __init__(self, status_code: int, body: Any = None) -> None:
        self.status_code = status_code
        self._body = body or {}

    def json(self) -> Any:
        return self._body

    @property
    def text(self) -> str:
        return str(self._body)


def test_backoff_retries_then_succeeds() -> None:
    responses = [FakeResponse(429), FakeResponse(429), FakeResponse(200, {"ok": True})]
    sleeps: list[float] = []

    def send() -> FakeResponse:
        return responses.pop(0)

    resp = request_with_backoff(
        send,
        base_delay=1.0,
        jitter=0.5,
        sleep_fn=sleeps.append,
        rng=random.Random(0),
    )
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
    assert len(sleeps) == 2
    # Exponential: each sleep >= base*2**attempt, jitter <= 50% above.
    assert 1.0 <= sleeps[0] <= 1.5
    assert 2.0 <= sleeps[1] <= 3.0
    assert sleeps[1] > sleeps[0]


def test_backoff_exhausts_and_raises() -> None:
    sleeps: list[float] = []

    def send() -> FakeResponse:
        return FakeResponse(429)

    with pytest.raises(SchwabAPIError) as exc:
        request_with_backoff(
            send,
            max_retries=3,
            base_delay=1.0,
            sleep_fn=sleeps.append,
            rng=random.Random(1),
        )
    assert exc.value.status_code == 429
    assert len(sleeps) == 3  # retried max_retries times before giving up


def test_non_retryable_status_returns_immediately() -> None:
    sleeps: list[float] = []

    def send() -> FakeResponse:
        return FakeResponse(404, {"err": "not found"})

    resp = request_with_backoff(send, sleep_fn=sleeps.append, rng=random.Random(2))
    assert resp.status_code == 404
    assert sleeps == []

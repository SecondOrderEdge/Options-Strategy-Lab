"""Playbook report: HTML build, reproducible digest, provenance footer."""

from __future__ import annotations

from datetime import UTC, datetime

from osl.report.playbook import (
    PlaybookData,
    StrategyRow,
    build_playbook_html,
    git_revision,
    report_digest,
)


def _data(as_of: datetime) -> PlaybookData:
    return PlaybookData(
        symbol="SPY",
        spot=500.0,
        as_of=as_of,
        iv30=0.18,
        iv_rank=0.42,
        iv_percentile=0.55,
        surface_note="SVI fit: no arbitrage detected.",
        strategies=[
            StrategyRow(
                name="Iron Condor",
                objective="ev_per_risk",
                expiry="2026-06-19",
                pop_rn=0.72,
                ev=35.0,
                expected_shortfall=410.0,
                max_loss=-465.0,
                loss_unbounded=False,
                liquidity=0.81,
                breakevens=(480.0, 520.0),
            ),
            StrategyRow(
                name="Short Strangle",
                objective="theta_per_risk",
                expiry="2026-06-19",
                pop_rn=0.80,
                ev=20.0,
                expected_shortfall=1500.0,
                max_loss=float("-inf"),
                loss_unbounded=True,
                liquidity=0.6,
                breakevens=(470.0, 530.0),
            ),
        ],
        assumptions=["RN POP is risk-neutral.", "EV is RN Monte Carlo."],
    )


def test_html_contains_key_content():
    html = build_playbook_html(_data(datetime(2026, 5, 24, 12, tzinfo=UTC)))
    assert "Options Playbook — SPY" in html
    assert "Iron Condor" in html
    assert "∞ (uncovered)" in html  # unbounded loss rendered, never a number
    assert "Research and education only" in html
    assert report_digest(_data(datetime(2026, 5, 24, 12, tzinfo=UTC))) in html


def test_digest_is_reproducible_across_timestamps():
    d1 = _data(datetime(2026, 5, 24, 9, tzinfo=UTC))
    d2 = _data(datetime(2027, 1, 1, 18, tzinfo=UTC))
    assert report_digest(d1) == report_digest(d2)  # timestamp excluded from digest


def test_digest_changes_with_content():
    base = _data(datetime(2026, 5, 24, tzinfo=UTC))
    changed = PlaybookData(**{**base.__dict__, "iv30": 0.25})
    assert report_digest(base) != report_digest(changed)


def test_git_revision_is_a_string():
    rev = git_revision()
    assert isinstance(rev, str) and rev

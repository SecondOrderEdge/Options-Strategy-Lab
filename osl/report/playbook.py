"""Build the per-name Options Playbook as reproducible HTML (+ optional PDF)."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from html import escape

DISCLAIMER = "Research and education only — not investment advice."

# Plain-English definitions for the top-strategies table, used both as hover
# tooltips on the column headers and as a visible glossary (so the standalone
# HTML/PDF report is self-explanatory when shared).
_COLUMN_GLOSSARY: tuple[tuple[str, str], ...] = (
    ("Objective", "The ranking goal this strategy topped (e.g. EV per risk, theta per risk)."),
    ("Strategy", "The option structure (vertical, iron condor, strangle, ...)."),
    ("Expiry", "Expiration date of the nearest leg."),
    (
        "POP (RN)",
        "Risk-neutral probability of finishing profitable - market-implied, not a forecast.",
    ),
    ("EV", "Risk-neutral expected P&L (edge vs fair value), from Monte Carlo."),
    ("ES(95%)", "Expected shortfall: average loss in the worst 5% of outcomes (tail risk)."),
    ("Max loss", "Largest possible loss at expiry; shown as infinity (uncovered) for naked risk."),
    ("Breakevens", "Underlying prices where the position breaks even at expiry."),
    ("Liquidity", "0-1 tradability score from spread, open interest, volume and ATM distance."),
)


@dataclass(frozen=True)
class StrategyRow:
    name: str
    objective: str
    expiry: str
    pop_rn: float
    ev: float
    expected_shortfall: float
    max_loss: float
    loss_unbounded: bool
    liquidity: float
    breakevens: tuple[float, ...]


@dataclass(frozen=True)
class PlaybookData:
    symbol: str
    spot: float
    as_of: datetime
    iv30: float
    iv_rank: float
    iv_percentile: float
    surface_note: str
    strategies: Sequence[StrategyRow]
    assumptions: Sequence[str] = field(default_factory=tuple)


def git_revision() -> str:
    """Short git revision of the working tree, or 'unknown'."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return out.stdout.strip() or "unknown"
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def report_digest(data: PlaybookData) -> str:
    """Content digest over the report's deterministic fields (excludes timestamp).

    Two reports with identical inputs share a digest regardless of when they are
    generated, making a report reproducible and verifiable.
    """
    payload = asdict(data)
    payload.pop("as_of", None)
    payload["spot"] = round(data.spot, 6)
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _fmt_money(x: float) -> str:
    return "∞ (uncovered)" if x == float("-inf") else f"{x:,.2f}"


def _strategy_rows_html(rows: Sequence[StrategyRow]) -> str:
    if not rows:
        return "<tr><td colspan='9'>No candidates.</td></tr>"
    out = []
    for r in rows:
        max_loss = "∞ (uncovered)" if r.loss_unbounded else _fmt_money(r.max_loss)
        bes = ", ".join(f"{b:,.2f}" for b in r.breakevens) or "—"
        out.append(
            "<tr>"
            f"<td>{escape(r.objective)}</td>"
            f"<td>{escape(r.name)}</td>"
            f"<td>{escape(r.expiry)}</td>"
            f"<td>{r.pop_rn:.1%}</td>"
            f"<td>{r.ev:,.0f}</td>"
            f"<td>{r.expected_shortfall:,.0f}</td>"
            f"<td>{max_loss}</td>"
            f"<td>{escape(bes)}</td>"
            f"<td>{r.liquidity:.2f}</td>"
            "</tr>"
        )
    return "\n".join(out)


def build_playbook_html(data: PlaybookData) -> str:
    """Render the playbook as a self-contained HTML document."""
    digest = report_digest(data)
    rev = git_revision()
    assumptions = "".join(f"<li>{escape(a)}</li>" for a in data.assumptions)
    header_cells = "".join(
        f'<th title="{escape(desc)}">{escape(term)}</th>' for term, desc in _COLUMN_GLOSSARY
    )
    glossary = "".join(
        f"<dt>{escape(term)}</dt><dd>{escape(desc)}</dd>" for term, desc in _COLUMN_GLOSSARY
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Options Playbook — {escape(data.symbol)}</title>
<style>
  body {{ font-family: -apple-system, Helvetica, Arial, sans-serif; margin: 2rem; color: #1a1a1a; }}
  h1 {{ margin-bottom: 0; }}
  .sub {{ color: #666; font-size: 0.9rem; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.85rem; }}
  th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: right; }}
  th {{ background: #f4f4f4; }}
  td:nth-child(1), td:nth-child(2), td:nth-child(3) {{ text-align: left; }}
  .metrics span {{ display: inline-block; margin-right: 1.5rem; }}
  footer {{ margin-top: 2rem; color: #888; font-size: 0.75rem; border-top: 1px solid #eee; padding-top: 0.5rem; }}
  .disclaimer {{ color: #b00; font-weight: bold; }}
  .glossary {{ font-size: 0.8rem; color: #444; }}
  .glossary dt {{ font-weight: bold; margin-top: 0.4rem; }}
  .glossary dd {{ margin: 0 0 0 1rem; }}
</style>
</head>
<body>
<h1>Options Playbook — {escape(data.symbol)}</h1>
<div class="sub">Spot {data.spot:,.2f} · generated {data.as_of:%Y-%m-%d %H:%M:%S %Z}</div>

<h2>Volatility</h2>
<div class="metrics">
  <span>IV30: <b>{data.iv30:.1%}</b></span>
  <span>IV rank: <b>{data.iv_rank:.0%}</b></span>
  <span>IV percentile: <b>{data.iv_percentile:.0%}</b></span>
</div>
<p>{escape(data.surface_note)}</p>

<h2>Top strategies</h2>
<table>
  <thead><tr>{header_cells}</tr></thead>
  <tbody>
{_strategy_rows_html(data.strategies)}
  </tbody>
</table>

<h2>How to read this table</h2>
<dl class="glossary">{glossary}</dl>

<h2>Model assumptions</h2>
<ul>{assumptions}</ul>

<p class="disclaimer">{escape(DISCLAIMER)}</p>
<footer>
  digest {digest} · git {escape(rev)} · {data.as_of:%Y-%m-%dT%H:%M:%SZ}
</footer>
</body>
</html>"""


def render_pdf(html: str) -> bytes:
    """Render report HTML to PDF bytes (requires the ``report`` extra: weasyprint)."""
    from weasyprint import HTML  # lazy: heavy system deps

    result: bytes = HTML(string=html).write_pdf()
    return result

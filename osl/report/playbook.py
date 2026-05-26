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
    (
        "Net",
        "Net cash at entry, in dollars: debit (paid) is positive, credit (received) is negative.",
    ),
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

_TICKET_HEADERS: tuple[str, ...] = (
    "Action",
    "Qty",
    "Right",
    "Strike",
    "Expiry",
    "Premium (mid)",
    "IV",
)


@dataclass(frozen=True)
class LegRow:
    """One leg of a strategy, in the form needed to actually place the trade."""

    side: str  # "Buy" or "Sell"
    quantity: int
    right: str  # "C" or "P"
    strike: float
    expiry: str  # ISO date
    premium: float  # per-share mid
    iv: float


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
    # Execution detail. Defaults preserve back-compat with older callers/tests
    # that did not carry leg-level data.
    net_debit: float = 0.0
    legs: tuple[LegRow, ...] = ()


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


def _fmt_net(x: float) -> str:
    """Format net entry cash: positive = debit, negative = credit."""
    if x > 0:
        return f"${x:,.2f} debit"
    if x < 0:
        return f"${abs(x):,.2f} credit"
    return "$0.00"


def _strategy_rows_html(rows: Sequence[StrategyRow]) -> str:
    if not rows:
        return "<tr><td colspan='10'>No candidates.</td></tr>"
    out = []
    for r in rows:
        max_loss = "∞ (uncovered)" if r.loss_unbounded else _fmt_money(r.max_loss)
        bes = ", ".join(f"{b:,.2f}" for b in r.breakevens) or "—"
        out.append(
            "<tr>"
            f"<td>{escape(r.objective)}</td>"
            f"<td>{escape(r.name)}</td>"
            f"<td>{escape(_fmt_net(r.net_debit))}</td>"
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


def _trade_tickets_html(rows: Sequence[StrategyRow]) -> str:
    """Per-strategy execution blocks: the exact legs (side, qty, right, strike, expiry)."""
    if not rows or not any(r.legs for r in rows):
        return ""
    headers = "".join(f"<th>{escape(h)}</th>" for h in _TICKET_HEADERS)
    blocks = []
    for r in rows:
        if not r.legs:
            continue
        leg_rows = "".join(
            "<tr>"
            f"<td>{escape(leg.side)}</td>"
            f"<td>{leg.quantity}</td>"
            f"<td>{escape(leg.right)}</td>"
            f"<td>{leg.strike:,.2f}</td>"
            f"<td>{escape(leg.expiry)}</td>"
            f"<td>{leg.premium:,.2f}</td>"
            f"<td>{leg.iv:.1%}</td>"
            "</tr>"
            for leg in r.legs
        )
        bes = ", ".join(f"{b:,.2f}" for b in r.breakevens) or "—"
        blocks.append(
            '<div class="ticket">'
            f"<h3>{escape(r.objective)}: {escape(r.name)} — {escape(r.expiry)}</h3>"
            f"<table><thead><tr>{headers}</tr></thead><tbody>{leg_rows}</tbody></table>"
            f"<p><b>Net:</b> {escape(_fmt_net(r.net_debit))} &nbsp;·&nbsp; "
            f"<b>Breakevens:</b> {escape(bes)}</p>"
            "</div>"
        )
    return "\n".join(blocks)


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
  /* Explicit white background so the report stays readable when embedded
     inside a dark host page (e.g. the Streamlit app) and when exported to PDF. */
  body {{ font-family: -apple-system, Helvetica, Arial, sans-serif; margin: 2rem; color: #1a1a1a; background: #ffffff; }}
  h1 {{ margin-bottom: 0; }}
  .sub {{ color: #666; font-size: 0.9rem; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.85rem; }}
  th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: right; color: #1a1a1a; background: #ffffff; }}
  th {{ background: #f4f4f4; }}
  /* Objective (1), Strategy (2), Expiry (4) are left-aligned text columns. */
  td:nth-child(1), td:nth-child(2), td:nth-child(4) {{ text-align: left; }}
  .metrics span {{ display: inline-block; margin-right: 1.5rem; }}
  footer {{ margin-top: 2rem; color: #888; font-size: 0.75rem; border-top: 1px solid #eee; padding-top: 0.5rem; }}
  .disclaimer {{ color: #b00; font-weight: bold; }}
  .glossary {{ font-size: 0.8rem; color: #444; }}
  .glossary dt {{ font-weight: bold; margin-top: 0.4rem; }}
  .glossary dd {{ margin: 0 0 0 1rem; }}
  .ticket {{ margin: 1rem 0 1.5rem; padding: 0.5rem 1rem 0.75rem; border: 1px solid #ddd; border-radius: 6px; background: #fafafa; }}
  .ticket h3 {{ margin: 0.25rem 0 0.5rem; font-size: 1rem; }}
  .ticket table {{ font-size: 0.8rem; margin: 0.5rem 0; }}
  /* Inside the ticket, every column is plain text — left-align side/right/expiry
     and right-align the numeric strike/premium/iv columns. */
  .ticket td:nth-child(1), .ticket td:nth-child(3), .ticket td:nth-child(5) {{ text-align: left; }}
  .ticket td:nth-child(2), .ticket td:nth-child(4), .ticket td:nth-child(6), .ticket td:nth-child(7) {{ text-align: right; }}
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

<h2>Trade tickets</h2>
<p class="sub">Exact legs for each top strategy — side, quantity, right (C/P), strike and expiry — so the trade can be entered directly.</p>
{_trade_tickets_html(data.strategies)}

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

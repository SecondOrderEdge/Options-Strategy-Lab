"""Per-name "Options Playbook" report generation.

Builds a self-contained, reproducible HTML report (and, with the ``report``
extra installed, a PDF). Each report carries a provenance footer: the git
revision, a UTC timestamp, and a content digest that is stable across runs for
identical inputs — so a report can be reproduced and verified.
"""

from __future__ import annotations

from osl.report.playbook import (
    PlaybookData,
    StrategyRow,
    build_playbook_html,
    git_revision,
    render_pdf,
    report_digest,
)

__all__ = [
    "PlaybookData",
    "StrategyRow",
    "build_playbook_html",
    "git_revision",
    "render_pdf",
    "report_digest",
]

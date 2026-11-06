"""
Report generation package.

Produces rich, self-contained, interactive HTML reports from check results,
with per-fault visualizations embedded as base64 images.

Features:
  - Failure tracebacks with root-cause analysis and remediation steps
  - Drilldown tables for null/zero/NaN vectors, duplicates, missing metadata
  - Help text tooltips on every check and chart
  - Interactive status filters (All / Pass / Fail / Info) and live search
  - Database-agnostic design — works with any vector DB connector

Usage::

    from report.html_report import generate_report

    html = generate_report(
        check_results=results,
        db_label="Chroma / my_collection",
    )
"""

from report.html_report import generate_report

__all__ = ["generate_report"]

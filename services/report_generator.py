"""
Report Generator — produces a comprehensive, self-contained HTML report
for a RAG test execution.

Usage::

    from services.report_generator import ReportGenerator

    gen = ReportGenerator()
    html = gen.generate(execution_data, project_settings)
"""

from typing import Any, Dict, Optional
import html as html_mod


class ReportGenerator:
    """Generates a professional HTML report from execution data."""

    BLUE = "#0033A1"
    BLUE_DARK = "#002878"
    BLUE_LIGHT = "#E8EEF7"

    def _esc(self, text: str) -> str:
        return html_mod.escape(str(text))

    def _truncate(self, text: str, length: int = 80) -> str:
        if len(text) <= length:
            return text
        return text[:length] + "..."

    def generate(
        self,
        execution_data: Dict[str, Any],
        project_settings: Optional[Dict[str, Any]] = None,
    ) -> str:
        if project_settings is None:
            project_settings = {}

        summary = execution_data.get("summary", {})
        results = execution_data.get("results", [])
        vector_checks = execution_data.get("vector_checks", {})
        execution_id = execution_data.get("execution_id", "N/A")
        executed_at = execution_data.get("executed_at", "N/A")
        duration = execution_data.get("duration_seconds", 0)

        total_q = summary.get("total_questions", 0)
        correct = summary.get("correct_answers", 0)
        accuracy = summary.get("accuracy", 0)
        avg_conf = summary.get("avg_confidence", 0)
        avg_ctx = summary.get("avg_context_relevance", 0)
        avg_rt = summary.get("avg_response_time_ms", 0)
        aut_endpoint = summary.get("aut_endpoint", "N/A")
        project_name = project_settings.get("name", "RAG Testing Project")

        styles = self._styles()
        header = self._header(project_name, execution_id, executed_at, duration)
        exec_summary = self._exec_summary(total_q, correct, accuracy, avg_conf, avg_ctx, avg_rt, aut_endpoint)
        dashboard = self._dashboard(accuracy, avg_conf, avg_ctx)
        table = self._results_table(results)
        vector = self._vector_section(vector_checks)
        trace = self._traceability(results)
        failures = self._failures(results)
        footer = self._footer(executed_at)
        scripts = self._scripts()

        return (
            "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
            "<meta charset=\"UTF-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
            f"<title>RAG Testing Report - {self._esc(project_name)}</title>\n"
            f"{styles}\n</head>\n<body>\n"
            f"{header}\n<div class=\"container\">\n"
            f"{exec_summary}\n{dashboard}\n{table}\n{vector}\n{trace}\n{failures}\n"
            f"</div>\n{footer}\n{scripts}\n</body>\n</html>"
        )

    def _styles(self) -> str:
        return (
            "<style>\n"
            "* { margin:0; padding:0; box-sizing:border-box; }\n"
            "body { font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; background:#f8fafc; color:#1e293b; line-height:1.6; }\n"
            ".container { max-width:1200px; margin:0 auto; padding:24px; }\n"
            f".header {{ background:{self.BLUE}; color:white; padding:24px 0; }}\n"
            ".header-inner { max-width:1200px; margin:0 auto; padding:0 24px; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:16px; }\n"
            ".header-title { font-size:24px; font-weight:700; }\n"
            ".header-subtitle { font-size:14px; opacity:0.85; margin-top:4px; }\n"
            ".header-meta { display:flex; flex-direction:column; gap:4px; font-size:12px; opacity:0.9; text-align:right; }\n"
            f".section {{ background:white; border-radius:12px; border:1px solid #e2e8f0; padding:24px; margin-bottom:24px; }}\n"
            f".section-title {{ font-size:18px; font-weight:700; color:{self.BLUE}; margin-bottom:16px; padding-bottom:8px; border-bottom:2px solid {self.BLUE_LIGHT}; }}\n"
            ".section-desc { font-size:13px; color:#64748b; margin-bottom:16px; }\n"
            ".summary-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:16px; margin-bottom:16px; }\n"
            ".summary-card { background:#f8fafc; border-radius:8px; padding:20px; text-align:center; border:1px solid #e2e8f0; }\n"
            f".summary-card.accent {{ background:{self.BLUE_LIGHT}; border-color:{self.BLUE}; }}\n"
            f".big-number {{ font-size:36px; font-weight:800; color:{self.BLUE}; }}\n"
            ".big-label { font-size:13px; color:#64748b; margin-top:4px; }\n"
            ".metric-value { font-size:28px; font-weight:700; color:#1e293b; }\n"
            ".metric-label { font-size:12px; color:#64748b; margin-top:4px; }\n"
            ".endpoint-info { font-size:12px; color:#64748b; }\n"
            ".endpoint-info code { background:#f1f5f9; padding:2px 6px; border-radius:4px; font-size:11px; }\n"
            ".status-badge { display:inline-block; padding:4px 12px; border-radius:20px; font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:0.5px; }\n"
            ".status-badge.pass { background:#dcfce7; color:#166534; }\n"
            ".status-badge.warn { background:#fef3c7; color:#92400e; }\n"
            ".status-badge.fail { background:#fee2e2; color:#991b1b; }\n"
            ".dashboard { display:flex; gap:32px; align-items:center; flex-wrap:wrap; }\n"
            ".gauge-container { flex:0 0 160px; }\n"
            ".gauge-svg { width:160px; height:160px; }\n"
            ".gauge-text { font-size:22px; font-weight:800; fill:#1e293b; }\n"
            ".gauge-sub { font-size:10px; fill:#64748b; }\n"
            ".bars-container { flex:1; min-width:300px; }\n"
            ".bar-row { display:flex; align-items:center; gap:12px; margin-bottom:12px; }\n"
            ".bar-label { width:140px; font-size:13px; color:#475569; font-weight:500; }\n"
            ".bar-track { flex:1; height:20px; background:#e2e8f0; border-radius:10px; overflow:hidden; }\n"
            ".bar-fill { height:100%; border-radius:10px; }\n"
            ".bar-value { width:50px; text-align:right; font-size:13px; font-weight:600; color:#1e293b; }\n"
            ".table-wrapper { overflow-x:auto; }\n"
            f".results-table {{ width:100%; border-collapse:collapse; font-size:13px; }}\n"
            f".results-table th {{ background:{self.BLUE}; color:white; padding:10px 12px; text-align:left; font-weight:600; cursor:pointer; white-space:nowrap; user-select:none; }}\n"
            f".results-table th:hover {{ background:{self.BLUE_DARK}; }}\n"
            ".results-table td { padding:8px 12px; border-bottom:1px solid #e2e8f0; vertical-align:top; }\n"
            ".results-table .center { text-align:center; }\n"
            ".expandable-row { cursor:pointer; }\n"
            ".expandable-row:hover { background:#f1f5f9; }\n"
            ".row-fail { background:#fef2f2; }\n"
            ".row-fail:hover { background:#fee2e2; }\n"
            ".icon-pass { color:#22c55e; font-weight:700; font-size:16px; }\n"
            ".icon-fail { color:#ef4444; font-weight:700; font-size:16px; }\n"
            ".q-cell { max-width:200px; }\n"
            ".detail-row td { padding:0 !important; }\n"
            f".detail-panel {{ background:#f8fafc; padding:20px; border-left:4px solid {self.BLUE}; }}\n"
            ".detail-section { margin-bottom:16px; }\n"
            f".detail-section h4 {{ font-size:13px; font-weight:700; color:{self.BLUE}; margin-bottom:6px; text-transform:uppercase; letter-spacing:0.5px; }}\n"
            ".detail-section p { font-size:13px; color:#334155; }\n"
            ".ctx-item { background:white; padding:8px 12px; border-radius:6px; margin-bottom:6px; font-size:12px; border:1px solid #e2e8f0; }\n"
            ".meta-table { width:100%; border-collapse:collapse; font-size:12px; }\n"
            ".meta-table th { background:#e2e8f0; padding:6px 10px; text-align:left; font-weight:600; }\n"
            ".meta-table td { padding:6px 10px; border-bottom:1px solid #e2e8f0; }\n"
            f".meta-key {{ font-weight:600; color:{self.BLUE}; white-space:nowrap; }}\n"
            ".vs-summary { display:flex; gap:24px; margin-bottom:16px; font-size:14px; }\n"
            ".pass-text { color:#22c55e; }\n"
            ".fail-text { color:#ef4444; }\n"
            ".checks-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:16px; }\n"
            ".check-card { border-radius:8px; padding:16px; border:1px solid #e2e8f0; }\n"
            ".check-pass { background:#f0fdf4; border-color:#bbf7d0; }\n"
            ".check-fail { background:#fef2f2; border-color:#fecaca; }\n"
            ".check-header { display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; }\n"
            ".check-name { font-weight:700; font-size:14px; }\n"
            ".check-detail-item { font-size:12px; color:#475569; margin-bottom:4px; }\n"
            ".check-key { font-weight:600; color:#1e293b; }\n"
            ".trace-table td { font-size:11px; white-space:nowrap; }\n"
            ".failure-table tr { background:#fef2f2; }\n"
            ".failure-table tr:hover { background:#fee2e2; }\n"
            ".muted { color:#94a3b8; font-size:13px; }\n"
            ".success-msg { color:#22c55e; font-size:14px; font-weight:600; }\n"
            f".footer {{ text-align:center; padding:24px; color:#94a3b8; font-size:12px; border-top:1px solid #e2e8f0; margin-top:24px; }}\n"
            f".footer strong {{ color:{self.BLUE}; }}\n"
            "@media print { body { background:white; } .header { -webkit-print-color-adjust:exact; print-color-adjust:exact; } }\n"
            "</style>"
        )

    def _header(self, project_name: str, execution_id: str, executed_at: str, duration: float) -> str:
        ts = executed_at[:19] if len(executed_at) > 19 else executed_at
        return (
            '<header class="header"><div class="header-inner"><div>'
            f'<h1 class="header-title">RAG Testing Report</h1>'
            f'<p class="header-subtitle">{self._esc(project_name)}</p>'
            '</div><div class="header-meta">'
            f'<span>Execution: <strong>{self._esc(execution_id)}</strong></span>'
            f'<span>Date: <strong>{self._esc(ts)}</strong></span>'
            f'<span>Duration: <strong>{duration}s</strong></span>'
            '</div></div></header>'
        )

    def _exec_summary(self, total_q: int, correct: int, accuracy: float,
                      avg_conf: float, avg_ctx: float, avg_rt: float, aut_endpoint: str) -> str:
        pct = round(accuracy * 100, 1)
        sc = "pass" if pct >= 70 else ("warn" if pct >= 50 else "fail")
        sl = "PASS" if pct >= 70 else ("MARGINAL" if pct >= 50 else "FAIL")
        return (
            '<section class="section">'
            '<h2 class="section-title">Executive Summary</h2>'
            '<div class="summary-grid">'
            '<div class="summary-card accent">'
            f'<div class="big-number">{pct}%</div>'
            '<div class="big-label">Overall Accuracy</div>'
            f'<div class="status-badge {sc}">{sl}</div></div>'
            f'<div class="summary-card"><div class="metric-value">{total_q}</div>'
            '<div class="metric-label">Total Questions</div></div>'
            f'<div class="summary-card"><div class="metric-value">{correct}</div>'
            '<div class="metric-label">Correct Answers</div></div>'
            f'<div class="summary-card"><div class="metric-value">{round(avg_conf * 100, 1)}%</div>'
            '<div class="metric-label">Avg Confidence</div></div>'
            f'<div class="summary-card"><div class="metric-value">{round(avg_ctx * 100, 1)}%</div>'
            '<div class="metric-label">Avg Context Relevance</div></div>'
            f'<div class="summary-card"><div class="metric-value">{round(avg_rt, 0)}ms</div>'
            '<div class="metric-label">Avg Response Time</div></div>'
            '</div>'
            f'<p class="endpoint-info">AUT Endpoint: <code>{self._esc(aut_endpoint)}</code></p>'
            '</section>'
        )

    def _dashboard(self, accuracy: float, avg_conf: float, avg_ctx: float) -> str:
        acc_color = "#22c55e" if accuracy >= 0.7 else ("#f59e0b" if accuracy >= 0.5 else "#ef4444")
        dash = accuracy * 314.16

        def bar(label: str, value: float, color: str) -> str:
            p = round(value * 100, 1)
            return (
                f'<div class="bar-row"><span class="bar-label">{label}</span>'
                f'<div class="bar-track"><div class="bar-fill" style="width:{p}%;background:{color}"></div></div>'
                f'<span class="bar-value">{p}%</span></div>'
            )

        return (
            '<section class="section">'
            '<h2 class="section-title">Metrics Dashboard</h2>'
            '<div class="dashboard">'
            '<div class="gauge-container">'
            '<svg viewBox="0 0 120 120" class="gauge-svg">'
            '<circle cx="60" cy="60" r="50" fill="none" stroke="#e2e8f0" stroke-width="10"/>'
            f'<circle cx="60" cy="60" r="50" fill="none" stroke="{acc_color}" stroke-width="10" '
            f'stroke-dasharray="{dash} 314.16" stroke-linecap="round" transform="rotate(-90 60 60)"/>'
            f'<text x="60" y="55" text-anchor="middle" class="gauge-text">{round(accuracy * 100, 1)}%</text>'
            '<text x="60" y="72" text-anchor="middle" class="gauge-sub">Accuracy</text>'
            '</svg></div>'
            '<div class="bars-container">'
            + bar("Accuracy", accuracy, acc_color)
            + bar("Confidence", avg_conf, self.BLUE)
            + bar("Context Relevance", avg_ctx, "#8b5cf6")
            + '</div></div></section>'
        )

    def _results_table(self, results: list) -> str:
        rows = []
        for i, r in enumerate(results, 1):
            ic = r.get("is_correct", False)
            rc = "row-pass" if ic else "row-fail"
            icon = '<span class="icon-pass">&#10003;</span>' if ic else '<span class="icon-fail">&#10007;</span>'
            q = self._esc(self._truncate(r.get("question", ""), 80))
            exp = self._esc(self._truncate(r.get("expected_answer", ""), 80))
            aut = self._esc(self._truncate(r.get("aut_answer", ""), 80))
            conf = round(r.get("aut_confidence", 0) * 100, 1)
            ovlp = round(r.get("keyword_overlap", 0) * 100, 1)
            crel = round(r.get("context_relevance", 0) * 100, 1)
            rt = round(r.get("response_time_ms", 0), 0)

            # Detail panel
            fq = self._esc(r.get("question", ""))
            fe = self._esc(r.get("expected_answer", ""))
            fa = self._esc(r.get("aut_answer", ""))
            meta = r.get("metadata", {})
            mr = "".join(
                f'<tr><td class="meta-key">{self._esc(str(k))}</td><td>{self._esc(str(v))}</td></tr>'
                for k, v in meta.items()
            ) if meta else '<tr><td colspan="2" class="muted">No metadata</td></tr>'

            ctxs = r.get("aut_context", [])
            ci = ""
            for idx, ctx in enumerate(ctxs):
                t = ctx.get("text", "") if isinstance(ctx, dict) else str(ctx)
                ci += f'<div class="ctx-item"><strong>Context {idx + 1}:</strong> {self._esc(self._truncate(t, 200))}</div>'
            if not ci:
                ci = '<p class="muted">No context retrieved</p>'

            srcs = r.get("aut_sources", [])
            st = ", ".join(self._esc(str(s)) for s in srcs) if srcs else "None"

            rows.append(
                f'<tr class="{rc} expandable-row" onclick="toggleDetail(\'d{i}\')">'
                f'<td>{i}</td><td class="q-cell">{q}</td><td>{exp}</td><td>{aut}</td>'
                f'<td class="center">{icon}</td><td class="center">{conf}%</td>'
                f'<td class="center">{ovlp}%</td><td class="center">{crel}%</td>'
                f'<td class="center">{rt}ms</td></tr>'
                f'<tr class="detail-row" id="d{i}" style="display:none"><td colspan="9">'
                f'<div class="detail-panel">'
                f'<div class="detail-section"><h4>Full Question</h4><p>{fq}</p></div>'
                f'<div class="detail-section"><h4>Expected Answer</h4><p>{fe}</p></div>'
                f'<div class="detail-section"><h4>AUT Answer</h4><p>{fa}</p></div>'
                f'<div class="detail-section"><h4>Retrieved Context</h4>{ci}</div>'
                f'<div class="detail-section"><h4>Sources</h4><p>{st}</p></div>'
                f'<div class="detail-section"><h4>Metadata Traceability</h4>'
                f'<table class="meta-table"><thead><tr><th>Field</th><th>Value</th></tr></thead>'
                f'<tbody>{mr}</tbody></table></div>'
                f'</div></td></tr>'
            )

        return (
            '<section class="section">'
            '<h2 class="section-title">Detailed Results</h2>'
            '<p class="section-desc">Click any row to expand and see full details with metadata traceability.</p>'
            '<div class="table-wrapper">'
            '<table class="results-table" id="resultsTable"><thead><tr>'
            '<th onclick="sortTable(0)">#</th>'
            '<th onclick="sortTable(1)">Question</th>'
            '<th onclick="sortTable(2)">Expected Answer</th>'
            '<th onclick="sortTable(3)">AUT Answer</th>'
            '<th onclick="sortTable(4)" class="center">Correct?</th>'
            '<th onclick="sortTable(5)" class="center">Confidence</th>'
            '<th onclick="sortTable(6)" class="center">Overlap</th>'
            '<th onclick="sortTable(7)" class="center">Context Rel.</th>'
            '<th onclick="sortTable(8)" class="center">Response Time</th>'
            '</tr></thead><tbody>'
            + "".join(rows)
            + '</tbody></table></div></section>'
        )

    def _vector_section(self, vector_checks: Dict[str, Any]) -> str:
        if not vector_checks or vector_checks.get("status") in ("skipped", "error", None):
            msg = vector_checks.get("message", "Vector store checks were not run.") if vector_checks else "Vector store checks were not run."
            return (
                '<section class="section">'
                '<h2 class="section-title">Vector Store Health</h2>'
                f'<p class="muted">{self._esc(msg)}</p></section>'
            )

        checks = vector_checks.get("checks", [])
        vs = vector_checks.get("summary", {})
        p = vs.get("passed", 0)
        f = vs.get("failed", 0)
        t = vs.get("total_checks", len(checks))

        cards = []
        for chk in checks:
            name = chk.get("check", "Unknown")
            ip = chk.get("passed", False)
            detail = chk.get("detail", {})
            badge = '<span class="status-badge pass">PASS</span>' if ip else '<span class="status-badge fail">FAIL</span>'
            di = "".join(
                f'<div class="check-detail-item"><span class="check-key">{self._esc(str(k))}:</span> {self._esc(str(v))}</div>'
                for k, v in detail.items()
            )
            cc = "check-pass" if ip else "check-fail"
            cards.append(
                f'<div class="check-card {cc}">'
                f'<div class="check-header"><span class="check-name">{self._esc(name.title())}</span>{badge}</div>'
                f'<div class="check-details">{di}</div></div>'
            )

        return (
            '<section class="section">'
            '<h2 class="section-title">Vector Store Health</h2>'
            '<div class="vs-summary">'
            f'<span><strong>{t}</strong> checks run</span>'
            f'<span class="pass-text"><strong>{p}</strong> passed</span>'
            f'<span class="fail-text"><strong>{f}</strong> failed</span>'
            '</div><div class="checks-grid">'
            + "".join(cards)
            + '</div></section>'
        )

    def _traceability(self, results: list) -> str:
        all_fields: set = set()
        for r in results:
            m = r.get("metadata", {})
            if isinstance(m, dict):
                all_fields.update(m.keys())

        if not all_fields:
            return (
                '<section class="section">'
                '<h2 class="section-title">Metadata Traceability</h2>'
                '<p class="muted">No metadata fields found in test results.</p></section>'
            )

        sf = sorted(all_fields)
        hc = "".join(f"<th>{self._esc(fld)}</th>" for fld in sf)
        rows = []
        for r in results:
            pid = self._esc(r.get("pair_id", ""))
            m = r.get("metadata", {})
            cells = "".join(f"<td>{self._esc(str(m.get(fld, '')))}</td>" for fld in sf)
            rows.append(f"<tr><td>{pid}</td>{cells}</tr>")

        return (
            '<section class="section">'
            '<h2 class="section-title">Metadata Traceability</h2>'
            '<p class="section-desc">Cross-reference between test pairs and source record metadata.</p>'
            '<div class="table-wrapper">'
            '<table class="results-table trace-table"><thead><tr><th>Pair ID</th>'
            + hc + '</tr></thead><tbody>'
            + "".join(rows)
            + '</tbody></table></div></section>'
        )

    def _failures(self, results: list) -> str:
        fails = [r for r in results if not r.get("is_correct", False)]
        if not fails:
            return (
                '<section class="section">'
                '<h2 class="section-title">Failure Analysis</h2>'
                '<p class="success-msg">All answers were correct! No failures to analyze.</p></section>'
            )

        rows = []
        for i, f in enumerate(fails, 1):
            q = self._esc(self._truncate(f.get("question", ""), 100))
            exp = self._esc(self._truncate(f.get("expected_answer", ""), 120))
            act = self._esc(self._truncate(f.get("aut_answer", ""), 120))
            ovlp = round(f.get("keyword_overlap", 0) * 100, 1)

            aa = f.get("aut_answer", "")
            if "ERROR" in aa:
                reason = "AUT returned an error"
            elif ovlp < 10:
                reason = "Completely different answer - possible retrieval failure"
            elif ovlp < 30:
                reason = "Low keyword overlap - answer may be partially relevant"
            elif f.get("aut_confidence", 0) < 0.3:
                reason = "Low confidence - AUT uncertain about the answer"
            else:
                reason = "Answer partially matches but below threshold"

            rows.append(
                f'<tr><td>{i}</td><td>{q}</td><td>{exp}</td><td>{act}</td>'
                f'<td class="center">{ovlp}%</td><td>{self._esc(reason)}</td></tr>'
            )

        return (
            '<section class="section">'
            '<h2 class="section-title">Failure Analysis</h2>'
            f'<p class="section-desc">{len(fails)} out of {len(results)} questions answered incorrectly.</p>'
            '<div class="table-wrapper">'
            '<table class="results-table failure-table"><thead><tr>'
            '<th>#</th><th>Question</th><th>Expected Answer</th><th>AUT Answer</th>'
            '<th class="center">Overlap</th><th>Possible Reason</th>'
            '</tr></thead><tbody>'
            + "".join(rows)
            + '</tbody></table></div></section>'
        )

    def _footer(self, executed_at: str) -> str:
        return (
            '<footer class="footer">'
            '<p>Generated by <strong>RAG Testing Platform</strong> &mdash; Cognizant</p>'
            f'<p>Report generated at {self._esc(executed_at)}</p>'
            '</footer>'
        )

    def _scripts(self) -> str:
        return (
            "<script>\n"
            "function toggleDetail(id) {\n"
            "  var el = document.getElementById(id);\n"
            "  if (el) el.style.display = el.style.display === 'none' ? 'table-row' : 'none';\n"
            "}\n"
            "function sortTable(col) {\n"
            "  var table = document.getElementById('resultsTable');\n"
            "  if (!table) return;\n"
            "  var tbody = table.tBodies[0];\n"
            "  var rows = Array.from(tbody.querySelectorAll('tr.expandable-row'));\n"
            "  var asc = table.getAttribute('data-sort-col') !== String(col) || table.getAttribute('data-sort-dir') !== 'asc';\n"
            "  table.setAttribute('data-sort-col', col);\n"
            "  table.setAttribute('data-sort-dir', asc ? 'asc' : 'desc');\n"
            "  rows.sort(function(a, b) {\n"
            "    var at = a.cells[col] ? a.cells[col].textContent.trim() : '';\n"
            "    var bt = b.cells[col] ? b.cells[col].textContent.trim() : '';\n"
            "    var an = parseFloat(at), bn = parseFloat(bt);\n"
            "    if (!isNaN(an) && !isNaN(bn)) return asc ? an - bn : bn - an;\n"
            "    return asc ? at.localeCompare(bt) : bt.localeCompare(at);\n"
            "  });\n"
            "  rows.forEach(function(row) {\n"
            "    var detail = row.nextElementSibling;\n"
            "    tbody.appendChild(row);\n"
            "    if (detail && detail.classList.contains('detail-row')) tbody.appendChild(detail);\n"
            "  });\n"
            "}\n"
            "</script>"
        )

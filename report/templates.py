"""
HTML/CSS/JS templates for the vector database health report.

All templates use simple Python string .format() — no Jinja2 dependency.
The report is fully self-contained: all CSS/JS is inline, all images are
base64-encoded, no external files or CDN links required.

Key design principles:
  - Database-agnostic: no Chroma/Pinecone/Weaviate branding — works for ANY
    vector database by accepting a generic ``db_label`` parameter.
  - Failure tracebacks: every failed check shows a structured traceback with
    root cause, affected IDs, and remediation steps so developers can fix issues.
  - Drilldown tables: null/zero/NaN vectors, duplicates, missing metadata etc.
    are rendered as sortable, searchable tables — not just counts.
  - Help text: every chart and section has a ``ⓘ`` tooltip explaining what
    the visualization shows and how to interpret it.
  - Interactive filters: JavaScript-powered status filter (All / Pass / Fail /
    Info) and a live search box — no server needed.
"""

# ─── Color palette ───────────────────────────────────────────────────
COLORS = {
    "pass": "#22c55e",       # green
    "fail": "#ef4444",       # red
    "info": "#3b82f6",       # blue
    "warn": "#f59e0b",       # amber
    "bg_dark": "#1e293b",    # slate-800
    "bg_card": "#ffffff",
    "bg_page": "#f1f5f9",    # slate-100
    "text": "#334155",       # slate-700
    "text_light": "#94a3b8", # slate-400
    "border": "#e2e8f0",     # slate-200
}

# ─── Main report skeleton ────────────────────────────────────────────
REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Vector Database Health Report</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: {bg_page};
    color: {text};
    line-height: 1.6;
  }}

  /* ── Header ─────────────────────────────────────────────── */
  .header {{
    background: {bg_dark};
    color: white;
    padding: 2rem 3rem;
  }}
  .header h1 {{
    font-size: 1.8rem;
    font-weight: 700;
    letter-spacing: -0.02em;
  }}
  .header .subtitle {{
    color: {text_light};
    font-size: 0.9rem;
    margin-top: 0.3rem;
  }}
  .header .db-badge {{
    display: inline-block;
    margin-top: 0.5rem;
    padding: 0.25rem 0.8rem;
    border-radius: 20px;
    background: rgba(255,255,255,0.12);
    color: #e2e8f0;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.03em;
  }}

  .container {{
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem;
    min-width: 0;
    overflow-x: hidden;
  }}

  /* ── Filter toolbar ─────────────────────────────────────── */
  .toolbar {{
    display: flex;
    gap: 1rem;
    margin-bottom: 1.5rem;
    flex-wrap: wrap;
    align-items: center;
  }}
  .filter-btn {{
    padding: 0.4rem 1rem;
    border-radius: 20px;
    border: 2px solid {border};
    background: {bg_card};
    cursor: pointer;
    font-size: 0.82rem;
    font-weight: 600;
    transition: all 0.15s;
    color: {text};
  }}
  .filter-btn:hover {{ border-color: {info_color}; color: {info_color}; }}
  .filter-btn.active {{ background: {bg_dark}; color: white; border-color: {bg_dark}; }}
  .filter-btn .count {{
    display: inline-block;
    min-width: 1.3rem;
    text-align: center;
    padding: 0 0.3rem;
    margin-left: 0.3rem;
    border-radius: 10px;
    font-size: 0.72rem;
    font-weight: 800;
    line-height: 1.4;
  }}
  .filter-btn.active .count {{ background: rgba(255,255,255,0.2); }}
  .filter-btn:not(.active) .count {{ background: {bg_page}; }}

  .search-box {{
    flex: 1;
    min-width: 200px;
    padding: 0.45rem 1rem;
    border-radius: 20px;
    border: 2px solid {border};
    font-size: 0.85rem;
    outline: none;
    transition: border-color 0.15s;
  }}
  .search-box:focus {{ border-color: {info_color}; }}

  /* ── Summary dashboard ──────────────────────────────────── */
  .summary {{
    display: flex;
    gap: 1.5rem;
    margin-bottom: 2rem;
    flex-wrap: wrap;
  }}
  .summary-card {{
    flex: 1;
    min-width: 140px;
    background: {bg_card};
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    border-top: 4px solid {border};
  }}
  .summary-card.pass {{ border-top-color: {pass_color}; }}
  .summary-card.fail {{ border-top-color: {fail_color}; }}
  .summary-card.info {{ border-top-color: {info_color}; }}
  .summary-card .number {{
    font-size: 2.5rem;
    font-weight: 800;
    line-height: 1;
  }}
  .summary-card.pass .number {{ color: {pass_color}; }}
  .summary-card.fail .number {{ color: {fail_color}; }}
  .summary-card.info .number {{ color: {info_color}; }}
  .summary-card .label {{
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: {text_light};
    margin-top: 0.5rem;
  }}

  /* ── Category sections ──────────────────────────────────── */
  .category {{
    margin-bottom: 2rem;
  }}
  .category-title {{
    font-size: 1.1rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    color: {bg_dark};
    padding: 0.8rem 0;
    border-bottom: 2px solid {bg_dark};
    margin-bottom: 1rem;
  }}

  /* ── Check cards ────────────────────────────────────────── */
  .check-card {{
    background: {bg_card};
    border-radius: 10px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    border-left: 4px solid {border};
    transition: box-shadow 0.15s;
  }}
  .check-card:hover {{ box-shadow: 0 4px 12px rgba(0,0,0,0.1); }}
  .check-card.pass {{ border-left-color: {pass_color}; }}
  .check-card.fail {{ border-left-color: {fail_color}; }}
  .check-card.info {{ border-left-color: {info_color}; }}
  .check-header {{
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin-bottom: 1rem;
  }}
  .badge {{
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }}
  .badge.pass {{ background: #dcfce7; color: #166534; }}
  .badge.fail {{ background: #fee2e2; color: #991b1b; }}
  .badge.info {{ background: #dbeafe; color: #1e40af; }}
  .check-name {{
    font-size: 1rem;
    font-weight: 600;
  }}
  .neighbor-pair {{
    border: 1px solid {border};
    border-radius: 8px;
    background: {bg_card};
    padding: 0.9rem 1rem;
    margin: 0.7rem 0;
    min-width: 0;
  }}
  .neighbor-pair-id {{
    font-family: 'SF Mono', 'Fira Code', monospace;
    font-size: 0.78rem;
    overflow-wrap: anywhere;
    word-break: break-word;
  }}
  .neighbor-ranks {{
    color: {text_light};
    font-size: 0.82rem;
    margin: 0.35rem 0 0.65rem;
    overflow-wrap: anywhere;
  }}
  .neighbor-metadata {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(min(100%, 220px), 1fr));
    gap: 0.6rem;
  }}
  .neighbor-metadata-side {{
    min-width: 0;
    border-radius: 6px;
    background: {bg_page};
    padding: 0.65rem 0.75rem;
    font-size: 0.78rem;
    overflow-wrap: anywhere;
    word-break: break-word;
  }}
  .neighbor-metadata-side strong {{
    display: block;
    margin-bottom: 0.25rem;
  }}
  .neighbor-metadata-side span {{
    display: block;
    margin-top: 0.15rem;
  }}
  .neighbor-chunk-text {{
    margin-top: 0.6rem;
    max-width: 100%;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    word-break: break-word;
    font-size: 0.78rem;
  }}
  .duplicate-candidates {{
    margin: 0.8rem 0;
  }}
  .duplicate-candidates-toggle {{
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.5rem 0.9rem;
    border: 1px solid {border};
    border-radius: 6px;
    background: {bg_page};
    color: {info_color};
    cursor: pointer;
    font-size: 0.85rem;
    font-weight: 700;
    list-style: none;
  }}
  .duplicate-candidates-toggle::-webkit-details-marker {{ display: none; }}
  .duplicate-candidates-toggle::before {{
    content: '▸';
    transition: transform 0.15s;
  }}
  .duplicate-candidates[open] > .duplicate-candidates-toggle::before {{
    transform: rotate(90deg);
  }}
  .duplicate-candidates-content {{
    padding-top: 0.4rem;
  }}
  .duplicate-pair {{
    border: 1px solid {border};
    border-radius: 8px;
    background: {bg_card};
    margin: 0.5rem 0;
    min-width: 0;
  }}
  .duplicate-pair-toggle {{
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.6rem;
    padding: 0.75rem 0.9rem;
    cursor: pointer;
    list-style: none;
  }}
  .duplicate-pair-toggle::-webkit-details-marker {{ display: none; }}
  .duplicate-pair-toggle::before {{
    content: '▸';
    color: {info_color};
    transition: transform 0.15s;
  }}
  .duplicate-pair[open] > .duplicate-pair-toggle::before {{
    transform: rotate(90deg);
  }}
  .duplicate-pair-ids {{
    flex: 1 1 300px;
    min-width: 0;
    font-family: 'SF Mono', 'Fira Code', monospace;
    font-size: 0.82rem;
    overflow-wrap: anywhere;
  }}
  .duplicate-pair-content {{
    padding: 0 0.9rem 0.9rem;
  }}
  .duplicate-similarity {{
    flex-shrink: 0;
    padding: 0.25rem 0.6rem;
    border-radius: 999px;
    background: #dbeafe;
    color: #1e40af;
    font-size: 0.78rem;
    font-weight: 700;
  }}
  .duplicate-evidence {{
    margin-top: 0.7rem;
    border-top: 1px solid {border};
    padding-top: 0.6rem;
  }}
  .duplicate-evidence summary {{
    cursor: pointer;
    color: {info_color};
    font-size: 0.8rem;
    font-weight: 600;
  }}

  /* ── Help tooltip ───────────────────────────────────────── */
  .help-icon {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background: {bg_page};
    color: {text_light};
    font-size: 0.7rem;
    font-weight: 800;
    cursor: help;
    position: relative;
    flex-shrink: 0;
    border: 1px solid {border};
  }}
  .help-icon .help-tooltip {{
    display: none;
    position: absolute;
    bottom: 130%;
    left: 50%;
    transform: translateX(-50%);
    width: 320px;
    padding: 0.8rem 1rem;
    background: {bg_dark};
    color: #e2e8f0;
    border-radius: 8px;
    font-size: 0.78rem;
    font-weight: 400;
    line-height: 1.5;
    z-index: 100;
    box-shadow: 0 4px 16px rgba(0,0,0,0.25);
    text-transform: none;
    letter-spacing: normal;
  }}
  .help-icon .help-tooltip::after {{
    content: '';
    position: absolute;
    top: 100%;
    left: 50%;
    transform: translateX(-50%);
    border: 6px solid transparent;
    border-top-color: {bg_dark};
  }}
  .help-icon:hover .help-tooltip {{ display: block; }}

  /* ── Traceback / root-cause panel ───────────────────────── */
  .traceback {{
    background: #fef2f2;
    border: 1px solid #fecaca;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    font-size: 0.82rem;
  }}
  .traceback-title {{
    font-weight: 700;
    color: #991b1b;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
  }}
  .traceback-body {{
    color: #7f1d1d;
    line-height: 1.7;
  }}
  .traceback-body .tb-label {{
    font-weight: 700;
    color: #991b1b;
  }}
  .traceback-body .tb-value {{
    font-family: 'SF Mono', 'Fira Code', monospace;
    background: #fee2e2;
    padding: 0.1rem 0.4rem;
    border-radius: 3px;
    font-size: 0.78rem;
  }}
  .remediation {{
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin-top: 0.6rem;
    font-size: 0.82rem;
    color: #166534;
  }}
  .remediation-title {{
    font-weight: 700;
    margin-bottom: 0.3rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
  }}

  /* ── Drilldown tables (sortable, searchable) ────────────── */
  .drilldown {{
    margin: 0.8rem 0;
  }}
  .drilldown-toggle {{
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.4rem 0.9rem;
    border-radius: 6px;
    border: 1px solid {border};
    background: {bg_page};
    cursor: pointer;
    font-size: 0.8rem;
    font-weight: 600;
    color: {info_color};
    transition: all 0.15s;
  }}
  .drilldown-toggle:hover {{ background: #dbeafe; border-color: {info_color}; }}
  .drilldown-toggle .arrow {{ transition: transform 0.2s; }}
  .drilldown-toggle.open .arrow {{ transform: rotate(90deg); }}
  .drilldown-content {{
    display: none;
    margin-top: 0.6rem;
  }}
  .drilldown-content.open {{ display: block; }}

  .drilldown-search {{
    width: 100%;
    max-width: 350px;
    padding: 0.35rem 0.8rem;
    border-radius: 6px;
    border: 1px solid {border};
    font-size: 0.8rem;
    margin-bottom: 0.5rem;
    outline: none;
  }}
  .drilldown-search:focus {{ border-color: {info_color}; }}

  /* ── Detail tables ──────────────────────────────────────── */
  .detail-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
    margin-top: 0.8rem;
  }}
  .detail-table th {{
    text-align: left;
    padding: 0.5rem 0.8rem;
    background: {bg_page};
    font-weight: 600;
    border-bottom: 1px solid {border};
    cursor: pointer;
    user-select: none;
    white-space: nowrap;
  }}
  .detail-table th:hover {{ background: #e2e8f0; }}
  .detail-table th .sort-arrow {{
    font-size: 0.65rem;
    margin-left: 0.3rem;
    opacity: 0.4;
  }}
  .detail-table th.sorted .sort-arrow {{ opacity: 1; }}
  .detail-table td {{
    padding: 0.5rem 0.8rem;
    border-bottom: 1px solid {border};
    font-family: 'SF Mono', 'Fira Code', monospace;
    font-size: 0.8rem;
  }}
  .detail-table tr:hover td {{ background: #f8fafc; }}
  .detail-table .id-cell {{
    max-width: 250px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }}
  .detail-table .id-cell:hover {{
    overflow: visible;
    white-space: normal;
    word-break: break-all;
  }}

  /* ── Charts ─────────────────────────────────────────────── */
  .chart-container {{
    margin: 1rem 0;
    text-align: center;
    position: relative;
  }}
  .chart-container img {{
    max-width: 100%;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
  }}
  .chart-help {{
    display: flex;
    align-items: flex-start;
    gap: 0.5rem;
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 8px;
    padding: 0.7rem 1rem;
    margin-top: 0.6rem;
    font-size: 0.78rem;
    color: #1e40af;
    line-height: 1.5;
  }}
  .chart-help-icon {{
    flex-shrink: 0;
    font-size: 1rem;
    margin-top: 0.1rem;
  }}

  /* ── Detail JSON ────────────────────────────────────────── */
  .detail-json {{
    background: #f8fafc;
    border: 1px solid {border};
    border-radius: 6px;
    padding: 1rem;
    font-family: 'SF Mono', 'Fira Code', monospace;
    font-size: 0.78rem;
    overflow-x: auto;
    white-space: pre-wrap;
    word-break: break-word;
    max-height: 400px;
    overflow-y: auto;
  }}

  /* ── Hero chart ─────────────────────────────────────────── */
  .hero-chart {{
    background: {bg_card};
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 2rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    text-align: center;
  }}
  .hero-chart h2 {{
    font-size: 1rem;
    margin-bottom: 0.5rem;
    color: {bg_dark};
  }}
  .hero-chart img {{
    max-width: 100%;
    border-radius: 8px;
  }}

  /* ── Footer ─────────────────────────────────────────────── */
  .footer {{
    text-align: center;
    padding: 2rem;
    color: {text_light};
    font-size: 0.8rem;
  }}

  /* ── Sample IDs ─────────────────────────────────────────── */
  .sample-ids {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.3rem;
    margin-top: 0.5rem;
  }}
  .sample-id {{
    background: #f1f5f9;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    font-family: monospace;
    font-size: 0.75rem;
  }}

  /* ── Stat row (key-value pairs inline) ──────────────────── */
  .stat-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 1.5rem;
    margin: 0.6rem 0;
  }}
  .stat-item {{
    display: flex;
    flex-direction: column;
  }}
  .stat-label {{
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: {text_light};
    font-weight: 600;
  }}
  .stat-value {{
    font-size: 1.1rem;
    font-weight: 700;
    font-family: 'SF Mono', 'Fira Code', monospace;
  }}

  /* ── No-results message ─────────────────────────────────── */
  .no-results {{
    text-align: center;
    padding: 2rem;
    color: {text_light};
    font-size: 0.9rem;
    display: none;
  }}

  /* ── Responsive ─────────────────────────────────────────── */
  @media (max-width: 768px) {{
    .header {{ padding: 1.5rem; }}
    .container {{ padding: 1rem; }}
    .summary {{ gap: 0.8rem; }}
    .summary-card {{ min-width: 100px; padding: 1rem; }}
    .summary-card .number {{ font-size: 1.8rem; }}
    .toolbar {{ flex-direction: column; }}
    .help-icon .help-tooltip {{ width: 240px; }}
  }}
</style>
</head>
<body>

<div class="header">
  <h1>&#x1F50D; Vector Database Health Report</h1>
  <div class="subtitle">{report_subtitle}</div>
  <div class="db-badge">{db_label}</div>
</div>

<div class="container">
  {summary_html}

  <!-- Interactive filter toolbar -->
  <div class="toolbar">
    <button class="filter-btn active" data-filter="all" onclick="filterChecks('all', this)">
      All <span class="count">{total_checks}</span>
    </button>
    <button class="filter-btn" data-filter="pass" onclick="filterChecks('pass', this)">
      &#x2705; Passed <span class="count">{passed}</span>
    </button>
    <button class="filter-btn" data-filter="fail" onclick="filterChecks('fail', this)">
      &#x274C; Failed <span class="count">{failed}</span>
    </button>
    <button class="filter-btn" data-filter="info" onclick="filterChecks('info', this)">
      &#x2139;&#xFE0F; Info <span class="count">{inconclusive}</span>
    </button>
    <input type="text" class="search-box" placeholder="&#x1F50D; Search checks, IDs, fields..."
           oninput="searchChecks(this.value)">
  </div>

  <div id="no-results" class="no-results">
    No checks match your filter. Try adjusting the status filter or search term.
  </div>

  {hero_chart_html}
  {checks_html}
</div>

<div class="footer">
  Generated by Vector Database Testing Framework &middot; {timestamp}<br>
  <span style="font-size:0.72rem; color:#cbd5e1;">
    Database-agnostic report &middot; Works with any vector DB implementing the connector interface
  </span>
</div>

<!-- ═══ Interactive JavaScript (no external deps) ═══ -->
<script>
/* ── Status filter ──────────────────────────────────────── */
function filterChecks(status, btn) {{
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');

  const cards = document.querySelectorAll('.check-card');
  let visible = 0;
  cards.forEach(card => {{
    const matchesStatus = status === 'all' || card.classList.contains(status);
    const matchesSearch = card.dataset.searchVisible !== 'false';
    card.style.display = (matchesStatus && matchesSearch) ? '' : 'none';
    if (matchesStatus && matchesSearch) visible++;
  }});

  // Show/hide category headers that have no visible cards
  document.querySelectorAll('.category').forEach(cat => {{
    const visibleCards = cat.querySelectorAll('.check-card:not([style*="display: none"])');
    cat.style.display = visibleCards.length > 0 ? '' : 'none';
  }});

  document.getElementById('no-results').style.display = visible === 0 ? 'block' : 'none';
  window._currentFilter = status;
}}

/* ── Live search ────────────────────────────────────────── */
function searchChecks(query) {{
  const q = query.toLowerCase().trim();
  const cards = document.querySelectorAll('.check-card');
  let visible = 0;
  const currentFilter = window._currentFilter || 'all';

  cards.forEach(card => {{
    const text = card.textContent.toLowerCase();
    const matchesSearch = q === '' || text.includes(q);
    card.dataset.searchVisible = matchesSearch ? 'true' : 'false';
    const matchesStatus = currentFilter === 'all' || card.classList.contains(currentFilter);
    card.style.display = (matchesSearch && matchesStatus) ? '' : 'none';
    if (matchesSearch && matchesStatus) visible++;
  }});

  document.querySelectorAll('.category').forEach(cat => {{
    const visibleCards = cat.querySelectorAll('.check-card:not([style*="display: none"])');
    cat.style.display = visibleCards.length > 0 ? '' : 'none';
  }});

  document.getElementById('no-results').style.display = visible === 0 ? 'block' : 'none';
}}

/* ── Drilldown toggle ───────────────────────────────────── */
function toggleDrilldown(id) {{
  const content = document.getElementById('dd-content-' + id);
  const toggle = document.getElementById('dd-toggle-' + id);
  if (content.classList.contains('open')) {{
    content.classList.remove('open');
    toggle.classList.remove('open');
  }} else {{
    content.classList.add('open');
    toggle.classList.add('open');
  }}
}}

/* ── Drilldown table search ─────────────────────────────── */
function searchDrilldownTable(inputEl, tableId) {{
  const q = inputEl.value.toLowerCase().trim();
  const rows = document.querySelectorAll('#' + tableId + ' tbody tr');
  rows.forEach(row => {{
    const text = row.textContent.toLowerCase();
    row.style.display = (q === '' || text.includes(q)) ? '' : 'none';
  }});
}}

/* ── Table sorting ──────────────────────────────────────── */
function sortTable(tableId, colIdx) {{
  const table = document.getElementById(tableId);
  const tbody = table.querySelector('tbody');
  const rows = Array.from(tbody.querySelectorAll('tr'));
  const th = table.querySelectorAll('th')[colIdx];
  const isAsc = th.dataset.sortDir !== 'asc';
  th.dataset.sortDir = isAsc ? 'asc' : 'desc';

  // Reset all sort arrows
  table.querySelectorAll('th').forEach(h => {{
    h.classList.remove('sorted');
    const arrow = h.querySelector('.sort-arrow');
    if (arrow) arrow.textContent = '⇅';
  }});
  th.classList.add('sorted');
  const arrow = th.querySelector('.sort-arrow');
  if (arrow) arrow.textContent = isAsc ? '↑' : '↓';

  rows.sort((a, b) => {{
    let va = a.cells[colIdx].textContent.trim();
    let vb = b.cells[colIdx].textContent.trim();
    // Try numeric sort
    const na = parseFloat(va), nb = parseFloat(vb);
    if (!isNaN(na) && !isNaN(nb)) {{
      return isAsc ? na - nb : nb - na;
    }}
    return isAsc ? va.localeCompare(vb) : vb.localeCompare(va);
  }});

  rows.forEach(row => tbody.appendChild(row));
}}

window._currentFilter = 'all';
</script>
</body>
</html>"""

# ─── Summary dashboard ────────────────────────────────────────────────
SUMMARY_TEMPLATE = """<div class="summary">
  <div class="summary-card pass">
    <div class="number">{passed}</div>
    <div class="label">Passed</div>
  </div>
  <div class="summary-card fail">
    <div class="number">{failed}</div>
    <div class="label">Failed</div>
  </div>
  <div class="summary-card info">
    <div class="number">{inconclusive}</div>
    <div class="label">Info</div>
  </div>
  <div class="summary-card" style="border-top-color: {bg_dark};">
    <div class="number" style="color: {bg_dark};">{total}</div>
    <div class="label">Total Checks</div>
  </div>
</div>"""

# ─── Hero chart (main UMAP/PCA plot) ─────────────────────────────────
HERO_CHART_TEMPLATE = """<div class="hero-chart">
  <h2>{title}</h2>
  <div class="chart-help">
    <span class="chart-help-icon">&#x2139;&#xFE0F;</span>
    <span>{help_text}</span>
  </div>
  <img src="data:image/png;base64,{image_b64}" alt="{title}">
</div>"""

# ─── Category section ─────────────────────────────────────────────────
CATEGORY_TEMPLATE = """<div class="category">
  <div class="category-title">{category_name}</div>
  {cards_html}
</div>"""

# ─── Individual check card ────────────────────────────────────────────
CHECK_CARD_TEMPLATE = """<div class="check-card {status_class}">
  <div class="check-header">
    <span class="badge {status_class}">{status_label}</span>
    <span class="check-name">{check_name}</span>
    <span class="help-icon">?<span class="help-tooltip">{help_text}</span></span>
  </div>
  {body_html}
</div>"""

# ─── Chart embed with help text ───────────────────────────────────────
CHART_EMBED_TEMPLATE = """<div class="chart-container">
  <img src="data:image/png;base64,{image_b64}" alt="{alt_text}">
  <div class="chart-help">
    <span class="chart-help-icon">&#x2139;&#xFE0F;</span>
    <span>{help_text}</span>
  </div>
</div>"""

# ─── Failure traceback panel ──────────────────────────────────────────
TRACEBACK_TEMPLATE = """<div class="traceback">
  <div class="traceback-title">&#x1F6A8; Failure Analysis</div>
  <div class="traceback-body">
    {traceback_rows}
  </div>
</div>
{remediation_html}"""

TRACEBACK_ROW_TEMPLATE = """<div style="margin-bottom:0.3rem;">
  <span class="tb-label">{label}:</span> <span class="tb-value">{value}</span>
</div>"""

# ─── Remediation panel ───────────────────────────────────────────────
REMEDIATION_TEMPLATE = """<div class="remediation">
  <div class="remediation-title">&#x1F6E0;&#xFE0F; How to Fix</div>
  {steps_html}
</div>"""

REMEDIATION_STEP_TEMPLATE = """<div style="margin-bottom:0.2rem;">&#x2022; {step}</div>"""

# ─── Detail as JSON block ────────────────────────────────────────────
DETAIL_JSON_TEMPLATE = """<div class="detail-json">{json_content}</div>"""

# ─── Key-value detail table ───────────────────────────────────────────
DETAIL_TABLE_TEMPLATE = """<table class="detail-table">
  <tbody>
    {rows_html}
  </tbody>
</table>"""

DETAIL_ROW_TEMPLATE = """<tr><th>{key}</th><td>{value}</td></tr>"""

# ─── Stat row (inline key-value pairs) ────────────────────────────────
STAT_ROW_TEMPLATE = """<div class="stat-row">{items_html}</div>"""

STAT_ITEM_TEMPLATE = """<div class="stat-item">
  <span class="stat-label">{label}</span>
  <span class="stat-value" style="color:{color};">{value}</span>
</div>"""

# ─── Drilldown table with search + sort ───────────────────────────────
DRILLDOWN_TEMPLATE = """<div class="drilldown">
  <button class="drilldown-toggle" id="dd-toggle-{drilldown_id}"
          onclick="toggleDrilldown('{drilldown_id}')">
    <span class="arrow">&#x25B6;</span> {toggle_label} ({item_count} items)
  </button>
  <div class="drilldown-content" id="dd-content-{drilldown_id}">
    <input type="text" class="drilldown-search"
           placeholder="&#x1F50D; Filter rows..."
           oninput="searchDrilldownTable(this, 'dd-table-{drilldown_id}')">
    <table class="detail-table" id="dd-table-{drilldown_id}">
      <thead><tr>{header_cells}</tr></thead>
      <tbody>{body_rows}</tbody>
    </table>
  </div>
</div>"""

DRILLDOWN_HEADER_CELL_TEMPLATE = """<th onclick="sortTable('dd-table-{table_id}', {col_idx})">{col_name} <span class="sort-arrow">&#x21C5;</span></th>"""

DRILLDOWN_ROW_TEMPLATE = """<tr>{cells}</tr>"""

DRILLDOWN_CELL_TEMPLATE = """<td class="{cell_class}">{value}</td>"""

# ─── Sample IDs display ──────────────────────────────────────────────
SAMPLE_IDS_TEMPLATE = """<div class="sample-ids">{ids_html}</div>"""
SAMPLE_ID_TEMPLATE = """<span class="sample-id">{id}</span>"""

# ─── Help text definitions per check type ─────────────────────────────
# These explain what each check does and how to interpret results.
# Used both in the check card tooltip and in chart help text.
HELP_TEXTS = {
    "dimension_consistency": (
        "Verifies every vector in the database has the same number of "
        "dimensions. A mismatch usually means two different embedding models "
        "wrote to the same index, or a pipeline bug truncated some vectors. "
        "All vectors MUST have identical dimensionality for kNN search to work."
    ),
    "magnitude_outliers": (
        "Flags vectors whose L2 norm (magnitude) is far outside the population "
        "distribution (beyond the z-score threshold). Extreme outliers often "
        "indicate encoding errors, model version mismatches, or corrupted data. "
        "The histogram shows the distribution of norms with threshold lines."
    ),
    "near_duplicate_detection": (
        "Finds nearest-neighbor candidates whose cosine similarity meets the "
        "selected threshold (0.65–0.98). Cosine similarity measures embedding "
        "relatedness, not duplicate text. The report also shows word-token "
        "overlap and chunk content to help you judge whether a candidate is "
        "actually a duplicate. This is informational, not an automatic duplicate verdict."
    ),
    "metadata_completeness": (
        "Scans every record and checks whether required metadata fields are "
        "present and non-empty. Missing metadata means records can't be "
        "filtered, traced back to source documents, or properly managed. "
        "The bar chart shows completeness percentage per field."
    ),
    "metadata_value_distribution": (
        "Reports the distribution of values for a metadata field. Useful for "
        "spotting unexpected values, skewed distributions, or fields that are "
        "technically present but always the same value."
    ),
    "known_similar_pairs": (
        "Informational diagnostic: samples consecutive chunks from the same "
        "source, section, and page when available, then reports whether each "
        "appears in the other's top-k neighbors. The report lists every sampled "
        "pair and its rank in both directions. Small collections that cannot "
        "exclude candidates by k are marked inconclusive. Metadata-derived pairs "
        "are not ground-truth relevance labels, so misses are not failures."
    ),
    "searchability_spot_check": (
        "Checks every stored record with a vector by querying its nearest "
        "neighbors and confirming that the record appears in the results. "
        "Records without vectors are counted as failures. This performs one "
        "kNN query per vector-bearing record, so runtime grows with collection size."
    ),
    "empty_vector_audit": (
        "Counts records that have no vector at all (empty list). These records "
        "occupy space but are useless for search and should be removed or "
        "re-embedded."
    ),
    "embedding_visualization": (
        "A 2D projection of the embedding space using PCA or UMAP. Each dot "
        "is one vector. Clusters indicate groups of similar content. Isolated "
        "points may be outliers. Color coding (if applied) shows different "
        "fault types or metadata categories."
    ),
    "drift_visualization": (
        "Side-by-side 2D projection of reference (blue) and current (coral) "
        "vectors. If the two clouds overlap well, there is little drift. If "
        "they are separated, the embedding distribution has shifted."
    ),
}

# ─── Remediation advice per check type ────────────────────────────────
# Structured advice developers can act on immediately.
REMEDIATION_ADVICE = {
    "dimension_consistency": [
        "Identify which embedding model produced the wrong-dimension vectors by checking the pipeline logs for those record IDs.",
        "Re-embed the affected records using the correct model (matching the majority dimension).",
        "Add a dimension validation gate in your ingestion pipeline to reject mismatched vectors before they reach the database.",
    ],
    "magnitude_outliers": [
        "Compare the outlier vectors' source content — they may come from a different model version or encoding scheme.",
        "If outliers are from a different model, re-embed them with the correct model.",
        "Consider L2-normalizing all vectors before insertion if your similarity metric is cosine.",
        "Add a magnitude check to your ingestion pipeline (reject vectors outside expected norm range).",
    ],
    "near_duplicate_detection": [
        "Compare each candidate's text token overlap and chunk content; cosine similarity alone does not mean the records are duplicates.",
        "Check whether high overlap is expected from chunking, such as adjacent or overlapping chunks.",
        "Only remove or merge records after confirming their content is redundant.",
    ],
    "metadata_completeness": [
        "Backfill missing metadata from your source documents using the sample IDs listed below.",
        "Update your ingestion pipeline to require all mandatory fields before insertion.",
        "Add schema validation at the API layer to reject records with missing required fields.",
    ],
    "known_similar_pairs": [
        "Review the automatically sampled adjacent chunks; consecutive chunks are only a likely-similar heuristic.",
        "Check whether source identity and chunk-index metadata correctly identify related chunks.",
        "Use a retrieval evaluation set with expected results when you need a ground-truth quality measurement.",
    ],
    "searchability_spot_check": [
        "The index may be corrupt — try rebuilding it.",
        "Check if the database has resource constraints (memory, disk) that prevent proper indexing.",
        "Verify the distance metric configuration matches what was used during insertion.",
    ],
    "empty_vector_audit": [
        "Re-embed the affected records using the sample IDs listed below.",
        "If these are metadata-only records that don't need vectors, move them to a separate collection.",
        "Add a non-empty vector check to your ingestion pipeline.",
    ],
}

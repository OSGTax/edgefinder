# Dashboard UI Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single-page dashboard with a 5-page fintech-style app using the Midnight Emerald color scheme, TradingView Lightweight Charts, and clean multi-page navigation.

**Architecture:** Jinja2 base template with per-page templates. Static CSS theme file + per-page JS files. FastAPI page routes serve rendered templates. Existing API endpoints unchanged — JS fetches data client-side. TradingView Lightweight Charts (CDN) for price/equity charts, Chart.js for bar/donut/heatmap.

**Tech Stack:** Python 3.11+, FastAPI, Jinja2, vanilla JS, CSS custom properties, TradingView Lightweight Charts (CDN), Chart.js (CDN).

**Spec:** `docs/superpowers/specs/2026-05-19-dashboard-ui-overhaul-design.md`

**Key reference files to read before starting any task:**
- `dashboard/app.py` — FastAPI app, currently serves single `index.html`
- `dashboard/templates/index.html` — Current monolithic dashboard (will be replaced)
- `dashboard/routers/trades.py` — API endpoints for trades data
- `dashboard/routers/strategies.py` — API endpoints for strategy/account data
- `dashboard/routers/research.py` — API endpoints for research/screener data
- `dashboard/routers/benchmarks.py` — API endpoints for benchmark/sector data

---

## File Structure

### New files (create in order)

| File | Responsibility |
|------|---------------|
| `dashboard/static/css/theme.css` | Midnight Emerald palette, card/table/nav components, typography |
| `dashboard/static/js/common.js` | Shared utilities: API fetch, number formatting, time formatting, nav state |
| `dashboard/templates/base.html` | Shared layout: head, nav bar, CDN scripts, page content block |
| `dashboard/routers/pages.py` | GET routes for `/`, `/strategies`, `/screener`, `/trades`, `/research` |
| `dashboard/templates/dashboard.html` | Dashboard page: stats, equity curve, positions, market overview |
| `dashboard/static/js/dashboard.js` | Dashboard page logic: fetch accounts, positions, render charts |
| `dashboard/templates/strategies.html` | Strategies page: strategy cards, detail expansion |
| `dashboard/static/js/strategies.js` | Strategies page logic: fetch per-strategy data, equity curves |
| `dashboard/templates/screener.html` | Screener page: sector rotation, stock table, filters |
| `dashboard/static/js/screener.js` | Screener page logic: fetch active tickers, sort, filter |
| `dashboard/templates/trades.html` | Trades page: stats, calendar, trade journal table |
| `dashboard/static/js/trades.js` | Trades page logic: fetch trades, stats, filters |
| `dashboard/templates/research.html` | Research page: search, ticker profile, charts, data panels |
| `dashboard/static/js/research.js` | Research page logic: search, fetch ticker report, render charts |

### Modified files

| File | Changes |
|------|---------|
| `dashboard/app.py` | Mount static files, add page router, remove old `/` route |

### Removed files

| File | Reason |
|------|--------|
| `dashboard/templates/index.html` | Replaced by base.html + page templates |

---

### Task 1: Foundation — CSS theme, common.js, base template, page router

This task creates all the shared infrastructure. Every subsequent task depends on it.

**Files:**
- Create: `dashboard/static/css/theme.css`
- Create: `dashboard/static/js/common.js`
- Create: `dashboard/templates/base.html`
- Create: `dashboard/routers/pages.py`
- Modify: `dashboard/app.py`

- [ ] **Step 1: Create static directory structure**

```bash
mkdir -p dashboard/static/css dashboard/static/js
```

- [ ] **Step 2: Create `dashboard/static/css/theme.css`**

This is the complete Midnight Emerald theme — palette, typography, navigation, cards, tables, stats, pills, utilities.

```css
/* ═══════════════════════════════════════════════════════════════
   EdgeFinder — Midnight Emerald Theme
   ═══════════════════════════════════════════════════════════════ */

/* ── Palette ──────────────────────────────────────── */
:root {
  --bg: #0b0f14;
  --surface: #111a25;
  --surface-elevated: #162030;
  --border: #1a2332;
  --accent: #00d4a1;
  --positive: #00d4a1;
  --negative: #ef4444;
  --warning: #f0b429;
  --text-primary: #e8f0f8;
  --text-secondary: #6b8aab;
  --text-muted: #4a6a8a;
  --dim-positive: rgba(0,212,161,0.12);
  --dim-negative: rgba(239,68,68,0.12);
  --dim-warning: rgba(240,180,41,0.12);
  --dim-accent: rgba(0,212,161,0.08);
  --radius: 10px;
  --radius-sm: 6px;
  /* Strategy colors */
  --color-coward: #00d4a1;
  --color-gambler: #6ea8fe;
  --color-degenerate: #f0b429;
}

/* ── Reset & Base ─────────────────────────────────── */
*, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', sans-serif;
  background: var(--bg);
  color: var(--text-primary);
  font-size: 13px;
  line-height: 1.5;
  overflow-x: hidden;
  min-height: 100vh;
}
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }

/* ── Scrollbar ────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

/* ── Top Navigation ───────────────────────────────── */
.topnav {
  position: sticky; top: 0; z-index: 100;
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  display: flex; align-items: center;
  padding: 0 24px; height: 52px; gap: 8px;
}
.topnav-brand {
  font-weight: 800; font-size: 16px;
  color: var(--accent); white-space: nowrap;
  display: flex; align-items: center; gap: 8px;
  margin-right: 24px;
}
.topnav-brand .icon { font-size: 20px; }
.topnav-tabs { display: flex; gap: 4px; }
.topnav-tab {
  padding: 14px 16px; font-size: 13px; font-weight: 500;
  color: var(--text-secondary); cursor: pointer;
  border-bottom: 2px solid transparent;
  transition: color 0.2s, border-color 0.2s;
  text-decoration: none;
}
.topnav-tab:hover { color: var(--text-primary); text-decoration: none; }
.topnav-tab.active {
  color: var(--accent);
  border-bottom-color: var(--accent);
}
.topnav-right {
  margin-left: auto; display: flex;
  align-items: center; gap: 16px; font-size: 12px;
}
.topnav-index {
  display: flex; align-items: center; gap: 4px;
  color: var(--text-secondary);
}
.topnav-index .sym { font-weight: 600; }
.topnav-index .val { color: var(--text-primary); font-weight: 600; }
.topnav-status {
  display: flex; align-items: center; gap: 6px;
  color: var(--text-muted); font-size: 11px;
}
.topnav-status .dot {
  width: 8px; height: 8px; border-radius: 50%;
  background: var(--positive);
}

/* ── Page Container ───────────────────────────────── */
.page { max-width: 1440px; margin: 0 auto; padding: 20px 24px; }
.page-title {
  font-size: 20px; font-weight: 700;
  color: var(--text-primary); margin-bottom: 20px;
}

/* ── Cards ────────────────────────────────────────── */
.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  overflow: hidden;
  transition: border-color 0.2s;
}
.card:hover { border-color: var(--surface-elevated); }
.card-header {
  padding: 12px 16px;
  font-size: 11px; font-weight: 600;
  color: var(--accent);
  text-transform: uppercase; letter-spacing: 0.5px;
  border-bottom: 1px solid var(--border);
  display: flex; align-items: center; gap: 8px;
}
.card-header .right { margin-left: auto; display: flex; gap: 8px; align-items: center; }
.card-body { padding: 16px; }

/* ── Stat Cards ───────────────────────────────────── */
.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 12px;
}
.stat-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px;
}
.stat-label {
  font-size: 10px; font-weight: 600;
  color: var(--text-muted);
  text-transform: uppercase; letter-spacing: 0.5px;
}
.stat-value {
  font-size: 24px; font-weight: 800;
  color: var(--text-primary);
  margin-top: 4px; line-height: 1.2;
}
.stat-sub {
  font-size: 12px; font-weight: 500;
  margin-top: 4px;
}

/* ── Grid Layouts ─────────────────────────────────── */
.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
.grid-4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
@media (max-width: 1024px) {
  .grid-2, .grid-3, .grid-4 { grid-template-columns: 1fr; }
}
.gap-12 { gap: 12px; }
.gap-16 { gap: 16px; }
.gap-20 { gap: 20px; }
.mb-12 { margin-bottom: 12px; }
.mb-16 { margin-bottom: 16px; }
.mb-20 { margin-bottom: 20px; }
.mt-20 { margin-top: 20px; }

/* ── Tables ───────────────────────────────────────── */
.data-table { width: 100%; border-collapse: collapse; font-size: 12px; }
.data-table th {
  text-align: left; padding: 8px 12px;
  color: var(--text-muted); font-weight: 600;
  font-size: 10px; text-transform: uppercase;
  letter-spacing: 0.3px;
  border-bottom: 1px solid var(--border);
  background: var(--surface);
  position: sticky; top: 0; cursor: pointer;
  white-space: nowrap; user-select: none;
}
.data-table th:hover { color: var(--text-secondary); }
.data-table td {
  padding: 8px 12px;
  border-bottom: 1px solid rgba(26,35,50,0.5);
  white-space: nowrap;
}
.data-table tr:hover { background: rgba(0,212,161,0.03); }
.data-table .text-right { text-align: right; }

/* ── Pills / Badges ───────────────────────────────── */
.pill {
  display: inline-block; padding: 2px 8px;
  font-size: 10px; font-weight: 700;
  text-transform: uppercase; letter-spacing: 0.3px;
  border-radius: 10px;
}
.pill-positive { background: var(--dim-positive); color: var(--positive); }
.pill-negative { background: var(--dim-negative); color: var(--negative); }
.pill-warning { background: var(--dim-warning); color: var(--warning); }
.pill-accent { background: var(--dim-accent); color: var(--accent); }
.pill-muted { background: rgba(106,138,171,0.1); color: var(--text-muted); }

/* ── Strategy Dots ────────────────────────────────── */
.strat-dot {
  width: 8px; height: 8px; border-radius: 50%;
  display: inline-block;
}
.strat-dot-coward { background: var(--color-coward); }
.strat-dot-gambler { background: var(--color-gambler); }
.strat-dot-degenerate { background: var(--color-degenerate); }

/* ── Gauge Bar ────────────────────────────────────── */
.gauge {
  height: 6px; border-radius: 3px;
  background: var(--border); overflow: hidden;
}
.gauge-fill { height: 100%; border-radius: 3px; transition: width 0.3s; }
.gauge-fill.good { background: var(--positive); }
.gauge-fill.warn { background: var(--warning); }
.gauge-fill.danger { background: var(--negative); }

/* ── Position Progress Bar ────────────────────────── */
.pos-bar {
  height: 8px; border-radius: 4px;
  background: var(--border); position: relative;
  overflow: visible;
}
.pos-bar-fill {
  height: 100%; border-radius: 4px;
  position: absolute; top: 0; left: 0;
}
.pos-bar-marker {
  width: 3px; height: 14px;
  background: var(--text-primary);
  border-radius: 2px;
  position: absolute; top: -3px;
  transform: translateX(-50%);
}

/* ── Filter Controls ──────────────────────────────── */
.filters { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
.filter-select, .filter-input {
  background: var(--bg); border: 1px solid var(--border);
  color: var(--text-primary); padding: 6px 10px;
  font-family: inherit; font-size: 12px;
  border-radius: var(--radius-sm);
}
.filter-select:focus, .filter-input:focus {
  outline: none; border-color: var(--accent);
}
.filter-tabs { display: flex; gap: 2px; }
.filter-tab {
  padding: 5px 12px; cursor: pointer;
  font-size: 11px; font-weight: 500;
  border-radius: 14px; border: 1px solid transparent;
  color: var(--text-secondary);
  transition: all 0.15s;
}
.filter-tab:hover { color: var(--text-primary); }
.filter-tab.active {
  border-color: var(--accent); color: var(--accent);
  background: var(--dim-accent);
}

/* ── Search Input ─────────────────────────────────── */
.search-box {
  position: relative; max-width: 600px; margin: 0 auto;
}
.search-input {
  width: 100%; padding: 14px 16px 14px 44px;
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--radius);
  color: var(--text-primary); font-size: 15px;
  font-family: inherit;
  transition: border-color 0.2s;
}
.search-input:focus {
  outline: none; border-color: var(--accent);
}
.search-input::placeholder { color: var(--text-muted); }
.search-icon {
  position: absolute; left: 16px; top: 50%;
  transform: translateY(-50%);
  color: var(--text-muted); font-size: 16px;
}
.search-results {
  position: absolute; top: 100%; left: 0; right: 0;
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  margin-top: 4px; max-height: 300px; overflow-y: auto;
  display: none; z-index: 50;
}
.search-results.open { display: block; }
.search-result {
  padding: 10px 16px; cursor: pointer;
  display: flex; align-items: center; gap: 8px;
  transition: background 0.1s;
}
.search-result:hover { background: var(--surface-elevated); }
.search-result .sym { font-weight: 700; color: var(--text-primary); }
.search-result .name { color: var(--text-secondary); font-size: 12px; }

/* ── Sector Rotation ──────────────────────────────── */
.sector-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 8px; }
.sector-item {
  display: flex; align-items: center; gap: 8px;
  padding: 10px 12px; background: var(--surface-elevated);
  border-radius: var(--radius-sm); border: 1px solid transparent;
  transition: border-color 0.2s;
}
.sector-item:hover { border-color: var(--border); }
.sector-item .etf { font-weight: 700; width: 40px; }
.sector-item .name { color: var(--text-secondary); flex: 1; font-size: 12px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

/* ── Data Row (label: value) ──────────────────────── */
.data-row {
  display: flex; justify-content: space-between;
  padding: 6px 0; font-size: 13px;
  border-bottom: 1px solid rgba(26,35,50,0.3);
}
.data-row:last-child { border-bottom: none; }
.data-row .label { color: var(--text-secondary); }
.data-row .value { font-weight: 600; color: var(--text-primary); }

/* ── News Items ───────────────────────────────────── */
.news-item {
  padding: 10px 0;
  border-bottom: 1px solid rgba(26,35,50,0.3);
}
.news-item:last-child { border-bottom: none; }
.news-item .headline {
  color: var(--text-primary); font-size: 13px;
  cursor: pointer;
}
.news-item .headline:hover { color: var(--accent); }
.news-item .meta {
  color: var(--text-muted); font-size: 11px;
  margin-top: 4px; display: flex; align-items: center; gap: 8px;
}

/* ── Chips (related tickers) ──────────────────────── */
.chip-row { display: flex; flex-wrap: wrap; gap: 6px; }
.chip {
  padding: 4px 12px; font-size: 12px; font-weight: 600;
  background: var(--surface-elevated); border: 1px solid var(--border);
  border-radius: 14px; cursor: pointer;
  color: var(--text-primary);
  transition: all 0.15s;
}
.chip:hover { border-color: var(--accent); color: var(--accent); }

/* ── Empty State ──────────────────────────────────── */
.empty-state {
  text-align: center; padding: 40px 20px;
  color: var(--text-muted); font-size: 13px;
}
.empty-state .icon { font-size: 32px; margin-bottom: 12px; opacity: 0.5; }

/* ── Helpers ──────────────────────────────────────── */
.text-positive { color: var(--positive); }
.text-negative { color: var(--negative); }
.text-warning { color: var(--warning); }
.text-muted { color: var(--text-muted); }
.text-secondary { color: var(--text-secondary); }
.font-mono { font-family: 'SF Mono', Consolas, 'Courier New', monospace; }
.fw-800 { font-weight: 800; }
.fw-700 { font-weight: 700; }
.truncate { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
```

- [ ] **Step 3: Create `dashboard/static/js/common.js`**

Shared utilities used by all page scripts.

```javascript
/* ═══════════════════════════════════════════════════════════════
   EdgeFinder — Common Utilities
   ═══════════════════════════════════════════════════════════════ */

// ── API Fetch ────────────────────────────────────────────────

async function api(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`API ${path}: ${res.status}`);
  return res.json();
}

// ── Number Formatting ────────────────────────────────────────

function fmtDollar(n) {
  if (n == null) return '—';
  const abs = Math.abs(n);
  if (abs >= 1e12) return (n < 0 ? '-' : '') + '$' + (abs / 1e12).toFixed(1) + 'T';
  if (abs >= 1e9) return (n < 0 ? '-' : '') + '$' + (abs / 1e9).toFixed(1) + 'B';
  if (abs >= 1e6) return (n < 0 ? '-' : '') + '$' + (abs / 1e6).toFixed(1) + 'M';
  return '$' + n.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function fmtPrice(n) {
  if (n == null) return '—';
  return '$' + Number(n).toFixed(2);
}

function fmtPct(n) {
  if (n == null) return '—';
  const sign = n >= 0 ? '+' : '';
  return sign + (n * 100).toFixed(1) + '%';
}

function fmtPctRaw(n) {
  if (n == null) return '—';
  const sign = n >= 0 ? '+' : '';
  return sign + Number(n).toFixed(1) + '%';
}

function fmtNum(n, decimals = 1) {
  if (n == null) return '—';
  return Number(n).toFixed(decimals);
}

function fmtInt(n) {
  if (n == null) return '—';
  return Number(n).toLocaleString('en-US', { maximumFractionDigits: 0 });
}

// ── P&L Formatting ───────────────────────────────────────────

function fmtPnl(n) {
  if (n == null) return '—';
  const sign = n >= 0 ? '+' : '';
  return sign + fmtDollar(n).replace('$', '$');
}

function pnlClass(n) {
  if (n == null) return '';
  return n >= 0 ? 'text-positive' : 'text-negative';
}

function pnlSign(n) {
  if (n == null) return '';
  return n >= 0 ? '▲' : '▼';
}

// ── Time Formatting ──────────────────────────────────────────

function fmtTime(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }) + ' ' +
         d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
}

function fmtDate(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
}

function timeAgo(iso) {
  if (!iso) return '';
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 60) return mins + 'm ago';
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return hrs + 'h ago';
  const days = Math.floor(hrs / 24);
  return days + 'd ago';
}

// ── Strategy Helpers ─────────────────────────────────────────

const STRATEGY_COLORS = {
  coward: '#00d4a1',
  gambler: '#6ea8fe',
  degenerate: '#f0b429',
};

function stratColor(name) {
  return STRATEGY_COLORS[name] || '#6b8aab';
}

function stratDot(name) {
  return `<span class="strat-dot strat-dot-${name}"></span>`;
}

// ── Gauge ────────────────────────────────────────────────────

function gaugeClass(pct) {
  if (pct < 0.10) return 'good';
  if (pct < 0.15) return 'warn';
  return 'danger';
}

// ── Nav State ────────────────────────────────────────────────

function initNav() {
  const path = window.location.pathname;
  document.querySelectorAll('.topnav-tab').forEach(tab => {
    if (tab.getAttribute('href') === path) {
      tab.classList.add('active');
    }
  });
}

// ── Market Indices (top bar) ─────────────────────────────────

async function loadTopBarIndices() {
  try {
    const data = await api('/api/benchmarks/comparison?days=1');
    const el = document.getElementById('topnav-indices');
    if (!el || !data) return;

    const indices = ['SPY', 'QQQ', 'VIX'];
    el.innerHTML = indices.map(sym => {
      const series = data[sym];
      if (!series || !series.length) return '';
      const latest = series[series.length - 1];
      const price = latest.close || latest.value || 0;
      const prev = series.length > 1 ? (series[series.length - 2].close || series[series.length - 2].value || price) : price;
      const chg = prev ? ((price - prev) / prev) * 100 : 0;
      const cls = chg >= 0 ? 'text-positive' : 'text-negative';
      const arrow = chg >= 0 ? '▲' : '▼';
      return `<div class="topnav-index">
        <span class="sym">${sym}</span>
        <span class="val">${fmtNum(price, 1)}</span>
        <span class="${cls}">${arrow}${Math.abs(chg).toFixed(1)}%</span>
      </div>`;
    }).join('');
  } catch (e) {
    // silently fail — indices are decorative
  }
}

async function loadTopBarHealth() {
  try {
    const data = await api('/api/health');
    const el = document.getElementById('topnav-status');
    if (!el) return;
    const ok = data.status === 'ok';
    el.innerHTML = `<span class="dot" style="background:${ok ? 'var(--positive)' : 'var(--negative)'}"></span> v${data.version}`;
  } catch (e) {}
}

// ── Init (runs on every page) ────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  initNav();
  loadTopBarIndices();
  loadTopBarHealth();
});
```

- [ ] **Step 4: Create `dashboard/templates/base.html`**

Shared layout template. All pages extend this.

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{% block title %}EdgeFinder{% endblock %}</title>
  <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>◆</text></svg>">
  <link rel="stylesheet" href="/static/css/theme.css">
  <script src="https://cdn.jsdelivr.net/npm/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <script src="/static/js/common.js"></script>
  {% block head %}{% endblock %}
</head>
<body>

<!-- Top Navigation Bar -->
<nav class="topnav">
  <div class="topnav-brand">
    <span class="icon">◆</span> EdgeFinder
  </div>
  <div class="topnav-tabs">
    <a href="/" class="topnav-tab">Dashboard</a>
    <a href="/strategies" class="topnav-tab">Strategies</a>
    <a href="/screener" class="topnav-tab">Screener</a>
    <a href="/trades" class="topnav-tab">Trades</a>
    <a href="/research" class="topnav-tab">Research</a>
  </div>
  <div class="topnav-right">
    <div id="topnav-indices" style="display:flex;gap:16px;"></div>
    <div id="topnav-status" class="topnav-status">
      <span class="dot"></span> ...
    </div>
  </div>
</nav>

<!-- Page Content -->
<main class="page">
  {% block content %}{% endblock %}
</main>

{% block scripts %}{% endblock %}
</body>
</html>
```

- [ ] **Step 5: Create `dashboard/routers/pages.py`**

Page routes that serve rendered Jinja2 templates.

```python
"""Page routes — serves rendered HTML templates for each dashboard page."""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))


@router.get("/", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    return templates.TemplateResponse(request=request, name="dashboard.html")


@router.get("/strategies", response_class=HTMLResponse)
async def strategies_page(request: Request):
    return templates.TemplateResponse(request=request, name="strategies.html")


@router.get("/screener", response_class=HTMLResponse)
async def screener_page(request: Request):
    return templates.TemplateResponse(request=request, name="screener.html")


@router.get("/trades", response_class=HTMLResponse)
async def trades_page(request: Request):
    return templates.TemplateResponse(request=request, name="trades.html")


@router.get("/research", response_class=HTMLResponse)
async def research_page(request: Request):
    return templates.TemplateResponse(request=request, name="research.html")
```

- [ ] **Step 6: Update `dashboard/app.py`**

Mount static files, add page router, remove old `/` route.

Read `dashboard/app.py` first. Then:

1. Add `from fastapi.staticfiles import StaticFiles` to imports
2. Add `from dashboard.routers import pages` to the router imports (alongside trades, strategies, etc.)
3. After the CORS middleware block, add: `app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")`
4. Replace the existing `app.include_router(pages.router)` BEFORE the API routers (so `/` is handled by pages, not the old route)
5. Remove the old `@app.get("/")` route
6. Add the pages router: `app.include_router(pages.router, tags=["pages"])`

- [ ] **Step 7: Create placeholder page templates**

Create minimal placeholder templates so the app runs. Each page will be fully implemented in subsequent tasks.

Create `dashboard/templates/dashboard.html`:
```html
{% extends "base.html" %}
{% block title %}Dashboard — EdgeFinder{% endblock %}
{% block content %}
<h1 class="page-title">Dashboard</h1>
<div class="empty-state">Loading...</div>
{% endblock %}
```

Create the same pattern for `strategies.html`, `screener.html`, `trades.html`, `research.html` — just changing the title and heading text.

- [ ] **Step 8: Verify the app runs**

```bash
cd /workspaces/edgefinder
uvicorn dashboard.app:app --reload --port 8000
```

Visit `http://localhost:8000/` — should see the nav bar with all 5 tabs and the Dashboard placeholder page. Click each tab to verify navigation works.

Visit `http://localhost:8000/api/health` — should still return health JSON (API routes unchanged).

- [ ] **Step 9: Run tests**

```bash
pytest tests/ -v -m "not integration" --tb=short -q 2>&1 | tail -5
```

Expected: ALL PASS (backend unchanged)

- [ ] **Step 10: Commit**

```bash
git add dashboard/static/ dashboard/templates/base.html dashboard/templates/dashboard.html dashboard/templates/strategies.html dashboard/templates/screener.html dashboard/templates/trades.html dashboard/templates/research.html dashboard/routers/pages.py dashboard/app.py
git commit -m "feat: add foundation — Midnight Emerald theme, base layout, page router"
```

---

### Task 2: Dashboard page — "How am I doing right now?"

**Files:**
- Modify: `dashboard/templates/dashboard.html`
- Create: `dashboard/static/js/dashboard.js`

- [ ] **Step 1: Implement `dashboard/templates/dashboard.html`**

The implementer should read the spec section "Page 1: Dashboard" and build the full template. The page has 4 sections:

1. **Hero stats row** (4 stat-cards): Total Equity, Today's P&L, Open Positions, Win Rate
2. **Equity curve** (full-width card with TradingView Lightweight Chart)
3. **Open positions grid** (card per position with progress bar from stop → target)
4. **Market overview row** (5 small index cards: SPY, QQQ, IWM, DIA, VIX)

The template extends `base.html` and includes `dashboard.js` in the `scripts` block.

All data is fetched client-side by `dashboard.js` from:
- `/api/strategies/accounts` — strategy account data (equity, cash, positions)
- `/api/strategies/equity-curve?days=90` — equity curve time series
- `/api/strategies/positions` — open positions with current prices
- `/api/trades/stats` — win rate, trade count
- `/api/benchmarks/comparison?days=1` — index prices

- [ ] **Step 2: Implement `dashboard/static/js/dashboard.js`**

The JS file should:
- Fetch all required API data on page load
- Render the 4 hero stat cards with computed aggregates
- Create a TradingView Lightweight Chart for the equity curve (area chart, accent color, gradient fill)
- Render position cards with P&L progress bars
- Render market overview index cards
- Handle empty states gracefully (no positions, no trades yet)

- [ ] **Step 3: Verify in browser**

Start the dev server and verify the dashboard page renders correctly with real API data. Check all 4 sections.

- [ ] **Step 4: Commit**

```bash
git add dashboard/templates/dashboard.html dashboard/static/js/dashboard.js
git commit -m "feat: add Dashboard page — stats, equity curve, positions, market overview"
```

---

### Task 3: Strategies page — "How is each strategy performing?"

**Files:**
- Modify: `dashboard/templates/strategies.html`
- Create: `dashboard/static/js/strategies.js`

- [ ] **Step 1: Implement `dashboard/templates/strategies.html`**

Read spec section "Page 2: Strategies". The page has:

1. **Strategy cards row** (3 across): Equity, cash, drawdown gauge, win rate, avg R, trade count, risk parameters, "View Details" button
2. **Strategy detail panel** (expands when clicking a strategy): Per-strategy equity curve (TradingView chart), trade history table, watchlist preview, risk budget visualization

The template extends `base.html` and includes `strategies.js`.

Data sources:
- `/api/strategies/accounts` — all strategy account states
- `/api/strategies/equity-curve?days=90` — equity curve per strategy
- `/api/trades?strategy=X` — trades filtered by strategy
- `/api/trades/stats?strategy=X` — per-strategy stats

- [ ] **Step 2: Implement `dashboard/static/js/strategies.js`**

The JS should:
- Fetch strategy accounts and render cards with the correct strategy color (coward=emerald, gambler=blue, degenerate=gold)
- Show drawdown as a gauge bar (green < 10%, yellow 10-15%, red > 15%)
- Show paused strategies with a warning badge
- Handle "View Details" click — fetch per-strategy equity curve and trades, render inline detail panel
- Create per-strategy TradingView chart with the strategy's color

- [ ] **Step 3: Verify in browser**

Check all 3 strategy cards render, drawdown gauges work, clicking "View Details" expands correctly.

- [ ] **Step 4: Commit**

```bash
git add dashboard/templates/strategies.html dashboard/static/js/strategies.js
git commit -m "feat: add Strategies page — strategy cards, detail panel, equity curves"
```

---

### Task 4: Screener page — "What's the market showing me?"

**Files:**
- Modify: `dashboard/templates/screener.html`
- Create: `dashboard/static/js/screener.js`

- [ ] **Step 1: Implement `dashboard/templates/screener.html`**

Read spec section "Page 3: Screener". The page has:

1. **Sector rotation card** — grid of sector ETFs with quadrant badges (Leading/Improving/Weakening/Lagging)
2. **Filter bar** — strategy dropdown (dynamically populated), sector dropdown, search input
3. **Stock screener table** — sortable columns: Symbol, Company, Sector, Price, MktCap, EG%, RG%, P/E, RSI, MACD, SI%, Strategies

Data sources:
- `/api/benchmarks/sectors` — sector rotation data
- `/api/research/active` — full screener data with fundamentals + technicals
- `/api/strategies` — strategy names for filter dropdown

- [ ] **Step 2: Implement `dashboard/static/js/screener.js`**

The JS should:
- Fetch and render sector rotation grid with color-coded quadrant badges
- Fetch active tickers and render the sortable table
- Color-code RSI (< 30 green oversold, > 70 red overbought)
- Color-code earnings growth (positive green, negative red)
- Show qualifying strategies as colored dots
- Implement client-side sorting (click column header to toggle sort)
- Implement client-side filtering (strategy dropdown, sector dropdown, text search)
- Dynamically populate strategy dropdown from `/api/strategies`
- Extract unique sectors from data for sector dropdown

- [ ] **Step 3: Verify in browser**

Check sector rotation renders, screener table populates, sorting and filtering work.

- [ ] **Step 4: Commit**

```bash
git add dashboard/templates/screener.html dashboard/static/js/screener.js
git commit -m "feat: add Screener page — sector rotation, filterable stock table"
```

---

### Task 5: Trades page — "What happened?"

**Files:**
- Modify: `dashboard/templates/trades.html`
- Create: `dashboard/static/js/trades.js`

- [ ] **Step 1: Implement `dashboard/templates/trades.html`**

Read spec section "Page 4: Trades". The page has:

1. **Stats bar** (6 stat cards): Total trades, Win Rate, Avg R, Best trade, Worst trade, Avg Hold
2. **Trade journal table** with filters: Strategy dropdown, status tabs (All/Open/Wins/Losses)

Columns: Symbol, Strategy, Direction, Entry Price, Exit Price, Shares, P&L $, P&L %, R-Multiple, Status, Entry Reasoning, Exit Reasoning, Hold Duration, Entry Time, Exit Time

Data sources:
- `/api/trades/stats` — aggregate trade statistics
- `/api/trades` — full trade list
- `/api/trades?strategy=X` — filtered by strategy

- [ ] **Step 2: Implement `dashboard/static/js/trades.js`**

The JS should:
- Fetch trade stats and render the 6 stat cards
- Fetch trades and render the journal table
- Implement status filter tabs (All/Open/Wins/Losses) — filter client-side
- Implement strategy dropdown filter — refetch from API with `?strategy=X`
- Color-code P&L columns (positive green, negative red)
- Show entry/exit reasoning truncated with hover to expand
- Show PDT flag and hold duration if available
- Handle empty state (no trades yet)

- [ ] **Step 3: Verify in browser**

Check stats render, trade table populates, filters work, P&L coloring is correct.

- [ ] **Step 4: Commit**

```bash
git add dashboard/templates/trades.html dashboard/static/js/trades.js
git commit -m "feat: add Trades page — stats, trade journal with filters"
```

---

### Task 6: Research page — "Tell me everything about this ticker"

**Files:**
- Modify: `dashboard/templates/research.html`
- Create: `dashboard/static/js/research.js`

- [ ] **Step 1: Implement `dashboard/templates/research.html`**

Read spec section "Page 5: Research". The page has:

1. **Search bar** — centered, prominent, with autocomplete dropdown
2. **Ticker profile** (shown after search):
   - Header card: company name, sector, price, market cap, qualifying strategy dots
   - Price chart: TradingView Lightweight Chart (area or candlestick)
   - Two-column layout: Fundamentals (left), Technicals (right)
   - Short Interest card
   - Dividends table
   - News feed with sentiment badges
   - Trade history for this ticker
   - Related tickers as clickable chips

Data sources:
- `/api/research/search?q=X` — autocomplete search
- `/api/research/ticker/{symbol}` — full ticker report (fundamentals, indicators, news, dividends, splits, trades, related)

- [ ] **Step 2: Implement `dashboard/static/js/research.js`**

The JS should:
- Implement search with debounced autocomplete (fetch from `/api/research/search?q=...` after 2+ chars)
- On ticker selection, fetch `/api/research/ticker/{symbol}` and render the full profile
- Create a TradingView Lightweight Chart for the price (using indicator data for EMAs if available)
- Render fundamentals as a data-row grid (label: value pairs)
- Render technicals with RSI as a colored gauge bar
- Render short interest card
- Render dividends as a simple table
- Render news items with sentiment badges (positive=green, negative=red, neutral=gray) and timeAgo
- Render trade history table filtered to this ticker
- Render related tickers as clickable chips that reload the profile
- Support URL parameter `?ticker=AAPL` for direct linking (from Dashboard position clicks)
- Handle empty states (no data, no trades, no news)

- [ ] **Step 3: Verify in browser**

Search for a known ticker (e.g., AAPL). Verify all sections render. Click a related ticker to verify navigation works.

- [ ] **Step 4: Commit**

```bash
git add dashboard/templates/research.html dashboard/static/js/research.js
git commit -m "feat: add Research page — ticker search, full profile with charts and data"
```

---

### Task 7: Cleanup, remove old dashboard, version bump

**Files:**
- Delete: `dashboard/templates/index.html`
- Modify: `dashboard/app.py` (version bump)

- [ ] **Step 1: Delete old index.html**

```bash
rm dashboard/templates/index.html
```

- [ ] **Step 2: Verify no references to index.html remain**

Search for `index.html` in the codebase. The old `@app.get("/")` route should already be removed (Task 1 step 6). If any test references it, update the test.

- [ ] **Step 3: Bump version**

In `dashboard/app.py`, change `__version__` from `"5.0.1"` to `"5.1.0"`.

- [ ] **Step 4: Run full test suite**

```bash
pytest tests/ -v -m "not integration" --tb=short -q
```

Expected: ALL PASS

- [ ] **Step 5: Final visual verification**

Start the dev server and visit each page:
1. `/` — Dashboard with stats, equity curve, positions, market overview
2. `/strategies` — Strategy cards with details
3. `/screener` — Sector rotation + sortable screener table
4. `/trades` — Stats + trade journal with filters
5. `/research` — Search bar, select a ticker, verify full profile
6. `/api/health` — API still works
7. Verify nav bar active tab highlights correctly on each page

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "[v5.1.0] dashboard UI overhaul — 5-page fintech app with Midnight Emerald theme"
```

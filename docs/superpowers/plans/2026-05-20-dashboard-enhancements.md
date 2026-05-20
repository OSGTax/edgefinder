# Dashboard Enhancement Pack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 9 visual enhancements to the existing dashboard — trade timelines, strategy comparison, streaks, P&L calendar, RSI gauges, sector treemap, position alerts, auto-refresh, and dark/light mode.

**Architecture:** All changes are frontend-only — modifying existing JS, CSS, and HTML template files. No new API endpoints, no backend Python changes. One new CDN dependency (chartjs-chart-treemap). Each task groups features by the page they belong to.

**Tech Stack:** Vanilla JS, CSS custom properties, TradingView Lightweight Charts (existing), Chart.js + treemap plugin (CDN), Jinja2 templates.

**Spec:** `docs/superpowers/specs/2026-05-20-dashboard-enhancements-design.md`

**Key files to read before starting:**
- `dashboard/static/css/theme.css` — Current Midnight Emerald theme (365 lines)
- `dashboard/static/js/common.js` — Shared utilities (174 lines)
- `dashboard/templates/base.html` — Shared layout (43 lines)
- `dashboard/static/js/dashboard.js` — Dashboard page logic (323 lines)
- `dashboard/static/js/trades.js` — Trades page logic (294 lines)
- `dashboard/static/js/strategies.js` — Strategies page logic (475 lines)
- `dashboard/static/js/screener.js` — Screener page logic (297 lines)

---

## File Structure

All modifications — no new files created.

| File | Enhancements Added |
|------|-------------------|
| `dashboard/static/css/theme.css` | Light mode, pulse animations, calendar heatmap, trade timeline, RSI gauge styles |
| `dashboard/static/js/common.js` | Theme toggle, localStorage helpers, strategy target constants |
| `dashboard/templates/base.html` | Theme toggle button, treemap CDN script |
| `dashboard/static/js/dashboard.js` | Strategy comparison chart, position alerts, auto-refresh |
| `dashboard/templates/dashboard.html` | Comparison chart card, auto-refresh UI |
| `dashboard/static/js/trades.js` | Trade reasoning timeline, P&L calendar heatmap |
| `dashboard/templates/trades.html` | Calendar card, timeline expansion rows |
| `dashboard/static/js/strategies.js` | Win/loss streak indicators |
| `dashboard/static/js/screener.js` | RSI gauges, MACD coloring, sector treemap |
| `dashboard/templates/screener.html` | Treemap canvas |

---

### Task 1: Theme infrastructure — light mode + toggle (Global)

This task adds the CSS light mode overrides, the toggle button in the nav, and the JS to persist the preference. Every subsequent task benefits from having the theme system in place.

**Files:**
- Modify: `dashboard/static/css/theme.css`
- Modify: `dashboard/static/js/common.js`
- Modify: `dashboard/templates/base.html`

- [ ] **Step 1: Add light mode CSS overrides to `theme.css`**

Append to the end of `dashboard/static/css/theme.css`:

```css
/* ── Light Mode ───────────────────────────────────── */
body.light-mode {
  --bg: #f5f7fa;
  --surface: #ffffff;
  --surface-elevated: #f0f2f5;
  --border: #e2e8f0;
  --text-primary: #1a202c;
  --text-secondary: #64748b;
  --text-muted: #94a3b8;
  --dim-positive: rgba(0,212,161,0.10);
  --dim-negative: rgba(239,68,68,0.10);
  --dim-warning: rgba(240,180,41,0.10);
  --dim-accent: rgba(0,212,161,0.06);
}

/* ── Theme Toggle Button ──────────────────────────── */
.theme-toggle {
  background: none; border: none; cursor: pointer;
  font-size: 18px; padding: 4px 8px; border-radius: 6px;
  transition: background 0.2s;
  line-height: 1;
}
.theme-toggle:hover { background: var(--surface-elevated); }
```

- [ ] **Step 2: Add theme toggle JS to `common.js`**

Add before the `document.addEventListener('DOMContentLoaded', ...)` block in `dashboard/static/js/common.js`:

```javascript
// ── Theme Toggle ─────────────────────────────────────────────

function initTheme() {
  const saved = localStorage.getItem('ef-theme');
  if (saved === 'light') {
    document.body.classList.add('light-mode');
  }
}

function toggleTheme() {
  const isLight = document.body.classList.toggle('light-mode');
  localStorage.setItem('ef-theme', isLight ? 'light' : 'dark');
  const btn = document.getElementById('theme-toggle-btn');
  if (btn) btn.textContent = isLight ? '☀️' : '🌙';
}
```

Then add `initTheme();` as the first line inside the existing `DOMContentLoaded` handler.

- [ ] **Step 3: Add toggle button and treemap CDN to `base.html`**

In `dashboard/templates/base.html`:

1. In the `<head>` section, after the Chart.js script tag, add:
```html
<script src="https://cdn.jsdelivr.net/npm/chartjs-chart-treemap@2.3.0"></script>
```

2. In the `.topnav-right` div, before the `#topnav-status` div, add:
```html
<button id="theme-toggle-btn" class="theme-toggle" onclick="toggleTheme()">🌙</button>
```

3. Add a `<script>` block right after the `<body>` tag (before the nav) to prevent flash:
```html
<script>
  if (localStorage.getItem('ef-theme') === 'light') document.body.classList.add('light-mode');
</script>
```

- [ ] **Step 4: Verify**

Open the dashboard in a browser. Click the moon icon — page should switch to light mode with white background. Click again — back to dark. Refresh the page — preference should persist.

- [ ] **Step 5: Commit**

```bash
git add dashboard/static/css/theme.css dashboard/static/js/common.js dashboard/templates/base.html
git commit -m "feat: add dark/light mode toggle with localStorage persistence"
```

---

### Task 2: Dashboard enhancements — comparison chart, position alerts, auto-refresh

**Files:**
- Modify: `dashboard/static/js/dashboard.js`
- Modify: `dashboard/templates/dashboard.html`
- Modify: `dashboard/static/css/theme.css`

- [ ] **Step 1: Add CSS for pulse animation and auto-refresh to `theme.css`**

Append to `dashboard/static/css/theme.css`:

```css
/* ── Position Alert Pulse ─────────────────────────── */
@keyframes pulse-red {
  0%, 100% { box-shadow: 0 0 0 0 rgba(239,68,68,0.4); }
  50% { box-shadow: 0 0 0 8px rgba(239,68,68,0); }
}
@keyframes pulse-green {
  0%, 100% { box-shadow: 0 0 0 0 rgba(0,212,161,0.4); }
  50% { box-shadow: 0 0 0 8px rgba(0,212,161,0); }
}
.pulse-stop { animation: pulse-red 2s ease-in-out infinite; border-color: var(--negative) !important; }
.pulse-target { animation: pulse-green 2s ease-in-out infinite; border-color: var(--positive) !important; }

.alert-badge {
  font-size: 10px; font-weight: 700; padding: 2px 6px;
  border-radius: 8px; text-transform: uppercase; letter-spacing: 0.3px;
}
.alert-badge.stop { background: var(--dim-negative); color: var(--negative); }
.alert-badge.target { background: var(--dim-positive); color: var(--positive); }

/* ── Auto-Refresh Bar ─────────────────────────────── */
.refresh-bar {
  display: flex; align-items: center; gap: 8px;
  font-size: 11px; color: var(--text-muted);
}
.refresh-toggle {
  background: none; border: 1px solid var(--border); color: var(--text-secondary);
  cursor: pointer; font-size: 14px; padding: 2px 8px; border-radius: 6px;
  transition: all 0.15s; line-height: 1;
}
.refresh-toggle:hover { border-color: var(--accent); color: var(--accent); }
.refresh-toggle.active { border-color: var(--accent); color: var(--accent); background: var(--dim-accent); }
```

- [ ] **Step 2: Update `dashboard.html` template**

Read `dashboard/templates/dashboard.html` first. Then add:

1. After the page title `<h1>`, add the auto-refresh bar:
```html
<div class="refresh-bar mb-16">
  <button id="refresh-toggle" class="refresh-toggle active" onclick="toggleAutoRefresh()">⟳</button>
  <span id="refresh-status">Auto-refresh: on · Updated just now</span>
</div>
```

2. After the equity curve card, add the strategy comparison card:
```html
<!-- Strategy Comparison -->
<div class="card mb-20">
  <div class="card-header">
    Strategy Comparison
    <div class="right">
      <div class="filter-tabs" id="cmp-range-tabs">
        <div class="filter-tab" data-days="30">30D</div>
        <div class="filter-tab active" data-days="90">90D</div>
        <div class="filter-tab" data-days="180">180D</div>
        <div class="filter-tab" data-days="365">1Y</div>
      </div>
    </div>
  </div>
  <div class="card-body">
    <div id="comparison-chart" style="height:280px;"></div>
    <div id="comparison-legend" style="display:flex;gap:16px;justify-content:center;margin-top:8px;font-size:12px;"></div>
  </div>
</div>
```

- [ ] **Step 3: Add comparison chart, position alerts, and auto-refresh to `dashboard.js`**

Read `dashboard/static/js/dashboard.js` first. The implementer should add three new features:

**A. Strategy Comparison Chart** — new function `loadComparisonChart(days=90)`:
- Fetch `/api/strategies/equity-curve?days=N` and `/api/benchmarks/comparison?days=N`
- For each strategy, normalize equity to cumulative % change from start: `((equity - first_equity) / first_equity) * 100`
- SPY data is already cumulative % from the benchmarks API
- Create a TradingView Lightweight Chart with 4 line series:
  - Coward line color `#00d4a1`, Gambler `#6ea8fe`, Degenerate `#f0b429`, SPY `#6b8aab`
- Render legend below with colored dots + strategy names
- Wire up the range tabs (30D/90D/180D/1Y) same pattern as the equity curve tabs
- Handle empty data (no equity curve yet) with a "No data yet" message

**B. Position Alerts** — modify existing `loadPositions()`:
- After computing current price and P&L for each position, check:
  - Stop price = entry × 0.80 (20% stop). If current < stop × 1.05 → add `pulse-stop` class + "⚠️ NEAR STOP" badge
  - Target price: use `STRATEGY_TARGETS` map: `{coward: 0.15, gambler: 0.25, degenerate: 0.50}`. Target = entry × (1 + target_pct). If current > target × 0.95 → add `pulse-target` class + "🎯 NEAR TARGET" badge
- Add a constant at the top of the file:
```javascript
const STRATEGY_TARGETS = { coward: 0.15, gambler: 0.25, degenerate: 0.50 };
```

**C. Auto-Refresh** — new functions `startAutoRefresh()`, `stopAutoRefresh()`, `toggleAutoRefresh()`:
- 60-second interval that calls `loadStats()`, `loadPositions()`, `loadMarketOverview()` (NOT equity curves)
- Track `lastUpdate = Date.now()` and update the status text every second: "Updated Xs ago"
- Save preference to `localStorage('ef-auto-refresh')`
- On load, start auto-refresh if preference is not 'off'
- Call `startAutoRefresh()` at the end of the DOMContentLoaded handler

- [ ] **Step 4: Verify**

1. Dashboard should show the comparison chart below the equity curve with 4 colored lines
2. If any position is near its stop/target, the card should pulse
3. The refresh bar should count up and refresh data every 60 seconds
4. Toggle button should pause/resume

- [ ] **Step 5: Commit**

```bash
git add dashboard/static/css/theme.css dashboard/static/js/dashboard.js dashboard/templates/dashboard.html
git commit -m "feat: add strategy comparison chart, position alerts, auto-refresh"
```

---

### Task 3: Trades enhancements — reasoning timeline + P&L calendar heatmap

**Files:**
- Modify: `dashboard/static/js/trades.js`
- Modify: `dashboard/templates/trades.html`
- Modify: `dashboard/static/css/theme.css`

- [ ] **Step 1: Add CSS for timeline and calendar to `theme.css`**

Append to `dashboard/static/css/theme.css`:

```css
/* ── Trade Reasoning Timeline ─────────────────────── */
.trade-timeline {
  display: flex; align-items: stretch; gap: 0;
  padding: 16px; background: var(--bg); border-radius: var(--radius-sm);
}
.timeline-card {
  flex: 1; padding: 12px; background: var(--surface);
  border-radius: var(--radius-sm); border: 1px solid var(--border);
}
.timeline-card h4 {
  font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px;
  margin-bottom: 8px;
}
.timeline-card.entry h4 { color: var(--accent); }
.timeline-card.exit h4 { color: var(--text-secondary); }
.timeline-connector {
  display: flex; flex-direction: column; align-items: center;
  justify-content: center; padding: 0 16px; min-width: 80px;
}
.timeline-line {
  width: 2px; flex: 1; background: var(--border);
}
.timeline-dot {
  width: 10px; height: 10px; border-radius: 50%;
  background: var(--accent); margin: 4px 0;
}
.timeline-duration {
  font-size: 11px; color: var(--text-muted);
  text-align: center; margin: 4px 0; white-space: nowrap;
}
.timeline-indicators {
  display: grid; grid-template-columns: 1fr 1fr;
  gap: 4px; margin-top: 8px; font-size: 11px;
}
.timeline-indicators .label { color: var(--text-muted); }
.timeline-indicators .value { color: var(--text-primary); font-weight: 600; text-align: right; }

/* ── P&L Calendar Heatmap ─────────────────────────── */
.calendar-grid {
  display: grid;
  grid-template-columns: 30px repeat(13, 1fr);
  gap: 3px; font-size: 10px;
}
.calendar-label {
  color: var(--text-muted); display: flex;
  align-items: center; font-size: 10px;
}
.calendar-cell {
  aspect-ratio: 1; border-radius: 3px;
  cursor: pointer; position: relative;
  transition: transform 0.1s;
  min-width: 14px; min-height: 14px;
}
.calendar-cell:hover { transform: scale(1.3); z-index: 1; }
.calendar-tooltip {
  display: none; position: absolute; bottom: 120%; left: 50%;
  transform: translateX(-50%); background: var(--surface);
  border: 1px solid var(--border); border-radius: 4px;
  padding: 4px 8px; font-size: 10px; white-space: nowrap;
  z-index: 10; pointer-events: none;
  color: var(--text-primary);
}
.calendar-cell:hover .calendar-tooltip { display: block; }
```

- [ ] **Step 2: Update `trades.html` template**

Read `dashboard/templates/trades.html` first. Add the P&L calendar card between the stats grid and the filters:

```html
<!-- P&L Calendar -->
<div class="card mb-20">
  <div class="card-header">Daily P&L <span id="calendar-range" style="color:var(--text-muted);font-weight:400;text-transform:none;letter-spacing:0;"></span></div>
  <div class="card-body">
    <div id="pnl-calendar" class="calendar-grid"></div>
  </div>
</div>
```

- [ ] **Step 3: Add timeline expansion and calendar to `trades.js`**

Read `dashboard/static/js/trades.js` first. The implementer should add two features:

**A. Trade Reasoning Timeline** — modify the table row rendering:
- Each trade row gets a click handler that toggles a detail row below it
- The detail row contains a `.trade-timeline` with:
  - Left: `.timeline-card.entry` — entry reasoning text + key indicators from `indicators_at_entry` (RSI, MACD histogram, close price, EMA 21)
  - Center: `.timeline-connector` with dots, line, and hold duration label
  - Right: `.timeline-card.exit` — exit reasoning text + key indicators from `indicators_at_exit`
- If `indicators_at_entry` is null (deferred columns not in DB), show only the reasoning text
- If `entry_reasoning` is also null, show "No reasoning captured"
- Track which row is expanded and collapse it when clicking another row
- The detail row spans all columns: `<tr><td colspan="13">...</td></tr>`

**B. P&L Calendar Heatmap** — new function `renderCalendar(trades)`:
- Takes the loaded trades array
- Groups closed trades by exit date (YYYY-MM-DD), sums `pnl_dollars` per day
- Builds a 7-row × 13-column grid (91 days = 13 weeks)
- First column is day labels (Mon, Tue, ..., Sun)
- Each cell is a `.calendar-cell` with:
  - Background color: interpolate between red (max loss) → gray (no trades) → green (max profit)
  - Tooltip showing date and P&L: "May 15: +$320.00" or "May 15: no trades"
- Color interpolation: `opacity = |pnl| / maxAbsPnl`, color is `--positive` for profit, `--negative` for loss, `var(--border)` for no-trade days
- Call `renderCalendar(_allTrades)` after loading trades
- Show date range in the header: "Last 90 days"

- [ ] **Step 4: Verify**

1. Click a trade row → detail expansion shows timeline with entry/exit cards
2. Calendar heatmap renders above the table with colored cells
3. Hover a calendar cell → tooltip shows date + P&L

- [ ] **Step 5: Commit**

```bash
git add dashboard/static/css/theme.css dashboard/static/js/trades.js dashboard/templates/trades.html
git commit -m "feat: add trade reasoning timeline and P&L calendar heatmap"
```

---

### Task 4: Strategies enhancement — win/loss streak indicators

**Files:**
- Modify: `dashboard/static/js/strategies.js`

- [ ] **Step 1: Add streak computation to `strategies.js`**

Read `dashboard/static/js/strategies.js` first. The implementer should:

**A. Add a streak computation function:**
```javascript
function computeStreak(trades) {
  // Filter to closed trades, sort by exit_time descending
  const closed = trades
    .filter(t => t.status === 'CLOSED' && t.pnl_dollars != null)
    .sort((a, b) => new Date(b.exit_time) - new Date(a.exit_time));

  if (closed.length === 0) return { type: 'none', count: 0 };

  const firstIsWin = closed[0].pnl_dollars > 0;
  let count = 0;
  for (const t of closed) {
    const isWin = t.pnl_dollars > 0;
    if (isWin === firstIsWin) count++;
    else break;
  }

  if (count < 2) return { type: 'none', count: 0 };
  return { type: firstIsWin ? 'win' : 'loss', count };
}
```

**B. In the strategy card rendering** (inside `loadStrategyCards` or equivalent), after the win rate / avg R / trades row:
- Fetch trades for each strategy: `/api/trades?strategy=X`
- Compute streak with `computeStreak(trades)`
- If streak count >= 2, add a line below the stats:
  - Win streak: `<div style="font-size:12px;" class="text-positive">🔥 ${count}W streak</div>`
  - Loss streak: `<div style="font-size:12px;" class="text-negative">❄️ ${count}L streak</div>`
  - No streak: don't show anything

**Note:** The trades fetch may already happen for per-strategy stats. If so, reuse that data rather than making a duplicate request.

- [ ] **Step 2: Verify**

Strategy cards should show streak indicators below the stats row when a strategy has 2+ consecutive wins or losses.

- [ ] **Step 3: Commit**

```bash
git add dashboard/static/js/strategies.js
git commit -m "feat: add win/loss streak indicators to strategy cards"
```

---

### Task 5: Screener enhancements — RSI gauges, MACD coloring, sector treemap

**Files:**
- Modify: `dashboard/static/js/screener.js`
- Modify: `dashboard/templates/screener.html`
- Modify: `dashboard/static/css/theme.css`

- [ ] **Step 1: Add CSS for RSI gauge to `theme.css`**

Append to `dashboard/static/css/theme.css`:

```css
/* ── RSI Gauge (inline) ───────────────────────────── */
.rsi-gauge {
  display: inline-flex; align-items: center; gap: 4px;
}
.rsi-bar {
  width: 50px; height: 6px; border-radius: 3px;
  background: var(--border); overflow: hidden;
  display: inline-block; vertical-align: middle;
}
.rsi-fill {
  height: 100%; border-radius: 3px;
  transition: width 0.3s;
}
.rsi-fill.oversold { background: var(--positive); }
.rsi-fill.neutral { background: var(--accent); }
.rsi-fill.overbought { background: var(--negative); }
```

- [ ] **Step 2: Update `screener.html` for treemap**

Read `dashboard/templates/screener.html` first. Replace the sector grid content:

Change the sector card body from `<div class="sector-grid" id="sector-grid"></div>` to:
```html
<div id="sector-grid">
  <canvas id="sector-treemap" height="200"></canvas>
</div>
```

- [ ] **Step 3: Update `screener.js` — RSI gauges, MACD coloring, treemap**

Read `dashboard/static/js/screener.js` first. The implementer should make three changes:

**A. RSI Gauge in table** — modify the row rendering for the RSI column:
Instead of just showing the number, render:
```javascript
function renderRsiGauge(rsi) {
  if (rsi == null) return '—';
  const pct = Math.min(100, Math.max(0, rsi));
  const cls = rsi < 30 ? 'oversold' : rsi > 70 ? 'overbought' : 'neutral';
  return `<span class="rsi-gauge">
    <span class="rsi-bar"><span class="rsi-fill ${cls}" style="width:${pct}%"></span></span>
    <span style="font-size:11px;${rsi < 30 ? 'color:var(--positive)' : rsi > 70 ? 'color:var(--negative)' : ''}">${rsi.toFixed(0)}</span>
  </span>`;
}
```

**B. MACD coloring** — modify the MACD column rendering:
```javascript
function renderMacd(val) {
  if (val == null) return '—';
  const cls = val > 0 ? 'text-positive' : 'text-negative';
  const arrow = val > 0 ? '▲' : '▼';
  return `<span class="${cls}">${arrow} ${val.toFixed(2)}</span>`;
}
```

**C. Sector Treemap** — replace the sector grid rendering (`loadSectors` or equivalent) with:
- Fetch `/api/benchmarks/sectors` as before
- Also count stocks per sector from the screener data (group by `sector` field)
- Build a Chart.js treemap chart on the `#sector-treemap` canvas:
```javascript
const treemapData = sectors.map(s => ({
  sector: s.symbol,
  name: s.name || s.symbol,
  count: stockCountBySector[s.name] || stockCountBySector[s.symbol] || 1,
  quadrant: s.quadrant || 'unknown',
}));

const quadrantColors = {
  leading: '#00d4a1', improving: '#f0b429',
  weakening: '#4a6a8a', lagging: '#ef4444',
  unknown: '#2a3a4a',
};

new Chart(document.getElementById('sector-treemap'), {
  type: 'treemap',
  data: {
    datasets: [{
      tree: treemapData,
      key: 'count',
      labels: { display: true, formatter: (ctx) => ctx.raw._data.sector },
      backgroundColor: (ctx) => quadrantColors[ctx.raw._data.quadrant] || '#2a3a4a',
      borderColor: 'var(--border)',
      borderWidth: 1,
      spacing: 2,
    }]
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          title: (items) => items[0].raw._data.name,
          label: (item) => `${item.raw._data.count} stocks · ${item.raw._data.quadrant}`,
        }
      }
    },
    onClick: (evt, elements) => {
      if (elements.length > 0) {
        const sector = elements[0].element.$context.raw._data.name;
        // Set sector filter and re-render table
        document.getElementById('filter-sector').value = sector;
        applyFilters(); // or whatever the filter function is called
      }
    }
  }
});
```

If no sector data is available, fall back to showing a message: "Sector data unavailable — collecting at 4:15 PM ET"

- [ ] **Step 4: Verify**

1. Screener table RSI column shows colored gauge bars instead of plain numbers
2. MACD column shows colored arrows
3. Sector card shows a treemap with colored boxes
4. Click a treemap sector → filters the table

- [ ] **Step 5: Commit**

```bash
git add dashboard/static/css/theme.css dashboard/static/js/screener.js dashboard/templates/screener.html
git commit -m "feat: add RSI gauges, MACD coloring, sector treemap to Screener"
```

---

### Task 6: Version bump and final verification

**Files:**
- Modify: `dashboard/app.py`

- [ ] **Step 1: Bump version**

Change `__version__` in `dashboard/app.py` from `"5.1.0"` to `"5.2.0"`.

- [ ] **Step 2: Run tests**

```bash
pytest tests/ -v -m "not integration" --tb=short -q
```

Expected: ALL PASS (no backend changes)

- [ ] **Step 3: Visual verification**

Check each page in the browser:
1. **Global**: Dark/light mode toggle works, persists across pages
2. **Dashboard**: Strategy comparison chart shows 4 lines, position cards pulse when near stop/target, auto-refresh counter ticks and refreshes data
3. **Strategies**: Win/loss streak indicators appear on cards with 2+ consecutive results
4. **Screener**: Sector treemap renders with clickable boxes, RSI gauges in table, MACD colored
5. **Trades**: P&L calendar heatmap renders with colored cells, clicking a trade row expands the reasoning timeline

- [ ] **Step 4: Commit**

```bash
git add dashboard/app.py
git commit -m "[v5.2.0] dashboard enhancement pack — 9 visual features"
```

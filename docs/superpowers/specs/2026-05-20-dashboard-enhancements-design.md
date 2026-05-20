# Dashboard Enhancement Pack — Design Spec

## Problem

The dashboard pages are functional but static and lack the visual depth that makes data actionable for traders. Key data like trade reasoning, strategy comparison, and P&L patterns are available in the API but not surfaced in the UI.

## Solution

9 frontend-only enhancements across existing pages. No new API endpoints, no backend changes, no new Python dependencies.

## Features

### 1. Trade Reasoning Timeline (Trades page — `trades.js`)

When a user clicks a trade row, expand an inline detail panel showing a horizontal timeline:

```
[Entry]──────────────────────[Exit]
RSI 28.4 → Coward entered    RSI 72.1 → Overbought exit
$182.40 entry                 $191.20 exit
                              +$1,320 (+4.8%) · 3.2 days
```

**Data source:** Trade object fields: `entry_reasoning`, `exit_reasoning`, `indicators_at_entry`, `indicators_at_exit`, `hold_duration_hours`, `pdt_flag`

**Implementation:**
- Click a trade row → toggle a `<tr>` detail row below it
- Left side: entry card with reasoning + key indicators (RSI, MACD, price, EMA)
- Right side: exit card with reasoning + key indicators
- Connecting line between them with hold duration label
- If indicators_at_entry/exit are null (deferred columns not yet in DB), show reasoning text only
- Color the timeline green (win) or red (loss)

### 2. Strategy Comparison Chart (Dashboard page — `dashboard.js`)

New full-width card below the equity curve: "Strategy Comparison"

- TradingView Lightweight Chart with 3 line series (one per strategy) + 1 for SPY benchmark
- Strategy colors: coward=`#00d4a1`, gambler=`#6ea8fe`, degenerate=`#f0b429`, SPY=`#6b8aab`
- Legend below the chart showing strategy names with colored dots
- Same 30D/90D/180D/1Y range selector as the equity curve

**Data sources:**
- `/api/strategies/equity-curve?days=N` — per-strategy equity data (normalize to % change from start for comparison)
- `/api/benchmarks/comparison?days=N` — SPY cumulative % change (already in this format)

**Normalization:** Convert each strategy's equity to cumulative % change from its starting point so they're comparable on the same Y-axis (all start at 0%).

### 3. Win/Loss Streak Indicators (Strategies page — `strategies.js`)

On each strategy card, below the win rate row, show current streak:

- "🔥 4W streak" (green text) if last 4+ closed trades were wins
- "❄️ 2L streak" (red text) if last 2+ closed trades were losses
- "—" if only 1 trade or no streak

**Computation:** Client-side from `/api/trades?strategy=X`. Sort closed trades by exit_time descending. Walk from newest, count consecutive same-result trades until the result changes.

### 4. P&L Calendar Heatmap (Trades page — `trades.js`)

New card above the trade table: "Daily P&L"

GitHub-contribution-style grid showing 90 days:

```
Mon ░░░█░░░░░██░░░
Tue ░░█░░░░░░░░░░░
Wed ░░░░░░█░░░░░░░
...
```

- Each cell = one day
- Color scale: deep green (#00d4a1 at 100% opacity) → light green → gray (no trades) → light red → deep red (#ef4444)
- Tooltip on hover: date + net P&L dollar amount
- 13 columns (weeks) × 7 rows (days)

**Computation:** Client-side from trades data. Group closed trades by exit date, sum `pnl_dollars` per day. Map to color intensity: normalize against the max absolute P&L day.

**Implementation:** Pure CSS grid — no Chart.js needed. Each cell is a `<div>` with inline background-color.

### 5. RSI Gauge Bars in Screener (Screener page — `screener.js`)

Replace the plain RSI number column with a visual gauge:

```
[████████░░] 62
```

- Gauge width proportional to RSI value (0-100)
- Color: green if RSI < 30 (oversold), red if RSI > 70 (overbought), accent for 30-70
- Number displayed to the right of the gauge

Replace the plain MACD number with a colored value:
- Green text + "▲" if MACD histogram > 0
- Red text + "▼" if MACD histogram < 0
- Muted if null

**Why not sparklines:** The screener's `/api/research/active` endpoint returns point-in-time values, not time series. Gauges are the best visual for the data available.

### 6. Sector Treemap (Screener page — `screener.js`)

Replace the current sector grid card with a treemap visualization.

- Uses Chart.js with the `chartjs-chart-treemap` plugin (CDN: `https://cdn.jsdelivr.net/npm/chartjs-chart-treemap@2.3.0`)
- Each box = one sector ETF
- Box size = number of qualifying stocks in that sector (from screener data)
- Box color = quadrant color (leading=`#00d4a1`, improving=`#f0b429`, weakening=`#4a6a8a`, lagging=`#ef4444`)
- Box label = ETF symbol + stock count
- Click a sector box → sets the sector filter dropdown and triggers table re-render

**Fallback:** If no sector data from `/api/benchmarks/sectors`, show the stock-count-only treemap with neutral colors.

### 7. Position P&L Alerts (Dashboard page — `dashboard.js`)

Enhance existing position cards with visual alerts when price is near stop or target:

**Near stop (within 5% of stop loss price):**
- Card border becomes red with pulse animation
- Badge: "⚠️ NEAR STOP" in red pill

**Near target (within 5% of target price):**
- Card border becomes green with pulse animation
- Badge: "🎯 NEAR TARGET" in green pill

**Computation:** For each position:
- Stop price = entry_price × (1 - 0.20) = 20% below entry
- Target price = entry_price × (1 + strategy_target_pct)
- Strategy target: coward=0.15, gambler=0.25, degenerate=0.50
- Near stop = current_price < stop_price × 1.05
- Near target = current_price > target_price × 0.95

**CSS animation:**
```css
@keyframes pulse-alert {
  0%, 100% { box-shadow: 0 0 0 0 rgba(color, 0.4); }
  50% { box-shadow: 0 0 0 8px rgba(color, 0); }
}
```

### 8. Auto-Refresh (Dashboard page — `dashboard.js`)

- 60-second interval refreshes: stats, positions, market overview (NOT equity curve — too expensive)
- Header shows "Last updated: Xs ago" counter that ticks every second
- Toggle button (pause/play icon) to disable auto-refresh
- Preference saved to `localStorage` (key: `ef-auto-refresh`)
- Auto-refresh ONLY on Dashboard page. Other pages are static.

**Implementation:**
```javascript
let refreshInterval = null;
let lastUpdate = Date.now();

function startAutoRefresh() {
  refreshInterval = setInterval(() => {
    loadStats();
    loadPositions();
    loadMarketOverview();
    lastUpdate = Date.now();
  }, 60000);
}
```

### 9. Dark/Light Mode Toggle (Global — `base.html` + `theme.css` + `common.js`)

Toggle button in the nav bar right section (before the status indicator):
- 🌙 icon = currently dark mode (click to switch to light)
- ☀️ icon = currently light mode (click to switch to dark)

**Light mode palette** (CSS class `.light-mode` on `<body>`):

| Role | Dark | Light |
|------|------|-------|
| `--bg` | `#0b0f14` | `#f5f7fa` |
| `--surface` | `#111a25` | `#ffffff` |
| `--surface-elevated` | `#162030` | `#f0f2f5` |
| `--border` | `#1a2332` | `#e2e8f0` |
| `--text-primary` | `#e8f0f8` | `#1a202c` |
| `--text-secondary` | `#6b8aab` | `#64748b` |
| `--text-muted` | `#4a6a8a` | `#94a3b8` |

Accent colors (`--accent`, `--positive`, `--negative`, `--warning`) stay the same in both modes.

**Implementation:**
- Add `.light-mode` CSS block to `theme.css` that overrides custom properties
- Toggle function in `common.js` that adds/removes the class and saves to `localStorage`
- On page load, check `localStorage.getItem('ef-theme')` and apply
- TradingView charts need their background/text colors updated when theme switches

**Preference persistence:** `localStorage.setItem('ef-theme', 'light')` / `'dark'`

## Files Modified

| File | Changes |
|------|---------|
| `dashboard/static/css/theme.css` | Light mode overrides, pulse animation, calendar heatmap styles, timeline styles |
| `dashboard/static/js/common.js` | Theme toggle function, localStorage helpers |
| `dashboard/templates/base.html` | Theme toggle button in nav bar |
| `dashboard/static/js/dashboard.js` | Strategy comparison chart, position alerts, auto-refresh, updated equity chart |
| `dashboard/templates/dashboard.html` | Strategy comparison card, auto-refresh UI |
| `dashboard/static/js/strategies.js` | Win/loss streak computation and display |
| `dashboard/static/js/trades.js` | Trade timeline expansion, P&L calendar heatmap |
| `dashboard/templates/trades.html` | Calendar heatmap card, trade detail expansion container |
| `dashboard/static/js/screener.js` | RSI gauges, MACD coloring, sector treemap |
| `dashboard/templates/screener.html` | Treemap canvas container |

## CDN Dependencies (added to base.html)

```html
<script src="https://cdn.jsdelivr.net/npm/chartjs-chart-treemap@2.3.0"></script>
```

## What's NOT changing

- No new API endpoints
- No backend Python changes
- No new Python dependencies
- Existing page structure and routing unchanged

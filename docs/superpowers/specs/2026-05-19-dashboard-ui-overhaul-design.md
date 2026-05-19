# Dashboard UI Overhaul — Design Spec

## Problem

The current dashboard is a single-page dark UI with everything stacked vertically in one giant HTML file. It's visually generic (GitHub-dark), overwhelming for non-technical traders, and doesn't showcase the rich data EdgeFinder collects (fundamentals, technicals, sentiment, news, dividends, short interest, trade reasoning).

## Solution

Replace the single-page dashboard with a 5-page fintech-style app. Each page has a clear purpose, uses a polished "Midnight Emerald" color scheme, and presents data in a way that non-technical traders can understand at a glance.

## Design Principles

- **Non-technical first**: Labels say "Win Rate" not "Sharpe Ratio". Green means good, red means bad.
- **Progressive disclosure**: Overview first, click for details. Never show 25 columns at once.
- **Charts are primary**: TradingView Lightweight Charts for price/equity. Visual > tabular.
- **One page, one question**: Each page answers one thing a trader would ask.

## Technical Approach

### No framework — upgraded vanilla stack

- **Jinja2 templates**: One base layout template, one HTML file per page
- **TradingView Lightweight Charts** (CDN): Price charts, equity curves — professional out of the box
- **Chart.js** (already in use): Bar charts, donuts, heatmaps
- **Vanilla JS**: Client-side fetch to existing API endpoints
- **CSS custom properties**: Single theme file, Midnight Emerald palette
- **No build step**: No React, no npm, no bundler. Deploys with `pip install`.

### File structure

```
dashboard/
├── templates/
│   ├── base.html            # Shared layout: nav bar, head, scripts, theme CSS
│   ├── dashboard.html        # Dashboard page
│   ├── strategies.html       # Strategies page
│   ├── screener.html         # Screener page
│   ├── trades.html           # Trades page
│   └── research.html         # Research page
├── static/
│   ├── css/
│   │   └── theme.css         # Midnight Emerald palette + shared component styles
│   └── js/
│       ├── common.js         # Shared utilities (fetch, formatting, nav state)
│       ├── dashboard.js      # Dashboard page logic
│       ├── strategies.js     # Strategies page logic
│       ├── screener.js       # Screener page logic
│       ├── trades.js         # Trades page logic
│       └── research.js       # Research page logic
├── routers/
│   └── pages.py              # Page routes (GET /, /strategies, /screener, etc.)
└── app.py                    # Add static file mounting + page router
```

### Backend changes

- New `dashboard/routers/pages.py` with GET routes for each page (returns rendered Jinja2 template)
- Mount `/static` directory in FastAPI for CSS/JS files
- Existing API endpoints unchanged — JS fetches from them client-side
- Remove old `index.html` (replaced by `base.html` + page templates)

## Color Palette — Midnight Emerald

Deep navy-black base with emerald green accents. Premium, calm, confident.

| Role | Color | Usage |
|------|-------|-------|
| `--bg` | `#0b0f14` | Page background |
| `--surface` | `#111a25` | Card backgrounds |
| `--surface-elevated` | `#162030` | Hover states, active cards |
| `--border` | `#1a2332` | Card borders, dividers |
| `--accent` | `#00d4a1` | Primary accent — logo, active tabs, CTAs |
| `--positive` | `#00d4a1` | Profit, gains, success |
| `--negative` | `#ef4444` | Loss, drawdown, errors |
| `--warning` | `#f0b429` | Caution, paused states |
| `--text-primary` | `#e8f0f8` | Headings, values, primary content |
| `--text-secondary` | `#6b8aab` | Descriptions, secondary labels |
| `--text-muted` | `#4a6a8a` | Timestamps, tertiary info |
| `--dim-positive` | `rgba(0,212,161,0.12)` | Positive background tint |
| `--dim-negative` | `rgba(239,68,68,0.12)` | Negative background tint |

### Strategy colors (for charts with multiple strategies)

| Strategy | Color |
|----------|-------|
| Coward | `#00d4a1` (emerald) |
| Gambler | `#6ea8fe` (blue) |
| Degenerate | `#f0b429` (gold) |

## Global Layout — base.html

### Top navigation bar

Fixed at top, consistent across all pages:

```
┌─────────────────────────────────────────────────────────────────┐
│  ◆ EdgeFinder          Dashboard  Strategies  Screener  Trades  Research     SPY 582 ▲1.2%  QQQ 510 ▲0.8%  VIX 14.2  ● v5.0.1 │
└─────────────────────────────────────────────────────────────────┘
```

- **Left**: Logo icon + "EdgeFinder" in accent color
- **Center**: Tab links — active tab highlighted with accent underline + text color
- **Right**: Market indices (SPY, QQQ, VIX) with directional arrows + color. System status dot (green = healthy). Version number.

### Typography

- Font: `-apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', sans-serif`
- Big numbers (equity, P&L): 24px, weight 800
- Card values: 16px, weight 700
- Body text: 13px, weight 400
- Labels: 10px, uppercase, letter-spacing 0.5px, `--text-muted`
- Tables: 12px

### Card component

Every data section is a card:
- Background: `--surface`
- Border: 1px solid `--border`
- Border radius: 10px
- Header: 11px uppercase label in `--accent`, optional badge/filter on right
- Body: 14px padding
- Hover: border shifts to `--surface-elevated`

## Page 1: Dashboard — "How am I doing right now?"

### Hero stats row (4 cards)

```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Total Equity  │  │ Today's P&L  │  │    Open      │  │  Win Rate    │
│   $14,832     │  │   +$247.80   │  │  Positions   │  │    67%       │
│  ▲ +8.3%      │  │              │  │      7       │  │   42 trades  │
└──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘
```

- Total Equity: sum of all strategy accounts. Change from starting capital as percentage.
- Today's P&L: sum of unrealized P&L changes today. Green/red colored.
- Open Positions: count across all strategies.
- Win Rate: closed trades with positive P&L / total closed trades.

### Equity curve (full width)

TradingView Lightweight Chart — area chart showing total equity over time.
- Data source: `/api/strategies/equity-curve`
- Time range selector: 30D / 90D / 180D / 1Y
- Line color: `--accent`
- Fill: gradient from accent to transparent

### Open positions grid

Card grid, one card per open position:

```
┌─────────────────────────────┐
│  AAPL          coward       │
│  150 shares @ $182.40       │
│  Current: $191.20           │
│  ┌───────────────────────┐  │
│  │ ████████████░░░░░░░░░ │  │  ← progress bar (entry → target, current position marked)
│  └───────────────────────┘  │
│  +$1,320.00  (+4.8%)    ▲   │
└─────────────────────────────┘
```

- Strategy shown as colored dot + name
- P&L bar: visual from stop → entry → target with current price marked
- Color: entire card gets subtle green/red tint based on P&L direction
- Click: navigates to Research page for that ticker

### Market overview row

Small horizontal cards for SPY, QQQ, IWM, DIA, VIX:

```
┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐
│ SPY  582.41│  │ QQQ  510.20│  │ IWM  222.10│  │ DIA  428.30│  │ VIX  14.2  │
│    ▲ +1.2% │  │    ▲ +0.8% │  │    ▼ -0.3% │  │    ▲ +0.5% │  │    ▼ -2.1% │
└────────────┘  └────────────┘  └────────────┘  └────────────┘  └────────────┘
```

Data source: from intraday snapshot data or `/api/benchmarks/comparison`

## Page 2: Strategies — "How is each strategy performing?"

### Strategy cards row

One large card per strategy (3 across):

```
┌──────────────────────────────────┐
│  ● Coward          conservative  │
│                                  │
│  Equity: $5,412    Cash: $3,200  │
│  ┌────────────────────────┐      │
│  │ Drawdown: ██░░░░ 4.2%  │      │  ← gauge bar
│  └────────────────────────┘      │
│                                  │
│  Win Rate    Avg R    Trades     │
│    71%       1.8R       14      │
│                                  │
│  Risk: 5%   Target: 15%  Stop: 20% │
│                                  │
│  [View Details →]                │
└──────────────────────────────────┘
```

- Strategy dot color matches chart color (emerald/blue/gold)
- Drawdown shown as horizontal gauge (green < 10%, yellow 10-15%, red > 15%)
- Paused strategies show a "PAUSED" badge with warning color

### Strategy detail (click "View Details")

Expands below the cards or replaces the view:

- **Equity curve**: Per-strategy TradingView chart
- **Trade history table**: Filtered to this strategy — symbol, entry/exit, P&L, R-multiple, reasoning
- **Watchlist**: Top 10 qualifying tickers with price and key metric
- **Risk budget**: Visual showing deployed capital vs available

## Page 3: Screener — "What's the market showing me?"

### Sector rotation heatmap

```
┌──────────────────────────────────────────────────────┐
│  Sector Rotation (RRG)                               │
│                                                      │
│  XLK ████ Leading    XLP ████ Improving              │
│  XLV ████ Improving  XLE ████ Lagging                │
│  XLF ████ Lagging    XLI ████ Lagging                │
│  ...                                                 │
└──────────────────────────────────────────────────────┘
```

- Grid of sector ETFs with quadrant badge (Leading/Improving/Weakening/Lagging)
- Color-coded: green = leading, gold = improving, gray = weakening, red = lagging
- Data source: `/api/benchmarks/sectors`

### Stock screener table

Full-width sortable table with key columns:

| Column set | Fields |
|-----------|--------|
| Identity | Symbol, Company, Sector |
| Price | Price, Market Cap |
| Growth | Earnings Growth %, Revenue Growth % |
| Value | P/E, P/B, PEG |
| Health | D/E, Current Ratio, FCF Yield |
| Technical | RSI (color-coded), MACD histogram |
| Short | Short Interest %, Days to Cover |
| Strategy | Qualifying strategy dots |

- **Filter bar**: Strategy dropdown, sector dropdown, search box
- **Click a row**: Slide-out panel with quick summary (key fundamentals + technicals + strategy status). "Open in Research →" link.
- **Color coding**: RSI < 30 green (oversold), > 70 red (overbought). Earnings growth green if positive, red if negative.
- Data source: `/api/research/active`

## Page 4: Trades — "What happened?"

### Stats bar

```
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│  Total   │ │ Win Rate │ │  Avg R   │ │  Best    │ │  Worst   │ │ Avg Hold │
│   42     │ │   67%    │ │  1.4R    │ │ +$820    │ │ -$340    │ │  3.2 days│
└──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘
```

Data source: `/api/trades/stats`

### P&L calendar heatmap

Grid of days (like GitHub contribution graph), colored by daily P&L:
- Deep green: strong profit day
- Light green: small profit
- Gray: no trades
- Light red: small loss
- Deep red: big loss day

Built with Chart.js matrix plugin or custom CSS grid.

### Trade journal table

Full trade history with filters:

| Columns | |
|---------|--|
| Symbol, Strategy, Direction | Identity |
| Entry Price, Exit Price, Shares | Execution |
| P&L $, P&L %, R-Multiple | Performance |
| Status (Open/Closed) | State |
| Entry Reasoning, Exit Reasoning | Context (new!) |
| Hold Duration, PDT Flag | Metadata (new!) |
| Entry Time, Exit Time | Timestamps |

- **Filters**: Strategy dropdown, status tabs (All/Open/Wins/Losses), date range picker
- **Click a row**: Expand inline to show full reasoning, indicator snapshot at entry/exit, market context
- Data source: `/api/trades`

## Page 5: Research — "Tell me everything about this ticker"

### Search bar

Prominent centered search input with autocomplete dropdown:
```
┌─────────────────────────────────────────────┐
│  🔍  Search any ticker...                    │
└─────────────────────────────────────────────┘
```
- Autocomplete fetches from `/api/research/search?q=...`
- Shows symbol + company name in dropdown
- Enter or click → loads full profile

### Ticker profile (loads after search)

#### Header card
```
┌─────────────────────────────────────────────────────────────────┐
│  AAPL    Apple Inc.    Technology    $3.2T market cap           │
│  $191.20  ▲ +1.8%                                              │
│  ● coward  ● gambler                    (qualifying strategies) │
└─────────────────────────────────────────────────────────────────┘
```

#### Price chart (full width)

TradingView Lightweight Chart:
- Candlestick or area chart (toggle)
- EMA 9/21/50/200 overlays (toggleable)
- Volume bars below
- Data source: Polygon daily bars via `/api/research/ticker/{symbol}` (indicators field)

#### Two-column data layout

**Left column — Fundamentals:**

```
┌─────────────────────────────┐
│  FUNDAMENTALS               │
│                             │
│  P/E Ratio        28.4     │
│  P/B Ratio         4.2     │
│  PEG Ratio         1.8     │
│  EV/EBITDA        22.1     │
│                             │
│  Earnings Growth  +15.2%   │
│  Revenue Growth    +8.4%   │
│  ROE              +42.1%   │
│  ROA              +18.7%   │
│                             │
│  D/E Ratio         1.2     │
│  Current Ratio     1.8     │
│  FCF Yield         3.4%    │
│  Quick Ratio       1.5     │
└─────────────────────────────┘
```

**Right column — Technicals:**

```
┌─────────────────────────────┐
│  TECHNICALS                 │
│                             │
│  RSI (14)    ████████░░ 62  │  ← gauge bar, colored
│                             │
│  MACD Line        +1.24    │
│  MACD Signal      +0.98    │
│  MACD Histogram   +0.26    │
│                             │
│  EMA 9           $190.40   │
│  EMA 21          $188.20   │
│  SMA 50          $185.10   │
│                             │
│  Stochastic RSI    68.4    │
│  Williams %R      -31.6    │
│  ADX               24.2    │
│  BB Width           4.8%   │
└─────────────────────────────┘
```

**Short Interest card:**

```
┌─────────────────────────────┐
│  SHORT INTEREST             │
│                             │
│  Short Interest    2.4%    │
│  Days to Cover     1.8     │
│  Short Shares      142M    │
└─────────────────────────────┘
```

#### Dividends table

| Ex-Date | Pay Date | Amount | Frequency |
|---------|----------|--------|-----------|
| 2026-05-10 | 2026-05-15 | $0.25 | Quarterly |
| 2026-02-08 | 2026-02-13 | $0.24 | Quarterly |

Data source: `/api/research/ticker/{symbol}` → dividends array

#### News feed

```
┌─────────────────────────────────────────────────┐
│  NEWS                                           │
│                                                 │
│  Apple Reports Record Q2 Earnings    positive   │
│  Reuters · 2 hours ago                          │
│                                                 │
│  iPhone Sales Slow in China          negative   │
│  Bloomberg · 5 hours ago                        │
│                                                 │
│  Apple Announces AI Partnership      neutral    │
│  CNBC · 1 day ago                               │
└─────────────────────────────────────────────────┘
```

- Sentiment tag colored (green/red/gray)
- Clickable headlines (open article URL)
- Data source: `/api/research/ticker/{symbol}` → recent_news array

#### Trade history on this ticker

Table of all EdgeFinder trades for this symbol:

| Strategy | Entry | Exit | Shares | P&L | Reasoning |
|----------|-------|------|--------|-----|-----------|
| coward | $182.40 | $191.20 | 150 | +$1,320 | RSI oversold at 28.4 |

#### Related tickers

Horizontal row of clickable chips:
```
MSFT  GOOG  AMZN  META  NVDA
```
Click → loads that ticker's profile (same page, new data)

## Migration Path

1. Create `dashboard/static/` directory with CSS and JS files
2. Create `dashboard/templates/base.html` layout template
3. Create each page template one at a time
4. Add `dashboard/routers/pages.py` with page routes
5. Mount static files in `app.py`
6. Update `app.py` to serve new pages at `/`, `/strategies`, `/screener`, `/trades`, `/research`
7. Remove old `index.html`
8. The old strategy dropdown in the screener hardcodes alpha/bravo/charlie — replace with dynamic strategy names from the API

## What's NOT changing

- All API endpoints stay the same
- Backend logic unchanged
- No new Python dependencies
- No build step or npm
- TradingView Lightweight Charts and Chart.js loaded from CDN

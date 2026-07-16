/* Trading Desk — the autonomous agent's window.
   Four panels: (stats + equity curve), holdings, live thinking feed,
   the latest decision (picks with why-now / rationale / news + watchlist),
   and the evidence (backtests it ran) + strategy journal. Reads the new
   /api/desk/* endpoints; reuses the core charts/dom/fmt/net modules. */

import { apiGet } from '../core/net.js';
import { toEpochSec, fmtDollar, fmtPnl, fmtPct, fmtPrice, fmtNum, timeAgo, fmtDateTimeET }
  from '../core/fmt.js';
import { h, svg, clear, skeleton, renderEmpty, renderError } from '../core/dom.js';
import { createChart, colors } from '../core/charts.js';
import { onThemeChange } from '../core/theme.js';

let equityChart = null;
let equitySeries = null;   // the recorded marks — history, never fabricated
let spySeries = null;      // faint benchmark overlay: $100k in SPY (total return)
let liveTipSeries = null;  // ONE dashed connector: last real mark → live estimate
let lastEquityData = [];

/* Live-marked book state — the source-of-truth "reference" portfolio (cash +
   position shares/avg + starting) and the running dict of live mids from the
   SSE tape. `applyLiveMarks` folds these two into a fresh, tick-fresh view
   of hero cards + positions rows, so the desk stops looking frozen between
   trading routines. */
const deskLive = {
  book: null,
  marks: {},
  stats: {},
  todayChip: null,
  lastEquity: null,   // for tick-direction (up/down) coloring
  lastTickTs: 0,      // for the "updated Xs ago" age chip
};

// Rolling "updated Xs ago" tick under the LIVE chip. Runs once per second
// once the first fold has landed, so a stalled stream reads visibly stale.
setInterval(() => {
  const el = document.getElementById('desk-hero-live-age');
  if (!el || !deskLive.lastTickTs) return;
  const age = Math.round((Date.now() - deskLive.lastTickTs) / 1000);
  el.textContent = age <= 1 ? 'just now' : `${age}s ago`;
}, 1000);

const ACTION_CLASS = { buy: 'up', add: 'up', hold: 'neutral', trim: 'warn', exit: 'down', sell: 'down' };

function pill(text, cls, title) {
  return h('span', { class: 'c-pill ' + (cls || 'neutral'), title: title || null, text });
}

/* ── LIVE pill truth ──
   The pulsing green LIVE dot only appears when a threshold fraction of the
   HELD symbols' quotes are genuinely fresh AND the market session is open
   (the SSE payload carries per-symbol staleness + a session tag). A stale
   tape shows DELAYED; a closed market shows CLOSED — frozen numbers never
   masquerade as ticking ones. */
const LIVE_PILL = {
  live: ['LIVE', 'Fresh real-time quotes are folding into the account value — '
    + 'the numbers above are the live market value.'],
  delayed: ['DELAYED', 'The live quote stream is stale — the numbers above are '
    + 'the most recent recorded prices, not a live market value.'],
  closed: ['CLOSED', 'The market is closed — prices shown are the last '
    + 'session’s marks.'],
};

function setLivePill(state) {
  const el = document.getElementById('desk-hero-live');
  if (!el) return;
  if (!state) { el.hidden = true; return; }
  const [label, tip] = LIVE_PILL[state] || LIVE_PILL.delayed;
  el.hidden = false;
  el.classList.toggle('delayed', state === 'delayed');
  el.classList.toggle('closed', state === 'closed');
  el.title = tip;
  const lbl = el.querySelector('.desk-hero-live-label');
  if (lbl) lbl.textContent = label;
}

/* What the pill should say for this SSE frame. null = keep it hidden (an
   all-cash or all-options book has nothing that folds live). */
function livePillState(snap) {
  const book = deskLive.book;
  const heldEq = (book && book.positions ? book.positions : [])
    .filter(p => !occParse(p.symbol)).map(p => p.symbol);
  if (!heldEq.length) return null;
  if (snap.session === 'closed') return 'closed';
  const quotes = snap.quotes || {};
  const fresh = heldEq.filter(s => {
    const q = quotes[s];
    return q && q.mid != null && !q.stale;
  }).length;
  return fresh >= Math.max(1, Math.ceil(heldEq.length / 2)) ? 'live' : 'delayed';
}

/* the account's change since the last completed trading session — grouped
   from the same equity marks the chart already shows, oldest→newest */
function todayChange(series) {
  if (!series || series.length < 2) return null;
  const byDay = new Map();
  for (const p of series) {
    if (p.t && p.equity != null) byDay.set(p.t.slice(0, 10), p.equity);
  }
  const days = [...byDay.keys()].sort();
  if (days.length < 2) return null;
  const latest = byDay.get(days[days.length - 1]);
  const prior = byDay.get(days[days.length - 2]);
  if (!prior) return null;
  return { dollars: latest - prior, pct: (latest - prior) / prior * 100 };
}

/* Populate the sticky hero: value, today's move, P&L / return / cash /
   count, and the two chips (strategy + market mood). Everything the reader
   most wants to know, in one place, above every zone. */
async function loadHeader() {
  const $ = id => document.getElementById(id);
  const setText = (id, txt, cls) => {
    const el = $(id); if (!el) return;
    el.textContent = txt;
    if (cls != null) {
      el.classList.remove('t-up', 't-down');
      if (cls) el.classList.add(cls);
    }
  };
  try {
    const [pf, strat, regime, eq, dataHealth] = await Promise.all([
      apiGet('/api/desk/portfolio'),
      apiGet('/api/desk/strategy'),
      apiGet('/api/desk/regime').catch(() => null),
      apiGet('/api/desk/equity?limit=500').catch(() => null),
      // A failed health check must render as VISIBLY unknown, not vanish —
      // the pill exists to surface exactly the states where fetches fail.
      apiGet('/api/desk/data-health').catch(() => ({ status: 'unknown' })),
    ]);

    setText('desk-hero-account', fmtDollar(pf.equity));

    const pnlCls = pf.total_pnl >= 0 ? 't-up' : 't-down';
    setText('desk-hero-pnl', fmtPnl(pf.total_pnl), pnlCls);
    // *_pct fields are already percent numbers — fmtPct renders as given
    // (the old /100 under-displayed every figure a hundredfold).
    setText('desk-hero-return', fmtPct(pf.total_return_pct, { signed: true }), pnlCls);
    if (pf.vs_spy && pf.vs_spy.alpha_pct != null) {
      const a = pf.vs_spy.alpha_pct;
      setText('desk-hero-alpha', fmtPct(a, { signed: true }),
        a >= 0 ? 't-up' : 't-down');
    } else {
      setText('desk-hero-alpha', '—', '');
    }
    setText('desk-hero-cash', fmtDollar(pf.cash));
    setText('desk-hero-count', String((pf.positions || []).length));

    // Cache the reference book so live tape ticks can fold in and refresh
    // the hero + positions rows between routine runs. Seed lastEquity so the
    // FIRST live tick (frozen → live) also flashes green/red — otherwise the
    // biggest visible jump on load would happen silently.
    deskLive.book = pf;
    if (deskLive.lastEquity == null) deskLive.lastEquity = pf.equity;

    // Today's move — the change since the last completed session
    const todayEl = $('desk-hero-today');
    if (todayEl) {
      clear(todayEl);
      const today = todayChange(eq);
      todayEl.append(h('span', { class: 'lbl', text: 'Today' }));
      if (today) {
        const cls = today.dollars >= 0 ? 't-up' : 't-down';
        todayEl.append(h('span', {
          class: cls,
          title: 'The change in account value since the last completed trading session',
          text: fmtPnl(today.dollars) + ' (' + fmtPct(today.pct) + ')',
        }));
      } else {
        todayEl.append(h('span', {
          class: 'empty',
          title: 'Not enough history yet to show a day-over-day change',
          text: '—',
        }));
      }
    }

    // Chips: strategy + market mood
    const chipsEl = $('desk-hero-chips');
    if (chipsEl) {
      clear(chipsEl);
      if (strat && strat.current) {
        chipsEl.append(h('span', {
          class: 'c-pill info',
          title: strat.current.thesis || 'The strategy the AI is currently running.',
          text: 'Strategy v' + strat.current.version + ' · ' + strat.current.name,
        }));
      }
      if (regime && regime.tag) {
        const MOOD = {
          risk_on: ['Market mood: favorable', 'up',
            'The major indexes are in uptrends — conditions that historically reward being invested.'],
          risk_off: ['Market mood: defensive', 'down',
            'The market is below its long-term trend — the AI leans toward caution and cash.'],
          neutral: ['Market mood: mixed', 'neutral',
            'Trend signals are conflicting — neither clearly favorable nor defensive.'],
        };
        const [label, cls, tip] = MOOD[regime.tag] || MOOD.neutral;
        chipsEl.append(h('span', { class: 'c-pill ' + cls, title: tip, text: label }));
      }
      if (dataHealth && dataHealth.status) {
        const DATA = {
          green: ['Research data: fresh', 'up',
            'The nightly whole-market data refresh is current — stock rankings and research use up-to-date history.'],
          amber: ['Research data: aging', 'warn',
            'Last night’s whole-market data refresh was missed — research rankings are one session behind.'],
          red: ['Research data: stale', 'down',
            'The whole-market data refresh has been down for several sessions — the AI limits itself to managing existing holdings until it recovers.'],
          unknown: ['Research data: unavailable', 'warn',
            'The data-health check itself failed — freshness cannot be verified right now.'],
        };
        const [label, cls, tip] = DATA[dataHealth.status] || DATA.unknown;
        const full = dataHealth.last_full_date
          ? ' (last full refresh: ' + dataHealth.last_full_date + ')' : '';
        chipsEl.append(h('span', { class: 'c-pill ' + cls, title: tip + full, text: label }));
      }
      // Degraded marks: the latest account valuation priced part of the book
      // at COST (no live quote, no stored close) — fake-flat P&L must be
      // visible, not buried in a JSON column.
      if (dataHealth && dataHealth.marks && dataHealth.marks.degraded) {
        const m = dataHealth.marks;
        const who = (m.cost_marked && m.cost_marked.length)
          ? ' Affected: ' + m.cost_marked.join(', ') + '.' : '';
        const pct = m.cost_marked_value_pct != null
          ? m.cost_marked_value_pct + '% of position value' : 'part of the book';
        chipsEl.append(h('span', {
          class: 'c-pill warn',
          title: 'The latest account valuation priced ' + pct + ' at what it '
            + 'PAID — no live quote or stored close was available, so profit/'
            + 'loss on those names reads as flat until the data feed recovers.'
            + who,
          text: 'Marks: partly at cost',
        }));
      }
    }
  } catch (err) {
    // The hero must never take the whole page down — leave placeholders in place
    console.error('desk header load failed', err);
  }
}

/* ── equity curve ── */
function ensureEquityChart() {
  if (equityChart) return;
  const el = document.getElementById('desk-equity-chart');
  equityChart = createChart(el, { height: 320 });
  const c = colors();
  // benchmark first so the account curve draws over it
  spySeries = equityChart.addLineSeries({
    color: c.benchmark, lineWidth: 1, priceLineVisible: false,
    lastValueVisible: false, crosshairMarkerVisible: false,
  });
  equitySeries = equityChart.addAreaSeries({
    lineColor: c.accent, topColor: c.accent + '55', bottomColor: c.accent + '08',
    lineWidth: 2, priceLineVisible: false,
  });
  // The live tip: one clearly-dashed connector from the last RECORDED mark
  // to the current live estimate. Its two points live in their own series —
  // never in the history the chart treats as recorded marks (the old code
  // fabricated ~240 synthetic points/hour into the real curve).
  liveTipSeries = equityChart.addLineSeries({
    color: c.accent, lineWidth: 1, lineStyle: 2 /* dashed */,
    priceLineVisible: false, lastValueVisible: true,
    crosshairMarkerVisible: false,
  });
  onThemeChange(() => {
    const cc = colors();
    equitySeries.applyOptions({ lineColor: cc.accent, topColor: cc.accent + '55', bottomColor: cc.accent + '08' });
    spySeries.applyOptions({ color: cc.benchmark });
    liveTipSeries.applyOptions({ color: cc.accent });
  });
}

async function loadEquity() {
  const metaEl = document.getElementById('desk-equity-meta');
  try {
    const body = await apiGet('/api/desk/equity?limit=2000&with_spy=1');
    const series = body.points || [];
    if (!series.length) {
      metaEl.textContent = 'no marks yet';
      return;
    }
    ensureEquityChart();
    // de-dup identical timestamps (chart requires strictly increasing time)
    const seen = new Set();
    const data = [];
    const degraded = [];
    for (const p of series) {
      const time = toEpochSec(p.t);
      if (!time || seen.has(time)) continue;
      seen.add(time);
      data.push({ time, value: p.equity });
      if (p.degraded) degraded.push(time);
    }
    lastEquityData = data;
    equitySeries.setData(data);
    // Degraded marks (part of the book priced at cost basis) get a visible
    // warn marker on the exact affected points.
    const c = colors();
    equitySeries.setMarkers(degraded.map(time => ({
      time, position: 'aboveBar', shape: 'circle', color: c.warn,
      id: 'degraded:' + time, text: '',
    })));
    // SPY overlay: what $100k in SPY (dividends reinvested) since the
    // account's first trade would be worth — same dollar axis, honest beta.
    const start = (deskLive.book && deskLive.book.starting_capital) || 100000;
    const spyData = (body.spy || [])
      .map(x => ({ time: toEpochSec(x.date),
                   value: Math.round(start * (1 + x.pct / 100) * 100) / 100 }))
      .filter(x => x.time);
    spySeries.setData(spyData);
    const legend = document.getElementById('desk-equity-legend');
    if (legend) legend.hidden = !spyData.length;
    // fresh history → the live tip re-anchors on the new last real mark
    liveTipSeries.setData([]);
    equityChart.timeScale().fitContent();
    const last = series[series.length - 1];
    metaEl.textContent = `${fmtDollar(last.equity)} · ${series.length} marks`
      + (degraded.length ? ` · ${degraded.length} degraded` : '');
    metaEl.title = degraded.length
      ? 'Some snapshots priced part of the book at cost basis during a '
        + 'quote/close outage — those points are flagged on the curve.'
      : '';
  } catch (err) {
    metaEl.textContent = 'error loading curve';
  }
}

/* Redraw the dashed live-estimate tip: last real mark → live equity now.
   Called from applyLiveMarks only while the pill says LIVE; the two points
   are setData()'d into their own series, so history stays untouched. */
function updateLiveTip(liveEquity) {
  if (!liveTipSeries || !lastEquityData.length
      || liveEquity == null || !Number.isFinite(liveEquity)) return;
  const last = lastEquityData[lastEquityData.length - 1];
  const nowSec = Math.max(Math.floor(Date.now() / 1000), last.time + 1);
  try {
    liveTipSeries.setData([
      { time: last.time, value: last.value },
      { time: nowSec, value: Math.round(liveEquity * 100) / 100 },
    ]);
  } catch (e) { /* chart not mounted — skip this fold */ }
}

/* ── holdings (equities + an options book when present) ── */
const OCC_RE = /^([A-Z]{1,6})(\d{2})(\d{2})(\d{2})([CP])(\d{8})$/;

function occParse(sym) {
  const m = OCC_RE.exec(sym);
  if (!m) return null;
  const expiry = new Date(Date.UTC(2000 + +m[2], +m[3] - 1, +m[4]));
  return {
    underlying: m[1], type: m[5], strike: +m[6] / 1000, expiry,
    dte: Math.ceil((expiry - Date.now()) / 86400000),
    label: `${m[1]} $${+m[6] / 1000}${m[5]} ${expiry.toISOString().slice(0, 10)}`,
  };
}

/* allocation donut — how the account is split across holdings + cash.
   SVG arcs (circumference normalized to 100), colored by the token series
   palette via classes — theme-safe, zero inline styles. */
function allocation(pf) {
  const segs = [];
  (pf.positions || []).forEach((p, i) => {
    if (p.weight > 0) segs.push({ label: p.symbol, pct: p.weight * 100, cls: 's' + (i % 8) });
  });
  const cashPct = pf.equity ? Math.max(0, pf.cash / pf.equity * 100) : 0;
  if (cashPct > 0.05) segs.push({ label: 'Cash', pct: cashPct, cls: 'cash' });
  if (!segs.length) return null;
  const R = 15.915494;  // circumference ≈ 100 → dasharray reads as percent
  const ring = svg('svg', {
    class: 'desk-alloc-ring', viewBox: '0 0 42 42', width: '132', height: '132',
    role: 'img', 'aria-label': 'Allocation by holding',
  }, svg('circle', { class: 'desk-alloc-bg', cx: '21', cy: '21', r: R, fill: 'none', 'stroke-width': '5' }));
  let start = 0;
  for (const s of segs) {
    ring.append(svg('circle', {
      class: 'desk-alloc-seg ' + s.cls, cx: '21', cy: '21', r: R, fill: 'none', 'stroke-width': '5',
      'stroke-dasharray': `${s.pct.toFixed(2)} ${(100 - s.pct).toFixed(2)}`,
      'stroke-dashoffset': ((25 - start % 100 + 100) % 100).toFixed(2),
    }));
    start += s.pct;
  }
  const legend = h('div', { class: 'desk-alloc-legend' },
    ...segs.map(s => h('div', { class: 'desk-alloc-item' },
      h('span', { class: 'desk-alloc-swatch ' + s.cls }),
      h('span', { class: 'desk-alloc-label', text: s.label }),
      h('span', { class: 'desk-alloc-pct t-dim', text: fmtNum(s.pct, 1) + '%' }))));
  return h('div', { class: 'desk-alloc' }, ring, legend);
}

function sparkline(series, up) {
  const W = 68, H = 20, n = series ? series.length : 0;
  if (n < 2) return h('span', { class: 't-dim', text: '—' });
  const min = Math.min(...series), max = Math.max(...series), rng = (max - min) || 1;
  const pts = series.map((v, i) =>
    `${(i / (n - 1) * W).toFixed(1)},${(H - (v - min) / rng * H).toFixed(1)}`).join(' ');
  return svg('svg', {
    class: 'desk-spark ' + (up ? 't-up' : 't-down'),
    viewBox: `0 0 ${W} ${H}`, width: W, height: H, preserveAspectRatio: 'none',
    'aria-hidden': 'true',
  }, svg('polyline', { points: pts, fill: 'none', stroke: 'currentColor', 'stroke-width': '1.5', 'stroke-linejoin': 'round' }));
}

function dayChangeCell(st) {
  if (!st || st.day_change_pct == null) return h('td', { class: 'num t-dim', text: '—' });
  const c = st.day_change_pct;
  return h('td', { class: 'num ' + (c >= 0 ? 't-up' : 't-down'),
    text: (c >= 0 ? '+' : '') + fmtNum(c, 2) + '%' });
}

function trendCell(st) {
  if (!st || !(st.spark && st.spark.length > 1)) return h('td', { class: 'num t-dim', text: '—' });
  const up = st.spark[st.spark.length - 1] >= st.spark[0];
  const range = (st.wk52_low != null && st.wk52_high != null)
    ? ` · 52-wk range ${fmtPrice(st.wk52_low)}–${fmtPrice(st.wk52_high)}` : '';
  return h('td', { class: 'num', title: '30-day price trend' + range }, sparkline(st.spark, up));
}

function equitiesTable(rows, stats) {
  stats = stats || {};
  const divs = deskLive.divs || {};
  const divNote = sym => {
    const d = divs[sym];
    if (!d || !d.next_ex_date) return null;
    return h('div', {
      class: 't-dim desk-pos-div',
      title: 'This holding pays a dividend'
        + (d.ttm_amount ? ` — about ${fmtPrice(d.ttm_amount)}/share per year` : '')
        + '. Shown: the next date you must own it by to receive the payment. '
        + 'Full history is on the stock\'s chart page under Events.',
      text: 'next dividend ' + d.next_ex_date,
    });
  };
  return h('table', { class: 'c-table' },
    h('thead', {}, h('tr', {},
      h('th', { text: 'Stock' }), h('th', { class: 'num', text: 'Shares' }),
      h('th', { class: 'num', text: 'Paid', title: 'Average price paid per share' }),
      h('th', { class: 'num', text: 'Now', title: 'Most recent market price' }),
      h('th', { class: 'num', text: 'Today', title: "The stock's move on the last completed trading session" }),
      h('th', { class: 'num', text: '30-day trend', title: 'The shape of the last 30 days; hover for the 52-week range' }),
      h('th', { class: 'num', text: 'Worth' }),
      h('th', { class: 'num', text: '% of account', title: 'How much of the whole account this holding represents' }),
      h('th', { class: 'num', text: 'Gain / loss', title: 'Profit or loss if sold at the current price' }))),
    h('tbody', {}, ...rows.map(p => h('tr', {},
      h('td', {}, h('a', { href: '/symbol/' + p.symbol, class: 'c-link', text: p.symbol }),
        divNote(p.symbol)),
      h('td', { class: 'num', text: fmtNum(p.shares, 2) }),
      h('td', { class: 'num', text: fmtPrice(p.avg_price) }),
      h('td', { class: 'num', text: fmtPrice(p.last_price) }),
      dayChangeCell(stats[p.symbol]),
      trendCell(stats[p.symbol]),
      h('td', { class: 'num', text: fmtDollar(p.market_value) }),
      // weight is a 0-1 fraction — scale to percent for display
      h('td', { class: 'num', text: fmtPct(p.weight * 100, { signed: false }) }),
      h('td', { class: 'num ' + (p.unrealized_pnl >= 0 ? 't-up' : 't-down'),
        text: fmtPnl(p.unrealized_pnl) })))));
}

function optionsTable(rows) {
  return h('table', { class: 'c-table' },
    h('thead', {}, h('tr', {},
      h('th', { text: 'Option contract', title: 'e.g. "NVDA $200C 2027-01-16" = the right to buy NVDA at $200 until Jan 16 2027' }),
      h('th', { text: 'Side', title: 'LONG = the AI bought it. SHORT = the AI sold it (always backed by shares, cash, or another option — never naked)' }),
      h('th', { class: 'num', text: 'Contracts', title: 'How many option contracts; each contract covers 100 shares' }),
      h('th', { class: 'num', text: 'Days left', title: 'Days until the contract expires' }),
      h('th', { class: 'num', text: 'Paid' }), h('th', { class: 'num', text: 'Now' }),
      h('th', { class: 'num', text: 'Worth' }),
      h('th', { class: 'num', text: 'Gain / loss' }))),
    h('tbody', {}, ...rows.map(p => {
      const o = occParse(p.symbol);
      const short = p.shares < 0;
      return h('tr', {},
        h('td', {}, h('a', { href: '/symbol/' + o.underlying, class: 'c-link', text: o.label })),
        h('td', {}, pill(short ? 'SHORT' : 'LONG', short ? 'warn' : 'info')),
        h('td', { class: 'num', text: fmtNum(Math.abs(p.shares), 0) }),
        h('td', { class: 'num ' + (o.dte <= 5 ? 't-down' : ''), text: String(o.dte) }),
        h('td', { class: 'num', text: fmtPrice(p.avg_price) }),
        h('td', { class: 'num', text: fmtPrice(p.last_price) }),
        h('td', { class: 'num', text: fmtDollar(p.market_value) }),
        h('td', { class: 'num ' + (p.unrealized_pnl >= 0 ? 't-up' : 't-down'),
          text: fmtPnl(p.unrealized_pnl) }));
    })));
}

function renderPositions(el, pf, stats) {
  const eqs = pf.positions.filter(p => !occParse(p.symbol));
  const opts = pf.positions.filter(p => occParse(p.symbol));
  clear(el);
  const donut = allocation(pf);
  if (donut) el.append(donut);
  if (eqs.length) el.append(equitiesTable(eqs, stats));
  if (opts.length) {
    el.append(h('div', { class: 'desk-subhead', text: 'Options' }), optionsTable(opts));
  }
}

/* Fold the running live-mid dict (`deskLive.marks`) onto the cached
   reference book (`deskLive.book`) and repaint the hero + positions rows.
   Called on every SSE tick — options fall back to their last mark (OPRA
   isn't on the equity SIP stream), so an all-options book is still frozen
   between routine runs; the improvement is for the ~90% of the account
   that's equities. */
function applyLiveMarks(pillState) {
  const ref = deskLive.book;
  if (!ref || !ref.positions) return;
  const marks = deskLive.marks;
  const positions = [];
  let posValue = 0;
  for (const p of ref.positions) {
    const isOpt = !!occParse(p.symbol);
    const live = !isOpt ? marks[p.symbol] : null;
    const price = (live != null && Number.isFinite(live)) ? live : p.last_price;
    const mult = isOpt ? 100 : 1;
    const mv = Math.round(p.shares * price * mult * 100) / 100;
    posValue += mv;
    positions.push({
      ...p,
      last_price: Math.round(price * 10000) / 10000,
      market_value: mv,
      unrealized_pnl: Math.round(p.shares * (price - p.avg_price) * mult * 100) / 100,
    });
  }
  const equity = Math.round((ref.cash + posValue) * 100) / 100;
  const start = ref.starting_capital || 100000;
  const totalPnl = Math.round((equity - start) * 100) / 100;
  const returnPct = Math.round(((equity - start) / start) * 10000) / 100;
  for (const r of positions) r.weight = equity ? Math.round((r.market_value / equity) * 10000) / 10000 : 0;
  positions.sort((a, b) => b.market_value - a.market_value);

  // Hero cards
  const setText = (id, txt, cls) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.textContent = txt;
    el.classList.remove('t-up', 't-down');
    if (cls) el.classList.add(cls);
  };

  // Account value with a tick-direction flash — makes tiny $$ changes read
  // visibly. Force reflow so the class replay actually re-triggers the CSS
  // animation on consecutive ticks in the same direction.
  const acctEl = document.getElementById('desk-hero-account');
  if (acctEl) {
    const prevEq = deskLive.lastEquity;
    acctEl.textContent = fmtDollar(equity);
    acctEl.classList.remove('desk-tick-up', 'desk-tick-down');
    if (prevEq != null && equity !== prevEq) {
      void acctEl.offsetWidth;
      acctEl.classList.add(equity > prevEq ? 'desk-tick-up' : 'desk-tick-down');
    }
  }
  deskLive.lastEquity = equity;

  const pnlCls = totalPnl >= 0 ? 't-up' : 't-down';
  setText('desk-hero-pnl', fmtPnl(totalPnl), pnlCls);
  setText('desk-hero-return', fmtPct(returnPct, { signed: true }), pnlCls);
  // Keep 'vs S&P 500' consistent with the live Return beside it — the SPY
  // side is daily-close based and static between page loads, so live alpha
  // is just live return minus the cached SPY return.
  const spy = ref && ref.vs_spy ? ref.vs_spy.spy_return_pct : null;
  if (spy != null) {
    const a = returnPct - spy;
    setText('desk-hero-alpha', fmtPct(a, { signed: true }),
      a >= 0 ? 't-up' : 't-down');
  }
  // cash and count don't change from live marks — leave them alone.

  // The pill tells the truth: LIVE only for a fresh tape in an open session
  // (see livePillState); stale → DELAYED, closed market → CLOSED.
  setLivePill(pillState);
  deskLive.lastTickTs = Date.now();
  const ageEl = document.getElementById('desk-hero-live-age');
  if (ageEl) ageEl.textContent = 'just now';

  // Redraw the dashed live-estimate tip — only while genuinely live; a
  // stale or closed tape must not draw a "current" estimate.
  if (pillState === 'live') updateLiveTip(equity);

  // Positions tables: repaint only if the container already has content
  // (first load hasn't finished yet → skeleton lives; leave it).
  const posEl = document.getElementById('desk-positions');
  if (posEl && !posEl.querySelector('.c-skel') && !posEl.querySelector('.c-empty')) {
    renderPositions(posEl, {
      ...ref, positions, positions_value: posValue,
      equity, total_pnl: totalPnl, total_return_pct: returnPct,
    }, deskLive.stats);
  }
}

async function loadPositions() {
  const el = document.getElementById('desk-positions');
  skeleton(el);
  try {
    const [pf, hs, dv] = await Promise.all([
      apiGet('/api/desk/portfolio'),
      apiGet('/api/desk/holding-stats').catch(() => null),
      apiGet('/api/desk/dividends').catch(() => null),
    ]);
    if (!pf.positions.length) { renderEmpty(el, 'All cash — no open positions.'); return; }
    // Cache stats + book so tape ticks can repaint with the same holding-stats
    // shape (day-change chip, 30-day trend) without another network round trip.
    deskLive.stats = (hs && hs.symbols) || {};
    // Dividend facts fold into the holdings rows (the standalone calendar
    // card retired in v9.5.0 — full history is on each chart page).
    deskLive.divs = {};
    for (const x of (dv && dv.holdings) || []) {
      if (x.has_dividend) deskLive.divs[x.symbol] = x;
    }
    deskLive.book = pf;
    renderPositions(el, pf, deskLive.stats);
  } catch (err) { renderError(el, err, loadPositions); }
}

/* ── live thinking feed ── */
async function loadThinking() {
  const el = document.getElementById('desk-thinking');
  const runEl = document.getElementById('desk-thinking-run');
  skeleton(el);
  try {
    const data = await apiGet('/api/desk/thinking?limit=80');
    if (!data.lines.length) { renderEmpty(el, 'No thinking recorded yet.'); runEl.textContent = ''; return; }
    runEl.textContent = data.run_id ? ('run ' + data.run_id) : '';
    clear(el);
    // Show the freshest handful; the full transcript is one click away —
    // the feed is the single longest block on the page when left uncapped.
    const VISIBLE = 6;
    const feed = h('div', { class: 'desk-feed' });
    data.lines.forEach((line, i) => {
      const row = h('div', { class: 'desk-feed-line' },
        h('span', { class: 'desk-feed-phase', text: line.phase || '·' }),
        h('span', { class: 'desk-feed-text', text: line.text }),
        h('span', { class: 'desk-feed-time t-dim', text: timeAgo(line.t) }));
      if (i >= VISIBLE) row.hidden = true;
      feed.append(row);
    });
    el.append(feed);
    if (data.lines.length > VISIBLE) {
      const btn = h('button', {
        class: 'desk-morebtn', type: 'button',
        text: 'Show all ' + data.lines.length + ' lines',
      });
      btn.addEventListener('click', () => {
        const hiddenNow = feed.querySelector('[hidden]') != null;
        feed.querySelectorAll('.desk-feed-line').forEach(r => { r.hidden = false; });
        if (!hiddenNow) {
          feed.querySelectorAll('.desk-feed-line').forEach((r, i) => {
            if (i >= VISIBLE) r.hidden = true;
          });
        }
        btn.textContent = hiddenNow
          ? 'Show only the latest'
          : 'Show all ' + data.lines.length + ' lines';
      });
      el.append(btn);
    }
  } catch (err) { renderError(el, err, loadThinking); }
}

/* ── what the AI is watching: tripwires + planned check-ins ── */
async function loadWatch() {
  const el = document.getElementById('desk-watch');
  const metaEl = document.getElementById('desk-watch-meta');
  if (!el) return;
  skeleton(el);
  try {
    const d = await apiGet('/api/desk/watch');
    const wires = (d.watches || []).filter(w => w.status === 'armed' || w.status === 'tripped');
    const now = Date.now();
    const wakes = (d.wakes || []).filter(k => !k.honored_run_id
      && k.at && (new Date(k.at).getTime() > now - 6 * 3600e3));
    clear(el);
    if (metaEl) {
      metaEl.textContent = wires.length
        ? wires.length + ' alarm' + (wires.length === 1 ? '' : 's') + ' set' : '';
    }
    if (!wires.length && !wakes.length) {
      renderEmpty(el, 'No price alarms set right now — the AI arms them on positions it needs to react to.');
      return;
    }
    const list = h('div', { class: 'desk-watch-list' });
    for (const w of wires) {
      const dir = w.kind === 'below' ? 'falls under' : 'climbs past';
      const tripped = w.status === 'tripped';
      list.append(h('div', { class: 'desk-watch-row' },
        pill(tripped ? 'TRIPPED' : 'watching', tripped ? 'down' : 'info'),
        h('span', { class: 'desk-watch-text', text:
          w.symbol + ' — alert if the price ' + dir + ' $' + fmtNum(w.level, 2)
          + (w.reason ? ' · ' + w.reason : '') }),
        h('span', { class: 'desk-feed-time t-dim',
          text: tripped && w.tripped_at ? 'tripped ' + timeAgo(w.tripped_at)
            : 'set ' + timeAgo(w.armed_at) })));
    }
    for (const k of wakes) {
      list.append(h('div', { class: 'desk-watch-row' },
        pill('next check', 'neutral'),
        h('span', { class: 'desk-watch-text', text:
          'Wants to look again ' + fmtDateTimeET(k.at)
          + (k.reason ? ' — ' + k.reason : '') })));
    }
    el.append(list);
  } catch (err) { renderError(el, err, loadWatch); }
}

/* ── decisions: the latest dossier + the browsable archive ──
   pickCard renders one pick everywhere (latest view + history dossiers),
   INCLUDING the prediction registry — the prediction / horizon / kill the
   agent committed to at buy time (previously recorded but never shown). */
function pickCard(p) {
  const action = (p.action || '').toLowerCase();
  const card = h('div', { class: 'desk-pick c-card' },
    h('div', { class: 'desk-pick-head' },
      h('a', { href: '/symbol/' + p.symbol, class: 'desk-pick-sym', text: p.symbol }),
      pill((p.action || '').toUpperCase() || '—', ACTION_CLASS[action] || 'neutral'),
      p.why_now ? h('span', { class: 'desk-pick-why t-dim', text: p.why_now }) : null),
    p.rationale ? h('p', { class: 'desk-pick-rationale', text: p.rationale }) : null);
  const facts = [];
  if (p.prediction) facts.push(['predicts', p.prediction]);
  if (p.horizon_days != null) facts.push(['horizon', p.horizon_days + ' sessions']);
  if (p.kill) facts.push(['abandon if', String(p.kill)]);
  if (facts.length) {
    card.append(h('div', { class: 'desk-pick-evidence c-chips' },
      ...facts.map(([k, v]) => h('span', {
        class: 'c-chip',
        title: 'The commitment made at decision time — graded later in “Predictions vs outcomes”.',
        text: `${k}: ${v}`,
      }))));
  }
  if (p.evidence && Object.keys(p.evidence).length) {
    const kv = h('div', { class: 'desk-pick-evidence c-chips' });
    for (const [k, v] of Object.entries(p.evidence)) {
      kv.append(h('span', { class: 'c-chip', text: `${k}: ${v}` }));
    }
    card.append(kv);
  }
  if (p.news && p.news.length) {
    const news = h('ul', { class: 'desk-pick-news' });
    for (const n of p.news.slice(0, 3)) {
      const title = typeof n === 'string' ? n : (n.title || '');
      const url = typeof n === 'object' ? n.url : null;
      news.append(h('li', {}, url
        ? h('a', { href: url, class: 'c-link', target: '_blank', rel: 'noopener', text: title })
        : h('span', { text: title })));
    }
    card.append(news);
  }
  return card;
}

function watchlistChips(watchlist) {
  return h('div', { class: 'c-chips' },
    h('span', { class: 't-dim', text: 'Watchlist: ' }),
    ...watchlist.map(w => h('span', { class: 'c-chip' },
      h('a', { href: '/symbol/' + (w.symbol || w), class: 'c-link', text: (w.symbol || w) }),
      w.note ? h('span', { class: 't-dim', text: ' — ' + w.note }) : null)));
}

async function loadDecision() {
  const picksEl = document.getElementById('desk-picks');
  const sumEl = document.getElementById('desk-summary');
  const whenEl = document.getElementById('desk-decision-when');
  const wlEl = document.getElementById('desk-watchlist');
  skeleton(picksEl);
  try {
    const d = await apiGet('/api/desk/decision/latest');
    if (!d.exists) { clear(sumEl); renderEmpty(picksEl, 'No decision recorded yet.'); clear(wlEl); whenEl.textContent = ''; return; }
    whenEl.textContent = d.ts ? fmtDateTimeET(d.ts) : '';
    clear(sumEl);
    sumEl.append(h('p', { class: 'desk-summary', text: d.summary || '' }));

    clear(picksEl);
    if (!(d.picks && d.picks.length)) {
      renderEmpty(picksEl, 'No per-name picks in this decision.');
    } else {
      for (const p of d.picks) picksEl.append(pickCard(p));
    }

    clear(wlEl);
    if (d.watchlist && d.watchlist.length) {
      wlEl.append(watchlistChips(d.watchlist));
    }
  } catch (err) { renderError(picksEl, err, loadDecision); }
}

/* ── decision archive: every past dossier, compact rows → expandable ── */
let decisionView = 'latest';
const decHist = { rows: [], nextBefore: null, loading: false };

const REGIME_PILL = { risk_on: 'up', risk_off: 'down', neutral: 'neutral' };

function decisionDossier(d) {
  const box = h('div', { class: 'desk-dec-hist-body' });
  if (d.summary) box.append(h('p', { class: 'desk-summary', text: d.summary }));
  for (const p of (d.picks || [])) box.append(pickCard(p));
  if (d.watchlist && d.watchlist.length) box.append(watchlistChips(d.watchlist));
  if (d.rejected && d.rejected.length) {
    box.append(
      h('div', { class: 'desk-subhead', text: 'Passed on' }),
      h('ul', { class: 'desk-dec-rej' }, ...d.rejected.map(r => {
        const sym = (r && r.symbol) || String(r);
        return h('li', {},
          h('a', { href: '/symbol/' + sym, class: 'c-link', text: sym }),
          r && r.why_not ? ' — ' + r.why_not : '');
      })));
  }
  return box;
}

function renderDecisionHistory() {
  const el = document.getElementById('desk-decision-history');
  if (!el) return;
  clear(el);
  if (!decHist.rows.length) {
    renderEmpty(el, 'No past decisions recorded yet.');
    return;
  }
  for (const d of decHist.rows) {
    const picks = d.picks || [];
    const btn = h('button', { class: 'desk-morebtn', type: 'button', text: 'Details' });
    const row = h('div', { class: 'desk-dec-hist-row' },
      h('div', { class: 'desk-dec-hist-head' },
        h('span', { class: 'desk-dec-hist-when', text: fmtDateTimeET(d.ts) }),
        d.regime ? pill(d.regime.replace(/_/g, ' '), REGIME_PILL[d.regime] || 'neutral') : null,
        h('span', { class: 't-dim', text: picks.length + ' pick' + (picks.length === 1 ? '' : 's') }),
        h('span', { class: 'spacer' }),
        btn));
    const first = String(d.summary || '').split('\n')[0];
    if (first) {
      row.append(h('p', { class: 'desk-dec-hist-sum',
        text: first.length > 180 ? first.slice(0, 180) + '…' : first }));
    }
    let body = null;
    btn.addEventListener('click', () => {
      if (body) {
        body.hidden = !body.hidden;
        btn.textContent = body.hidden ? 'Details' : 'Hide';
        return;
      }
      body = decisionDossier(d);
      row.append(body);
      btn.textContent = 'Hide';
    });
    el.append(row);
  }
  if (decHist.nextBefore != null) {
    const more = h('button', { class: 'desk-morebtn', type: 'button', text: 'Load older decisions' });
    more.addEventListener('click', () => loadDecisionHistory(decHist.nextBefore));
    el.append(more);
  }
}

async function loadDecisionHistory(before) {
  const el = document.getElementById('desk-decision-history');
  if (!el || decHist.loading) return;
  decHist.loading = true;
  if (before == null) { decHist.rows = []; skeleton(el); }
  try {
    const d = await apiGet('/api/desk/decisions?limit=10'
      + (before != null ? '&before=' + encodeURIComponent(before) : ''));
    decHist.rows = decHist.rows.concat(d.decisions || []);
    decHist.nextBefore = d.next_before;
    renderDecisionHistory();
  } catch (err) {
    renderError(el, err, () => loadDecisionHistory(before));
  } finally {
    decHist.loading = false;
  }
}

function applyDecisionView() {
  const hist = document.getElementById('desk-decision-history');
  const showHist = decisionView === 'history';
  for (const id of ['desk-summary', 'desk-picks', 'desk-watchlist']) {
    const el = document.getElementById(id);
    if (el) el.hidden = showHist;
  }
  if (hist) hist.hidden = !showHist;
  if (showHist && !decHist.rows.length) loadDecisionHistory();
}

/* ── predictions vs outcomes: the scoreboard (/api/desk/outcomes) ──
   Open picks first with a horizon countdown and the kill level against the
   live price; closed picks with how they exited; verdict chips once the
   weekly review has judged them. */
const EXIT_KIND_LABEL = {
  same_run: ['closed same cycle', 'neutral'],
  cross_run: ['closed by a later cycle', 'neutral'],
  hardstop: ['stop-loss exit', 'down'],
  settlement: ['expired / settled', 'neutral'],
};
const VERDICT_CLASS = { TRUE: 'up', FALSE: 'down', NOT_YET: 'neutral' };

function outcomeRow(r) {
  const sym = r.is_option ? (r.symbol.match(/^[A-Z]+/) || [r.symbol])[0] : r.symbol;
  const open = r.status === 'open';
  const [exitLabel, exitCls] = EXIT_KIND_LABEL[r.exit_kind] || ['closed', 'neutral'];
  const head = h('div', { class: 'desk-outcome-head' },
    h('a', { href: '/symbol/' + sym, class: 'desk-outcome-sym', text: r.symbol }),
    open ? pill('OPEN', 'info') : pill(exitLabel, exitCls),
    r.verdict ? pill(r.verdict.replace(/_/g, ' '), VERDICT_CLASS[r.verdict] || 'neutral',
      r.verdict_note || 'The weekly review’s judgment of this prediction.') : null,
    r.degraded ? pill('marks degraded', 'warn',
      'This pick’s symbol was priced at cost basis in the latest valuation — '
      + 'its performance numbers are withheld until real marks return.') : null,
    h('span', { class: 'spacer' }),
    h('span', { class: 't-dim', title: 'decision run ' + (r.run_id || ''),
      text: r.decision_ts ? timeAgo(r.decision_ts) : '' }));

  const pred = h('p', { class: 'desk-outcome-pred',
    text: r.prediction ? '“' + r.prediction + '”'
      : 'No prediction was recorded with this pick.' });

  const chips = [];
  const num = (v, suffix) => h('span', {
    class: 'num ' + (v >= 0 ? 't-up' : 't-down'),
    text: fmtPct(v) + (suffix || ''),
  });
  if (r.since_pct != null) {
    chips.push(h('span', { class: 'c-chip' },
      (open ? 'since entry: ' : 'result: '), num(r.since_pct)));
  }
  if (r.alpha_pct != null) {
    chips.push(h('span', {
      class: 'c-chip',
      title: 'The move minus SPY’s move over the same window (dividends included) — skill, not market tide.',
    }, 'vs S&P 500: ', num(r.alpha_pct)));
  }
  if (r.realized_pnl != null && !open) {
    chips.push(h('span', { class: 'c-chip' }, 'booked: ',
      h('span', { class: 'num ' + (r.realized_pnl >= 0 ? 't-up' : 't-down'),
        text: fmtPnl(r.realized_pnl) })));
  }
  if (r.horizon_days != null) {
    const done = r.sessions_elapsed != null
      ? Math.min(r.sessions_elapsed, r.horizon_days) : null;
    const reached = r.horizon_elapsed === true
      || (done != null && done >= r.horizon_days);
    chips.push(h('span', {
      class: 'c-chip',
      title: 'Trading sessions elapsed against the prediction’s own deadline.',
      text: reached ? 'horizon reached (' + r.horizon_days + ' sessions)'
        : (done != null ? 'session ' + done + ' of ' + r.horizon_days
          : r.horizon_days + '-session horizon'),
    }));
  }
  if (open && r.kill_level != null) {
    const live = deskLive.marks[r.symbol];
    const now = (live != null && Number.isFinite(live)) ? live : r.mark_px;
    chips.push(h('span', {
      class: 'c-chip' + (r.kill_breached ? ' t-down' : ''),
      title: r.kill ? 'The stated abandon condition: ' + r.kill : 'The stated abandon level.',
      text: 'kill ' + fmtPrice(r.kill_level)
        + (now != null ? ' · now ' + fmtPrice(now) : '')
        + (r.kill_breached ? ' — BREACHED' : ''),
    }));
  } else if (open && r.kill) {
    chips.push(h('span', { class: 'c-chip', text: 'abandon if: ' + r.kill }));
  }

  const rowEl = h('div', { class: 'desk-outcome' }, head, pred);
  if (chips.length) rowEl.append(h('div', { class: 'desk-outcome-facts c-chips' }, ...chips));
  if (r.verdict_note) {
    rowEl.append(h('p', { class: 'desk-outcome-note t-dim', text: 'Review: ' + r.verdict_note }));
  }
  return rowEl;
}

async function loadPredictions() {
  const el = document.getElementById('desk-predictions');
  const metaEl = document.getElementById('desk-predictions-meta');
  if (!el) return;
  skeleton(el);
  try {
    const d = await apiGet('/api/desk/outcomes?limit=60');
    const rows = d.rows || [];
    const s = d.summary || {};
    if (metaEl) {
      const bits = [];
      if (s.open) bits.push(s.open + ' open');
      if (s.closed) bits.push(s.closed + ' closed');
      if (s.hit_rate_pct != null) bits.push(s.hit_rate_pct + '% hit rate');
      metaEl.textContent = bits.join(' · ');
    }
    clear(el);
    if (!rows.length) {
      renderEmpty(el, 'No graded predictions yet — every buy records one, and the grader scores them from real prices.');
      return;
    }
    if (s.closed_graded) {
      el.append(h('p', { class: 'desk-lab-honesty t-dim', text:
        'Every buy commits to a written prediction, a deadline, and an abandon '
        + 'level. A grader scores them from recorded prices — '
        + s.closed_graded + ' closed prediction' + (s.closed_graded === 1 ? '' : 's')
        + ' judged so far, ' + (s.hit_rate_pct != null ? s.hit_rate_pct + '% came true.' : '') }));
    }
    const opens = rows.filter(r => r.status === 'open');
    const closed = rows.filter(r => r.status !== 'open');
    if (opens.length) {
      el.append(h('div', { class: 'desk-outcome-subhead', text: 'Open' }));
      for (const r of opens) el.append(outcomeRow(r));
    }
    if (closed.length) {
      el.append(h('div', { class: 'desk-outcome-subhead', text: 'Closed' }));
      for (const r of closed) el.append(outcomeRow(r));
    }
  } catch (err) { renderError(el, err, loadPredictions); }
}

/* ── backtest evidence ── */
/* ── Strategy Lab: tonight's board + recent tests, told in sentences.
   One card, two views ("Tonight's board" / "Recent tests" seg in the card
   header). The raw rule shorthand lives in tooltips only. ── */
const LAB_RULE_NAMES = {
  momentum: 'Pure momentum',
  momo_trend: 'Momentum, uptrends only',
  meanrev: 'Buy the dip (uptrends only)',
  breakout: '52-week-high breakouts',
  regime_momentum: 'Momentum with a market-crash switch',
  equal_weight: 'Own everything equally',
  buyhold: 'Buy and hold',
  trend: 'Ride the trend',
  value_momentum: 'Momentum, profitable & fairly priced only',
};

function labRuleName(rule) {
  const [fam, k] = String(rule || '').split(':');
  const base = LAB_RULE_NAMES[fam] || rule;
  return k ? base + ' (top ' + k + ')' : base;
}

function labUniverseText(u) {
  const s = String(u || '');
  if (s === 'mid200') return 'mid-sized companies (market ranks 41\u2013240 by trading volume)';
  if (s.startsWith('top')) return 'the ' + s.slice(3) + ' most-traded stocks';
  return s;
}

function labHowItPicks(rule, universe, schedule) {
  const [fam, kRaw] = String(rule || '').split(':');
  const k = kRaw || 'a few';
  const uni = labUniverseText(universe);
  const rhythm = schedule === 'weekly' ? 'once a week' : 'once a month';
  const HOW = {
    momentum: `re-picks the ${k} strongest recent risers among ${uni}, ${rhythm}`,
    momo_trend: `re-picks the ${k} strongest risers still in long-term uptrends among ${uni}, ${rhythm}`,
    meanrev: `buys the ${k} most beaten-down names that are still in long-term uptrends among ${uni}, ${rhythm}`,
    breakout: `buys the ${k} names pushing closest to fresh 52-week highs among ${uni}, ${rhythm}`,
    regime_momentum: `rides the ${k} strongest risers among ${uni}, and moves fully to cash whenever the whole market falls below its long-term trend`,
    value_momentum: `rides the ${k} strongest risers among ${uni} \u2014 but only companies that are profitable and not expensive next to their peers, judged by their own SEC filings`,
    equal_weight: `owns ${uni} in equal slices, rebalanced ${rhythm}`,
  };
  return HOW[fam] || `follows the rule \u201c${rule}\u201d on ${uni}, ${rhythm}`;
}

let labView = 'board';
const labCache = { board: null, tests: null };

function renderLabBoard(el, d) {
  clear(el);
  if (!d || !d.combos_tested) {
    renderEmpty(el, 'First nightly sweep pending \u2014 the lab runs after each market close.');
    return;
  }
  // The honesty line comes FIRST \u2014 winners only mean something next to
  // the number of attempts.
  el.append(h('p', { class: 'desk-lab-honesty t-dim', text:
    d.combos_tested + ' strategy variations tested over the last two weeks \u2014 '
    + d.qualified + ' qualified (beat the S&P 500 in BOTH halves of history). '
    + 'Scores show each strategy\u2019s WORSE half; expect live results to shrink.' }));
  if (!d.top || !d.top.length) {
    el.append(h('p', { class: 'desk-lab-honesty', text:
      'Nothing currently qualifies \u2014 an honest filter says no most nights.' }));
    return;
  }
  for (const e of d.top) {
    el.append(h('div', { class: 'desk-lab-entry' },
      h('div', { class: 'desk-lab-entry-name',
        title: 'Lab shorthand: ' + e.rule + ' on ' + e.universe + ', ' + e.schedule,
        text: labRuleName(e.rule) }),
      h('p', { class: 'desk-lab-entry-body' },
        'It ' + labHowItPicks(e.rule, e.universe, e.schedule)
          + '. Even in its weaker half of 21 years it beat the market by ',
        h('span', { class: 'num ' + (e.score >= 0 ? 't-up' : 't-down'),
          text: (e.score >= 0 ? '+' : '') + fmtNum(e.score, 1) + '%' }),
        e.max_dd_out != null
          ? ` \u2014 though the ride included a ${fmtNum(Math.abs(e.max_dd_out), 0)}% drop at its worst.`
          : '.')));
  }
}

function renderLabTests(el, rows) {
  clear(el);
  if (!rows || !rows.length) { renderEmpty(el, 'No history tests run yet.'); return; }
  el.append(h('p', { class: 'desk-lab-honesty t-dim', text:
    'Before risking (paper) money on an idea, the AI asks how it would have '
    + 'done in past markets, after trading costs, versus simply buying the '
    + 'S&P 500. Failed ideas stay on this list on purpose \u2014 they are the point.' }));
  for (const r of rows) {
    const res = r.result || {};
    const ex = res.excess_return_pct;
    const body = h('p', { class: 'desk-lab-entry-body' }, 'Tested ' + timeAgo(r.t) + '. ');
    if (res.return_pct != null) {
      body.append(`It would have returned ${fmtNum(res.return_pct, 1)}% over the test window`);
      if (ex != null) {
        body.append(' \u2014 ',
          h('span', { class: 'num ' + (ex >= 0 ? 't-up' : 't-down'),
            text: (ex >= 0 ? fmtNum(ex, 1) + ' points ahead of'
                           : fmtNum(Math.abs(ex), 1) + ' points behind') }),
          ' simply buying the S&P 500');
      }
      body.append(res.max_drawdown_pct != null
        ? `, with a worst dip of ${fmtNum(Math.abs(res.max_drawdown_pct), 0)}% along the way.`
        : '.');
    }
    el.append(h('div', { class: 'desk-lab-entry' },
      h('div', { class: 'desk-lab-entry-name', text: r.label }), body));
  }
}

function renderLab() {
  const el = document.getElementById('desk-lab');
  if (!el) return;
  if (labView === 'tests') renderLabTests(el, labCache.tests);
  else renderLabBoard(el, labCache.board);
}

async function loadLab() {
  const el = document.getElementById('desk-lab');
  if (!el) return;
  skeleton(el);
  try {
    const [board, tests] = await Promise.all([
      apiGet('/api/desk/lab').catch(() => null),
      apiGet('/api/desk/backtests?limit=12').catch(() => []),
    ]);
    labCache.board = board;
    labCache.tests = tests;
    renderLab();
  } catch (err) { renderError(el, err, loadLab); }
}

/* ── The AI's notebook: lessons (wiki) + diary (strategy journal).
   One card, two views. Lessons are curated pages rewritten from measured
   results; the diary is every approach change, in order, in plain words. ── */
const WIKI_TITLES = {
  playbook: 'Playbook', setups: 'Setups', lessons: 'Lessons',
  mistakes: 'Mistakes', postmortems: 'Postmortems',
  'market-notes': 'Market notes',
};
const REGIME_TAGS = {
  risk_on: 'learned in a rising market',
  risk_off: 'learned in a falling market',
  neutral: 'learned in a mixed market',
};

function wikiBlocks(body) {
  // markdown-lite: blank-line-separated blocks; "- " blocks \u2192 bullet lists,
  // everything else \u2192 paragraphs. All text nodes \u2014 zero innerHTML.
  // A [risk_on]-style tag anywhere in a bullet becomes a small plain-English
  // pill instead of raw shorthand.
  const out = [];
  for (const block of String(body || '').split(/\n\s*\n/)) {
    const lines = block.split('\n').map(l => l.trim()).filter(Boolean);
    if (!lines.length) continue;
    if (lines.every(l => l.startsWith('- '))) {
      out.push(h('ul', { class: 'desk-wiki-list' },
        ...lines.map(l => {
          const item = h('li', {});
          let text = l.slice(2);
          const m = text.match(/\s*\[(risk_on|risk_off|neutral)\]\s*/);
          if (m) {
            text = (text.slice(0, m.index) + ' '
              + text.slice(m.index + m[0].length)).trim();
            item.append(text, ' ',
              h('span', { class: 'c-pill neutral desk-wiki-tag', text: REGIME_TAGS[m[1]] }));
          } else {
            item.append(text);
          }
          return item;
        })));
    } else {
      out.push(h('p', { class: 'desk-wiki-p', text: lines.join(' ') }));
    }
  }
  return out;
}

let wikiView = 'lessons';
const wikiCache = { pages: null, journal: null };

const DIARY_KIND = {
  pivot: 'changed its approach',
  tweak: 'made a small adjustment',
  note: 'made a note',
};

function renderNotebook() {
  const el = document.getElementById('desk-wiki');
  const metaEl = document.getElementById('desk-wiki-meta');
  if (!el) return;
  clear(el);
  if (wikiView === 'diary') {
    const journal = wikiCache.journal || [];
    if (metaEl) metaEl.textContent = journal.length ? journal.length + ' entries' : '';
    if (!journal.length) {
      renderEmpty(el, 'No diary entries yet \u2014 the AI writes one every time it changes its approach.');
      return;
    }
    for (const j of journal) {
      el.append(h('div', { class: 'desk-diary-entry' },
        h('div', { class: 'desk-diary-when t-dim' },
          h('span', { text: timeAgo(j.t) + ' \u2014 the AI ' }),
          h('span', { class: 'desk-diary-kind' + (j.kind === 'pivot' ? ' pivot' : ''),
            text: DIARY_KIND[j.kind] || DIARY_KIND.note })),
        h('div', { class: 'desk-diary-title', text: j.title }),
        j.body ? h('p', { class: 'desk-diary-body t-dim', text: j.body }) : null));
    }
    return;
  }
  const pages = wikiCache.pages || [];
  if (metaEl) metaEl.textContent = pages.length ? pages.length + ' page(s)' : '';
  if (!pages.length) {
    renderEmpty(el, 'The notebook is empty \u2014 lessons appear once real results come in.');
    return;
  }
  for (const p of pages) {
    el.append(h('div', { class: 'desk-wiki-page' },
      h('div', { class: 'desk-wiki-head' },
        h('span', { class: 'desk-wiki-title',
          text: p.title || WIKI_TITLES[p.slug] || p.slug }),
        h('span', { class: 't-dim',
          title: 'Rewritten ' + p.revision + ' time(s) as real results came in',
          text: timeAgo(p.updated_at) })),
      ...wikiBlocks(p.body)));
  }
}

async function loadWiki() {
  const el = document.getElementById('desk-wiki');
  if (!el) return;
  skeleton(el);
  try {
    const [w, s] = await Promise.all([
      apiGet('/api/desk/wiki').catch(() => null),
      apiGet('/api/desk/strategy').catch(() => null),
    ]);
    wikiCache.pages = (w && w.pages) || [];
    wikiCache.journal = (s && s.journal) || [];
    renderNotebook();
  } catch (err) { renderError(el, err, loadWiki); }
}

/* ── live stream consumer: real-time SIP quotes over SSE.
   No tape card anymore (v9.5.0) — ticks feed the hero (account value,
   index chips) and the holdings rows. Full quote detail lives on each
   symbol's chart page. ── */
let tapePrev = {};        // symbol -> last mid (for up/down tick coloring)
let tapeLastEvent = 0;    // client-side staleness watchdog
const INDEX_CHIPS = ['SPY', 'QQQ', 'IWM'];

function renderTape(snap) {
  const quotes = snap.quotes || {};
  const syms = Object.keys(quotes);
  if (!syms.length) return;

  // Live index chips in the hero: the market's pulse without a card.
  const chipsEl = document.getElementById('desk-hero-indices');
  if (chipsEl) {
    clear(chipsEl);
    for (const s of INDEX_CHIPS) {
      const q = quotes[s];
      if (!q || q.mid == null) continue;
      const dir = tapePrev[s] != null
        ? (q.mid > tapePrev[s] ? 't-up' : q.mid < tapePrev[s] ? 't-down' : '') : '';
      chipsEl.append(h('span', { class: 'desk-idx-chip' + (q.stale ? ' stale' : '') },
        h('a', { href: '/symbol/' + s, class: 'desk-idx-sym', text: s }),
        h('span', { class: 'num ' + dir, text: fmtPrice(q.mid) })));
    }
  }
  for (const s of syms) {
    const q = quotes[s];
    if (q.mid != null) tapePrev[s] = q.mid;
    // Feed the live-marks fold: fresh quote → update mark; stale → drop so
    // applyLiveMarks falls back to the last recorded price for that symbol.
    if (q.mid != null && !q.stale) deskLive.marks[s] = q.mid;
    else if (q.stale) delete deskLive.marks[s];
  }
  // Fold the live mids we just captured into hero + positions rows, with an
  // honest pill verdict for this frame (freshness × session).
  applyLiveMarks(livePillState(snap));
}

function startTape() {
  let es;
  const connect = () => {
    es = new EventSource('/api/desk/stream');
    es.addEventListener('quotes', ev => {
      tapeLastEvent = Date.now();
      try { renderTape(JSON.parse(ev.data)); } catch (e) { /* skip bad frame */ }
    });
    es.onerror = () => { /* EventSource auto-reconnects; watchdog surfaces it */ };
  };
  connect();
  // client-side watchdog: when the stream itself goes quiet the pill flips
  // to DELAYED (if it was showing) so frozen numbers never masquerade as
  // ticking ones.
  setInterval(() => {
    if (tapeLastEvent && Date.now() - tapeLastEvent > 6000) {
      const liveEl = document.getElementById('desk-hero-live');
      if (liveEl && !liveEl.hidden) setLivePill('delayed');
    }
  }, 3000);
}

/* ── recent fills: each trade + the live bid/ask it priced off ── */
const OCC_RE_F = /^[A-Z]{1,6}\d{6}[CP]\d{8}$/;

/* Rationale cell: truncated with an inline more/less toggle — the full
   reasoning is readable in place, not trapped in a hover title. */
function fillWhyCell(rationale) {
  if (!rationale) return h('td', { class: 't-dim', text: '—' });
  const CUT = 90;
  const td = h('td', { class: 'desk-fills-why' });
  const short = rationale.length > CUT ? rationale.slice(0, CUT) + '…' : rationale;
  const txt = h('span', { text: short });
  td.append(txt);
  if (short !== rationale) {
    let open = false;
    const btn = h('button', { class: 'desk-fills-more', type: 'button', text: 'more' });
    btn.addEventListener('click', () => {
      open = !open;
      txt.textContent = open ? rationale : short;
      td.classList.toggle('expanded', open);
      btn.textContent = open ? 'less' : 'more';
    });
    td.append(' ', btn);
  }
  return td;
}

async function loadFills() {
  const el = document.getElementById('desk-fills');
  const metaEl = document.getElementById('desk-fills-meta');
  if (!el) return;
  skeleton(el);
  try {
    const rows = await apiGet('/api/desk/trades?limit=200');
    if (!rows.length) { renderEmpty(el, 'No trades yet.'); if (metaEl) metaEl.textContent = ''; return; }
    if (metaEl) metaEl.textContent = rows.length + ' most recent';
    clear(el);
    const table = h('table', { class: 'c-table' },
      h('thead', {}, h('tr', {},
        h('th', { text: 'When' }), h('th', { text: 'Stock' }),
        h('th', { text: 'Side' }), h('th', { class: 'num', text: 'Shares' }),
        h('th', { class: 'num', text: 'Fill price' }),
        h('th', { class: 'num', text: 'Live bid / ask', title: 'The real-time bid and ask the fill priced against at that moment, plus any fee, session, or override receipts stamped on the fill' }),
        h('th', { class: 'num', text: 'Value' }),
        h('th', { text: 'Why', title: 'The AI\'s stated reason for this trade at the time' }))),
      h('tbody', {}, ...rows.map(r => {
        const q = r.fill_quote || {};
        const isOpt = OCC_RE_F.test(r.symbol);
        const buy = (r.side || '').toUpperCase() === 'BUY';
        const quoteCell = h('td', { class: 'num' },
          (q.bid != null && q.ask != null)
            ? h('span', { class: 't-dim', text: fmtPrice(q.bid) + ' / ' + fmtPrice(q.ask) })
            : h('span', { class: 't-dim', text: '—' }));
        // Receipt extras stamped on the fill: option fee, session tag,
        // override/degraded-gate warnings — the honesty trail, visible.
        const extras = [];
        if (q.session && q.session !== 'regular') {
          extras.push(pill(String(q.session).toUpperCase(), 'neutral',
            'This fill booked outside regular trading hours.'));
        }
        if (q.fee && q.fee.total) {
          extras.push(h('span', { class: 't-dim',
            title: q.fee.contracts + ' contract(s) × ' + fmtPrice(q.fee.per_contract) + ' per-contract fee, included in the value',
            text: 'fee ' + fmtPrice(q.fee.total) }));
        }
        if (q.warnings && q.warnings.length) {
          extras.push(pill('override', 'warn', q.warnings.join('\n')));
        }
        if (extras.length) quoteCell.append(h('div', { class: 'desk-fill-extra' }, ...extras));
        return h('tr', {},
          h('td', { class: 't-dim', title: fmtDateTimeET(r.t), text: timeAgo(r.t) }),
          h('td', {}, h('a', { href: '/symbol/' + (isOpt ? r.symbol.match(/^[A-Z]+/)[0] : r.symbol), class: 'c-link', text: r.symbol })),
          h('td', {}, pill((r.side || '').toUpperCase() || '—', buy ? 'up' : 'down')),
          h('td', { class: 'num', text: fmtNum(Math.abs(r.shares), isOpt ? 0 : 2) }),
          h('td', { class: 'num', text: fmtPrice(r.price) }),
          quoteCell,
          h('td', { class: 'num', text: fmtDollar(Math.abs(r.dollars)) }),
          fillWhyCell((r.rationale || '').trim()));
      })));
    el.append(table);
  } catch (err) { renderError(el, err, loadFills); }
}

/* ── what's new: dashboard improvements the agent shipped ── */
const WN_KIND_CLASS = { feature: 'info', improvement: 'info', data: 'neutral', disclaimer: 'warn', fix: 'up' };

function wnEntry(e) {
  return h('div', { class: 'desk-wn-entry' },
    h('div', { class: 'desk-wn-entry-head' },
      pill((e.kind || 'feature').toUpperCase(), WN_KIND_CLASS[e.kind] || 'info'),
      h('span', { class: 'desk-wn-entry-title', text: e.title }),
      e.version ? h('span', { class: 't-dim', text: 'v' + e.version }) : null,
      h('span', { class: 'spacer' }),
      h('span', { class: 't-dim', text: timeAgo(e.t) })),
    e.detail ? h('p', { class: 'desk-wn-entry-detail t-dim', text: e.detail }) : null);
}

async function loadWhatsNew() {
  const btn = document.getElementById('desk-whatsnew-btn');
  const badge = document.getElementById('desk-whatsnew-badge');
  const banner = document.getElementById('desk-whatsnew-banner');
  const panel = document.getElementById('desk-whatsnew-panel');
  try {
    const data = await apiGet('/api/desk/whatsnew?limit=25');
    const entries = data.entries || [];
    if (!entries.length) { btn.hidden = true; banner.hidden = true; panel.hidden = true; return; }
    btn.hidden = false;

    // header badge: count of entries still inside the spotlight window
    if (data.new_count > 0) {
      badge.textContent = String(data.new_count);
      badge.hidden = false;
      btn.classList.add('has-new');
    } else {
      badge.hidden = true;
      btn.classList.remove('has-new');
    }

    // full feed — stays collapsed until the header button is clicked
    clear(panel);
    panel.append(
      h('div', { class: 'desk-wn-panel-head' },
        h('span', { text: "What's New" }),
        h('span', { class: 't-dim', text: 'how this dashboard is evolving' })),
      h('div', { class: 'desk-wn-list' }, ...entries.map(wnEntry)));

    // attention banner — the newest entry, while still "new" and not dismissed
    const latest = data.latest;
    const dismissed = localStorage.getItem('ef-wn-banner');
    if (latest && data.new_count > 0 && dismissed !== String(latest.id)) {
      clear(banner);
      banner.append(
        h('span', { class: 'desk-wn-spark', text: '◆' }),
        pill('NEW', WN_KIND_CLASS[latest.kind] || 'info'),
        h('span', { class: 'desk-wn-banner-title', text: latest.title }),
        latest.detail ? h('span', { class: 'desk-wn-banner-detail t-dim', text: latest.detail }) : null,
        h('span', { class: 'spacer' }),
        h('button', {
          class: 'desk-wn-dismiss', type: 'button', title: 'Dismiss', 'aria-label': 'Dismiss',
          text: '×',
          onclick: () => { localStorage.setItem('ef-wn-banner', String(latest.id)); banner.hidden = true; },
        }));
      banner.hidden = false;
    } else {
      banner.hidden = true;
    }
  } catch (err) {
    // What's New is non-critical chrome — never break the page over it
    btn.hidden = true; banner.hidden = true; panel.hidden = true;
  }
}

// toggle the full panel from the header button (wired once)
(function wireWhatsNewToggle() {
  const btn = document.getElementById('desk-whatsnew-btn');
  const panel = document.getElementById('desk-whatsnew-panel');
  if (!btn || !panel) return;
  btn.addEventListener('click', () => {
    const show = panel.hidden;
    panel.hidden = !show;
    btn.setAttribute('aria-expanded', show ? 'true' : 'false');
    if (show) panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  });
})();

async function loadAll() {
  await Promise.all([
    loadHeader(), loadEquity(), loadPositions(), loadThinking(),
    loadDecision(), loadWhatsNew(), loadFills(), loadWiki(),
    loadLab(), loadWatch(), loadPredictions(),
  ]);
}

/* ── card collapse: chevron in each card header toggles visibility,
   preferences persisted per-card in localStorage. Cards can opt into a
   default-collapsed state via data-collapsed="1" on the .desk-card. ── */
const COLLAPSE_KEY = 'ef-desk-collapse-v1';
function loadCollapseSet() {
  try { return new Set(JSON.parse(localStorage.getItem(COLLAPSE_KEY) || '[]')); }
  catch (e) { return new Set(); }
}
function saveCollapseSet(set) {
  try { localStorage.setItem(COLLAPSE_KEY, JSON.stringify([...set])); }
  catch (e) { /* private mode, quota — silent */ }
}
function wireCollapse() {
  const cards = document.querySelectorAll('.desk-card[data-collapse-key]');
  const persisted = loadCollapseSet();
  // First-visit defaults come from data-collapsed="1"; user prefs override.
  for (const card of cards) {
    const key = card.getAttribute('data-collapse-key');
    if (!key) continue;
    if (persisted.has(key) || (card.getAttribute('data-collapsed') === '1' && !persisted.has('!' + key))) {
      card.classList.add('collapsed');
    }
  }
  document.addEventListener('click', ev => {
    const btn = ev.target.closest('[data-collapse-btn]');
    if (!btn) return;
    const card = btn.closest('.desk-card');
    if (!card) return;
    const key = card.getAttribute('data-collapse-key');
    if (!key) return;
    const set = loadCollapseSet();
    const wasCollapsed = card.classList.toggle('collapsed');
    if (wasCollapsed) {
      set.add(key); set.delete('!' + key);
    } else {
      set.delete(key);
      // remember the user explicitly opened a default-collapsed card
      if (card.getAttribute('data-collapsed') === '1') set.add('!' + key);
    }
    saveCollapseSet(set);
  });
}

/* ── anchor nav scrollspy: highlight the visible zone in the sticky nav.
   Also smooth-scroll into view on click, respecting scroll-margin-top so
   the target zone header clears the sticky topnav + anchor bar. ── */
function wireAnchorNav() {
  const nav = document.getElementById('desk-anchornav');
  if (!nav) return;
  const anchors = [...nav.querySelectorAll('.desk-anchor')];
  const setActive = zoneId => {
    for (const a of anchors) {
      const on = a.getAttribute('data-zone') === zoneId;
      a.classList.toggle('active', on);
    }
  };
  // click: smooth-scroll to the zone
  nav.addEventListener('click', ev => {
    const a = ev.target.closest('.desk-anchor');
    if (!a) return;
    const zoneId = a.getAttribute('data-zone');
    const zone = document.getElementById(zoneId);
    if (!zone) return;
    ev.preventDefault();
    zone.scrollIntoView({ behavior: 'smooth', block: 'start' });
    setActive(zoneId);
    history.replaceState(null, '', '#' + zoneId);
  });
  // scrollspy: highlight whichever zone the reader is looking at
  const zones = [...document.querySelectorAll('.desk-zone')];
  if (!zones.length || !('IntersectionObserver' in window)) return;
  const io = new IntersectionObserver(entries => {
    const visible = entries
      .filter(e => e.isIntersecting)
      .sort((a, b) => b.intersectionRatio - a.intersectionRatio);
    if (visible.length) setActive(visible[0].target.id);
  }, {
    // Zone counts as "visible" when its top clears the sticky bar
    rootMargin: '-120px 0px -55% 0px',
    threshold: [0, 0.1, 0.25, 0.5],
  });
  for (const z of zones) io.observe(z);
  // default: first zone is active
  setActive(zones[0].id);
}

/* ── card-header seg toggles (lab: board/tests, notebook: lessons/diary) ── */
function wireSegs() {
  const wire = (segId, apply) => {
    const seg = document.getElementById(segId);
    if (!seg) return;
    seg.addEventListener('click', ev => {
      const btn = ev.target.closest('button[data-view]');
      if (!btn) return;
      for (const b of seg.querySelectorAll('button')) b.classList.remove('active');
      btn.classList.add('active');
      apply(btn.dataset.view);
    });
  };
  wire('desk-lab-seg', v => { labView = v; renderLab(); });
  wire('desk-wiki-seg', v => { wikiView = v; renderNotebook(); });
  wire('desk-decision-seg', v => { decisionView = v; applyDecisionView(); });
}

wireCollapse();
wireAnchorNav();
wireSegs();
loadAll();
startTape();
// refresh the live panels periodically (the agent updates several times/day)
setInterval(() => { loadHeader(); loadThinking(); loadDecision(); loadWhatsNew(); loadWiki(); loadLab(); loadWatch(); loadPredictions(); }, 60_000);

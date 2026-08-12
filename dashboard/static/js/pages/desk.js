/* Trading Desk — the autonomous agent's window.
   Four panels: (stats + equity curve), holdings, live thinking feed,
   the latest decision (picks with why-now / rationale / news + watchlist),
   and the evidence (backtests it ran) + strategy journal. Reads the new
   /api/desk/* endpoints; reuses the core charts/dom/fmt/net modules. */

import { apiGet } from '../core/net.js';
import { toEpochSec, fmtDollar, fmtPnl, fmtPct, fmtPrice, fmtNum, timeAgo,
         fmtDateET, fmtTimeET, fmtDateTimeET } from '../core/fmt.js';
import { h, svg, clear, skeleton, renderEmpty, renderError } from '../core/dom.js';
import { createChart, colors } from '../core/charts.js';
import { onThemeChange } from '../core/theme.js';
import { treemap } from '../components/treemap.js';

let equityChart = null;
let equitySeries = null;   // the recorded marks — history, never fabricated
let spySeries = null;      // faint benchmark overlay: SPY price return, rebased
let liveTipSeries = null;  // ONE dashed connector: last real mark → live estimate
let lastEquityData = [];
let cutoverTime = null;    // epoch sec of the first era-2 point (cutover marker)

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

/* ── capped lists ──
   Long corpora (179 claims, 50 holdings, every graded prediction) used to
   render in full and made the page 40,971px tall. Cap them, and put the real
   total in the toggle: the count IS the disclosure, so nothing is ever
   silently truncated. Rows past the cap are rendered but hidden, so expanding
   is instant and costs no request.

   `rows` are already-built elements; `noun`/`nouns` name what is being
   counted. Returns the toggle button, or null when nothing was hidden. */
function capList(rows, cap, noun, nouns) {
  if (rows.length <= cap) return null;
  const hidden = rows.slice(cap);
  for (const r of hidden) r.hidden = true;
  const label = n => `Show all ${rows.length} ${rows.length === 1 ? noun : (nouns || noun + 's')}`;
  const btn = h('button', { class: 'desk-morebtn', type: 'button', text: label() });
  btn.addEventListener('click', () => {
    const expanding = hidden[0].hidden;
    for (const r of hidden) r.hidden = !expanding;
    btn.textContent = expanding ? `Show fewer` : label();
  });
  return btn;
}

/* ── display typography for agent-authored prose ──

   " -- " is the agent's habit for an em dash and turns up in nearly every
   rationale; it is the clearest "no human read this back" tell on the page.
   The trading skill now forbids writing it, but the corpus already on the
   book is full of it, so normalise at render too. Spaces are required on
   BOTH sides, which is exactly what leaves a CLI flag like `--bump` alone. */
function dashes(s) {
  return String(s == null ? '' : s).replace(/ -- /g, ' — ');
}

/* `**bold**` → a real <strong>, built as nodes. Returns an array to spread
   into h(); this codebase sets no innerHTML anywhere and this is not the
   place to start. */
function inlineMd(text) {
  const out = [];
  const re = /\*\*(.+?)\*\*/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(text.slice(last, m.index));
    out.push(h('strong', { text: m[1] }));
    last = m.index + m[0].length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out.length ? out : [text];
}

function pill(text, cls, title) {
  return h('span', { class: 'c-pill ' + (cls || 'neutral'), title: title || null, text });
}

/* ── LIVE pill truth ──
   The pulsing green LIVE dot only appears when a threshold fraction of the
   HELD symbols' quotes are genuinely fresh AND the session tag says the
   market is verifiably open — 'regular' or 'extended' (the SSE payload
   carries per-symbol staleness + the session tag). Fresh quotes alone are
   NOT proof the tape is live: REST cache warming stamps recv=now, so a warm
   cache on a closed market would otherwise read as ticking. A stale tape
   shows DELAYED; a closed market shows CLOSED; an unverifiable session
   (null) caps the pill at DELAYED — frozen numbers never masquerade as
   ticking ones. */
const LIVE_PILL = {
  live: ['LIVE', 'Fresh real-time quotes are folding into the account value — '
    + 'the numbers above are the live market value.'],
  delayed: ['DELAYED', 'The live quote stream is stale — the numbers above are '
    + 'the most recent recorded prices, not a live market value.'],
  closed: ['CLOSED', 'The market is closed — prices shown are the last '
    + 'session’s marks.'],
  unknown: ['UNKNOWN', 'The market session can’t be verified right now and '
    + 'the quote stream isn’t fresh — the numbers above are the most recent '
    + 'recorded prices, not a live market value.'],
};

function setLivePill(state) {
  const el = document.getElementById('desk-hero-live');
  if (!el) return;
  if (!state) { el.hidden = true; return; }
  const [label, tip] = LIVE_PILL[state] || LIVE_PILL.delayed;
  el.hidden = false;
  el.classList.toggle('delayed', state === 'delayed');
  el.classList.toggle('closed', state === 'closed' || state === 'unknown');
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
  const freshEnough = fresh >= Math.max(1, Math.ceil(heldEq.length / 2));
  // LIVE requires a VERIFIED open session ('regular' | 'extended'). A null
  // session (no keys / clock unreachable) must never fall through to
  // freshness alone — cache warming stamps recv=now, so warm REST quotes on
  // a closed market look fresh. Unknown session: DELAYED at best.
  if (snap.session === 'regular' || snap.session === 'extended') {
    return freshEnough ? 'live' : 'delayed';
  }
  return freshEnough ? 'delayed' : 'unknown';
}

/* the account's change since the last completed trading session — grouped
   from the same equity marks the chart already shows, oldest→newest */
function todayChange(series) {
  if (!series || series.length < 2) return null;
  const byDay = new Map();
  for (const p of series) {
    if (p.ts && p.equity != null) byDay.set(p.ts.slice(0, 10), p.equity);
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
    const [pf, strat, regime, eqBody, dataHealth] = await Promise.all([
      apiGet('/api/desk/portfolio'),
      apiGet('/api/desk/strategy'),
      apiGet('/api/desk/regime').catch(() => null),
      apiGet('/api/desk/equity?limit=500').catch(() => null),
      // A failed health check must render as VISIBLY unknown, not vanish —
      // the pill exists to surface exactly the states where fetches fail.
      apiGet('/api/desk/data-health').catch(() => ({ status: 'unknown' })),
    ]);
    const eq = (eqBody && eqBody.points) || [];

    // The paper account can be unreachable (no trade keys on this host,
    // broker outage). Render the gap honestly — dashes, never fake numbers.
    const degraded = pf && pf.available === false;
    if (degraded) {
      setText('desk-hero-account', '—');
      setText('desk-hero-pnl', '—', '');
      setText('desk-hero-return', '—', '');
      setText('desk-hero-alpha', '—', '');
      setText('desk-hero-cash', '—');
      setText('desk-hero-count', '—');
    } else {
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
    }

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
      if (degraded) {
        chipsEl.append(h('span', {
          class: 'c-pill warn',
          title: 'The Alpaca paper account could not be reached from this '
            + 'host — account figures are unavailable until it recovers.',
          text: 'Account: unreachable',
        }));
      }
      if (strat && strat.current) {
        // The agent names its own strategies, and the names pile up clauses:
        // "aggressive barbell: conviction core + wide trial sleeve + tactical
        // toolkit" is the first thing a visitor reads, and it ran 160px past
        // a phone viewport. Show the head of the name; the full name and the
        // thesis behind it live in the tooltip.
        const full = String(strat.current.name || '');
        const head = full.split(':')[0].trim() || full;
        const title = [full !== head ? full : null, strat.current.thesis]
          .filter(Boolean).join(' — ')
          || 'The strategy the AI is currently running.';
        chipsEl.append(h('span', {
          class: 'c-pill info',
          title,
          text: 'Strategy v' + strat.current.version + (head ? ' · ' + head : ''),
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
  // No explicit height: #desk-equity-chart is a CSS-sized .ch-pane.tall, and
  // the shared ResizeObserver in charts.js is built to track a sized
  // container on both axes. Passing a pixel height here just got overwritten
  // on the first observer callback, so the responsive height lives in
  // charts.css alone — one source of truth.
  equityChart = createChart(el);
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
    // markers capture their color at set time (applyOptions can't restyle
    // them) — re-set the cutover marker in the new palette
    applyCutoverMarker();
  });
}

/* The era cutover (hand-rolled ledger → Alpaca paper account) gets one
   labeled marker on the first era-2 point. A helper so the theme-change
   handler can re-apply it in the new palette. */
function applyCutoverMarker() {
  if (!equitySeries) return;
  const c = colors();
  equitySeries.setMarkers(cutoverTime == null ? [] : [{
    time: cutoverTime, position: 'aboveBar', shape: 'arrowDown',
    color: c.warn, id: 'cutover', text: 'cutover',
  }]);
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
    // ONE continuous series across both eras; de-dup identical timestamps
    // (the chart requires strictly increasing time).
    const seen = new Set();
    const data = [];
    let firstEra2 = null;
    const hasEra1 = series.some(p => p.era === 1);
    for (const p of series) {
      const time = toEpochSec(p.ts);
      if (!time || seen.has(time)) continue;
      seen.add(time);
      data.push({ time, value: p.equity });
      if (firstEra2 == null && p.era === 2) firstEra2 = time;
    }
    lastEquityData = data;
    // The cutover marker only means something once BOTH eras are on the
    // chart — a pure era-2 curve starts at the cutover by definition.
    cutoverTime = (hasEra1 && body.era2_inception != null) ? firstEra2 : null;
    equitySeries.setData(data);
    applyCutoverMarker();
    // SPY overlay (price return — the paper book collects no dividends, so
    // the benchmark drops its dividends too): the account's starting value
    // ridden on SPY from the close before the first trade.
    const start = data.length ? data[0].value : 100000;
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
    metaEl.textContent = `${fmtDollar(last.equity)} · ${series.length} marks`;
    metaEl.title = '';
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

/* Allocation treemap — how the account is split across holdings + cash.

   Was a 48-slice donut with its legend in one tall column: unreadable by
   construction (a 0.4% arc is invisible) and 1,023px of page, more than the
   decision and thinking feed put together. A treemap sizes by area, so the
   positions that actually matter are the ones you can read, and the long
   tail pools into one honest tile instead of forty lines.

   The tail is pooled, never dropped — the tile states its own count. */
const ALLOC_TOP = 12;
const ALLOC_HEIGHT = 260;
const HOLDINGS_CAP = 10;
const CLAIMS_CAP = 5;
const OUTCOMES_CAP = 8;
const LESSONS_CAP = 3;
const ORDERS_CAP = 12;
const FILLS_CAP = 15;

function allocationNodes(pf) {
  const held = (pf.positions || [])
    .filter(p => p.weight > 0)
    .map(p => ({ symbol: p.symbol, value: p.weight * 100 }))
    .sort((a, b) => b.value - a.value);
  const nodes = held.slice(0, ALLOC_TOP).map((p, i) => {
    const occ = occParse(p.symbol);
    return {
      name: occ ? occ.label : p.symbol,
      symbol: occ ? occ.underlying : p.symbol,
      value: p.value,
      count: fmtNum(p.value, 1) + '%',
      quadrant: 's' + (i % 8),
    };
  });
  const tail = held.slice(ALLOC_TOP);
  if (tail.length) {
    const tailPct = tail.reduce((s, p) => s + p.value, 0);
    nodes.push({
      name: `${tail.length} smaller position${tail.length === 1 ? '' : 's'}`,
      value: tailPct,
      count: fmtNum(tailPct, 1) + '%',
      quadrant: 'tail',
    });
  }
  const cashPct = pf.equity ? Math.max(0, pf.cash / pf.equity * 100) : 0;
  if (cashPct > 0.05) {
    nodes.push({ name: 'Cash', value: cashPct, count: fmtNum(cashPct, 1) + '%', quadrant: 'cash' });
  }
  return nodes;
}

/* Kept as module state so a tab switch can re-lay-out: treemap() reads
   clientWidth, which is 0 inside a hidden panel. */
let allocNodes = null;

function renderAllocation(host) {
  if (!allocNodes || !allocNodes.length) return;
  treemap(host, allocNodes, {
    height: ALLOC_HEIGHT,
    onClick: name => {
      const n = allocNodes.find(x => x.name === name);
      if (n && n.symbol) location.href = '/symbol/' + n.symbol;
    },
  });
}

function redrawAllocation() {
  const host = document.getElementById('desk-alloc');
  if (host && host.offsetParent !== null) renderAllocation(host);
}

function allocation(pf) {
  allocNodes = allocationNodes(pf);
  if (!allocNodes.length) return null;
  const host = h('div', { id: 'desk-alloc', class: 'desk-alloc' });
  // treemap() measures clientWidth, so it can only run once the node is in
  // the document — renderPositions calls back through renderAllocation.
  return host;
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
      h('td', { class: 'num', text: fmtNum(p.qty, 2) }),
      h('td', { class: 'num', text: fmtPrice(p.avg_entry_price) }),
      h('td', { class: 'num', text: fmtPrice(p.current_price) }),
      dayChangeCell(stats[p.symbol]),
      trendCell(stats[p.symbol]),
      h('td', { class: 'num', text: fmtDollar(p.market_value) }),
      // weight is a 0-1 fraction — scale to percent for display
      h('td', { class: 'num', text: fmtPct(p.weight * 100, { signed: false }) }),
      h('td', { class: 'num ' + (p.unrealized_pl >= 0 ? 't-up' : 't-down'),
        text: fmtPnl(p.unrealized_pl) })))));
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
      const short = p.qty < 0;
      return h('tr', {},
        h('td', {}, h('a', { href: '/symbol/' + o.underlying, class: 'c-link', text: o.label })),
        h('td', {}, pill(short ? 'SHORT' : 'LONG', short ? 'warn' : 'info')),
        h('td', { class: 'num', text: fmtNum(Math.abs(p.qty), 0) }),
        h('td', { class: 'num ' + (o.dte <= 5 ? 't-down' : ''), text: String(o.dte) }),
        h('td', { class: 'num', text: fmtPrice(p.avg_entry_price) }),
        h('td', { class: 'num', text: fmtPrice(p.current_price) }),
        h('td', { class: 'num', text: fmtDollar(p.market_value) }),
        h('td', { class: 'num ' + (p.unrealized_pl >= 0 ? 't-up' : 't-down'),
          text: fmtPnl(p.unrealized_pl) }));
    })));
}

function renderPositions(el, pf, stats) {
  const eqs = pf.positions.filter(p => !occParse(p.symbol));
  const opts = pf.positions.filter(p => occParse(p.symbol));
  clear(el);
  const alloc = allocation(pf);
  if (alloc) { el.append(alloc); renderAllocation(alloc); }
  if (eqs.length) {
    const table = equitiesTable(eqs, stats);
    el.append(table);
    // The book runs 40+ names; the top ten are 80%+ of it. The rest are one
    // click away with their count on the button.
    //
    // "stocks", not "holdings": this caps the EQUITY table only, and the hero
    // counts stocks + option contracts together. A button reading "show all 44
    // holdings" under a hero reading "50 positions held" invites the reader to
    // wonder which number is lying.
    const more = capList([...table.querySelectorAll('tbody > tr')], HOLDINGS_CAP,
      'stock', 'stocks');
    if (more) el.append(more);
  }
  // The options book is NOT capped: it runs a handful of rows and is the part
  // of the account least legible from anywhere else.
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
    const price = (live != null && Number.isFinite(live)) ? live : p.current_price;
    const mult = isOpt ? 100 : 1;
    const mv = Math.round(p.qty * price * mult * 100) / 100;
    posValue += mv;
    positions.push({
      ...p,
      current_price: Math.round(price * 10000) / 10000,
      market_value: mv,
      unrealized_pl: Math.round(p.qty * (price - p.avg_entry_price) * mult * 100) / 100,
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
  // Keep 'vs S&P 500' consistent with the live ticks — the SPY side is
  // daily-close based and static between page loads. Same window both
  // sides: with an era-1 archive the benchmark window is all-time, so our
  // side re-derives from the era-1 base equity, not the era-2 return.
  const vs = ref ? ref.vs_spy : null;
  const spy = vs ? vs.spy_return_pct : null;
  if (spy != null) {
    const base = vs.alltime_base_equity;
    const ours = base > 0 ? ((equity / base) - 1) * 100 : returnPct;
    const a = ours - spy;
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
  livePositions = {
    ...ref, positions, positions_value: posValue,
    equity, total_pnl: totalPnl, total_return_pct: returnPct,
  };
  repaintPositions();
}

/* The last live-folded book, so a tab switch can repaint from it. */
let livePositions = null;

/* Repaint the holdings card from the live fold — but only when it is
   actually on screen.

   Two reasons this guard exists. The treemap lays out against
   `clientWidth`, which is 0 inside a hidden panel, so it would silently fall
   back to its 600px default and lay the tiles out for the wrong box. And the
   rebuild runs once per tape frame: at market open that is a full table plus
   a squarify pass, several times a second, for a panel nobody is looking at.
   wireTabs calls this on activation, so switching back paints current marks
   immediately. */
function repaintPositions() {
  const el = document.getElementById('desk-positions');
  if (!livePositions || !el || el.offsetParent === null) return;
  if (el.querySelector('.c-skel') || el.querySelector('.c-empty')) return;
  renderPositions(el, livePositions, deskLive.stats);
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
    // Honesty strip: the running estimate of dividends the paper broker
    // never paid (rendered even at $0 — the disclosure is the point).
    const missedEl = document.getElementById('desk-honesty-missed');
    if (missedEl) {
      const total = dv && dv.missed_dividends ? dv.missed_dividends.total : null;
      missedEl.textContent = fmtDollar(total || 0);
    }
    if (pf.available === false) {
      renderEmpty(el, 'The paper account is unreachable right now — holdings '
        + 'will reappear when the broker connection recovers.');
      return;
    }
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
        h('span', { class: 'desk-feed-text', text: dashes(line.text) }),
        // The exact ET stamp, with the relative age under it. A feed line is
        // the agent narrating a moment — "4m ago" says it is fresh, but only
        // the clock time lets a reader line the thought up against the tape
        // and the fill it led to.
        h('span', { class: 'desk-feed-time t-dim' },
          h('div', { text: fmtTimeET(line.t) }),
          h('div', { class: 'desk-feed-ago', text: timeAgo(line.t) })));
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

/* ── open orders & resting protection: real orders on the broker's book ──

   Rendered as 44 near-identical English sentences until v10.3.0 ("ABNB —
   protective stop at $168.00 on 3.00 shares", forty-three more times). The
   same facts in a table are scannable and a fifth of the height; the prose
   only ever restated the column headers. ── */

/* "placed today" reads; "resting 1d" needs decoding. */
function orderAge(days) {
  if (days == null) return '—';
  if (days <= 0) return 'today';
  if (days === 1) return 'yesterday';
  return days + ' days ago';
}

function orderTriggerCell(o) {
  if (o.kind === 'stop') {
    return fmtPrice(o.stop_price)
      + (o.limit_price != null ? ' limit ' + fmtPrice(o.limit_price) : '');
  }
  if (o.kind === 'limit') return fmtPrice(o.limit_price);
  return '—';
}

function openOrderRow(o) {
  const stop = o.kind === 'stop';
  const occ = occParse(o.symbol);
  const kindPill = pill(
    stop ? 'STOP' : (o.kind === 'limit' ? 'LIMIT' : (o.order_type || 'ORDER').toUpperCase()),
    stop ? 'warn' : 'info',
    stop ? 'A resting stop-loss order on the broker’s own book — it fires even while the AI is offline.'
         : 'A working order resting at the broker.');
  const symCell = h('td', {},
    h('a', { href: '/symbol/' + (occ ? occ.underlying : o.symbol),
      class: 'c-link', text: occ ? occ.label : o.symbol }));
  // A GTC order Alpaca is about to cancel out from under us is real risk, so
  // it keeps its loud pill rather than becoming another quiet cell.
  if (o.tif === 'gtc' && o.age_days != null && o.age_days >= 80) {
    symCell.append(h('div', {}, pill('GTC expires at 90d', 'down',
      'Alpaca silently cancels GTC orders after 90 days — this one is '
      + o.age_days + ' days old and needs re-arming soon.')));
  }
  return h('tr', {},
    symCell,
    h('td', {}, kindPill),
    h('td', { class: 'num', text: orderTriggerCell(o) }),
    h('td', { class: 'num', text: o.qty != null ? fmtNum(o.qty, 2) : '—' }),
    h('td', { class: 'num t-dim', text: orderAge(o.age_days) }));
}

function ordersTable(rows) {
  const body = h('tbody', {}, ...rows.map(openOrderRow));
  return h('div', { class: 'c-table-wrap' },
    h('table', { class: 'c-table' },
      h('thead', {}, h('tr', {},
        h('th', { text: 'Stock' }),
        h('th', { text: 'Type' }),
        h('th', { class: 'num', text: 'Trigger', title: 'The price that fires this order' }),
        h('th', { class: 'num', text: 'Quantity' }),
        h('th', { class: 'num', text: 'Placed' }))),
      body));
}

async function loadOpenOrders() {
  const el = document.getElementById('desk-orders');
  const metaEl = document.getElementById('desk-orders-meta');
  if (!el) return;
  skeleton(el);
  try {
    const d = await apiGet('/api/desk/open-orders');
    clear(el);
    if (d.available === false) {
      if (metaEl) metaEl.textContent = '';
      renderEmpty(el, 'The broker connection is unavailable on this host — resting orders cannot be shown.');
      return;
    }
    const rows = d.orders || [];
    const stops = rows.filter(o => o.kind === 'stop').length;
    if (metaEl) {
      metaEl.textContent = rows.length
        ? rows.length + ' resting' + (stops ? ' · ' + stops + ' stop' + (stops === 1 ? '' : 's') : '')
        : '';
    }
    if (!rows.length) {
      renderEmpty(el, 'Nothing resting at the broker right now — the AI arms protective stops on positions that need them.');
      return;
    }
    // One sentence of what the protection actually amounts to, then the
    // table. The dollar figure is what a stop is FOR — how much of the book
    // has a floor under it — and no per-row sentence ever said it.
    const covered = rows
      .filter(o => o.kind === 'stop' && o.qty != null && o.stop_price != null)
      .reduce((s, o) => s + o.qty * o.stop_price * (occParse(o.symbol) ? 100 : 1), 0);
    el.append(h('p', { class: 'desk-orders-summary' },
      h('strong', { text: String(rows.length) }),
      h('span', { text: ' order' + (rows.length === 1 ? '' : 's') + ' resting at the broker' }),
      stops
        ? h('span', { text: ' — ' + stops + ' protective stop' + (stops === 1 ? '' : 's')
            + (covered > 0 ? ' with a floor under ' + fmtDollar(covered) + ' of the book' : '') + '.' })
        : h('span', { text: '.' })));
    const table = ordersTable(rows);
    el.append(table);
    const more = capList([...table.querySelectorAll('tbody > tr')], ORDERS_CAP,
      'order', 'orders');
    if (more) el.append(more);
  } catch (err) { renderError(el, err, loadOpenOrders); }
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
      p.why_now ? h('span', { class: 'desk-pick-why t-dim', text: dashes(p.why_now) }) : null),
    p.rationale ? h('p', { class: 'desk-pick-rationale', text: dashes(p.rationale) }) : null);
  // The pre-registered prediction. This is the most valuable thing on the
  // page — a falsifiable commitment made BEFORE the trade — and until
  // v10.3.0 it rendered as three lowercase key-colon-value chips that read
  // like debug output. A labelled block, in plain words, instead.
  const commitments = [];
  if (p.prediction) commitments.push(['The bet', dashes(p.prediction)]);
  if (p.horizon_days != null) {
    commitments.push(['Checked by',
      p.horizon_days + ' trading session' + (p.horizon_days === 1 ? '' : 's') + ' from now']);
  }
  if (p.kill) commitments.push(['Called off if', dashes(p.kill)]);
  if (commitments.length) {
    const dl = h('dl', {
      class: 'desk-pick-commit',
      title: 'What the AI committed to before buying — graded later in “Predictions vs outcomes”.',
    });
    for (const [k, v] of commitments) {
      dl.append(h('dt', { text: k }), h('dd', { text: v }));
    }
    card.append(dl);
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
      w.note ? h('span', { class: 't-dim', text: ' — ' + dashes(w.note) }) : null)));
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
    sumEl.append(h('p', { class: 'desk-summary', text: dashes(d.summary || '') }));

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
  if (d.summary) box.append(h('p', { class: 'desk-summary', text: dashes(d.summary) }));
  for (const p of (d.picks || [])) box.append(pickCard(p));
  if (d.watchlist && d.watchlist.length) box.append(watchlistChips(d.watchlist));
  if (d.rejected && d.rejected.length) {
    box.append(
      h('div', { class: 'desk-subhead', text: 'Passed on' }),
      h('ul', { class: 'desk-dec-rej' }, ...d.rejected.map(r => {
        const sym = (r && r.symbol) || String(r);
        return h('li', {},
          h('a', { href: '/symbol/' + sym, class: 'c-link', text: sym }),
          r && r.why_not ? ' — ' + dashes(r.why_not) : '');
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
    // When the desk committed to this prediction — the stamp is what makes
    // the claim checkable against the price at that moment, not just "3d ago".
    h('span', { class: 't-dim', title: 'decision run ' + (r.run_id || ''),
      text: r.decision_ts ? fmtDateTimeET(r.decision_ts) : '' }));

  const pred = h('p', { class: 'desk-outcome-pred',
    text: r.prediction ? '“' + dashes(r.prediction) + '”'
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
      title: 'The move minus SPY’s price move over the same window (price return on both sides — the paper book collects no dividends) — skill, not market tide.',
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
      text: reached ? 'deadline reached (' + r.horizon_days + ' sessions)'
        : (done != null ? 'session ' + done + ' of ' + r.horizon_days
          : r.horizon_days + '-session deadline'),
    }));
  }
  if (open && r.kill_level != null) {
    const live = deskLive.marks[r.symbol];
    const now = (live != null && Number.isFinite(live)) ? live : r.mark_px;
    chips.push(h('span', {
      class: 'c-chip' + (r.kill_breached ? ' t-down' : ''),
      title: r.kill ? 'The condition that calls this pick off: ' + r.kill
                    : 'The level that calls this pick off.',
      text: 'called off at ' + fmtPrice(r.kill_level)
        + (now != null ? ' · now ' + fmtPrice(now) : '')
        + (r.kill_breached ? ' — BREACHED' : ''),
    }));
  } else if (open && r.kill) {
    chips.push(h('span', { class: 'c-chip', text: 'called off if ' + r.kill }));
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
    // Open and closed cap independently: an open prediction is live state a
    // reader wants all of, a closed one is history.
    const section = (label, list, noun) => {
      if (!list.length) return;
      el.append(h('div', { class: 'desk-outcome-subhead', text: label }));
      const built = list.map(outcomeRow);
      for (const r of built) el.append(r);
      const more = capList(built, OUTCOMES_CAP, noun, noun + 's');
      if (more) el.append(more);
    };
    section('Open', rows.filter(r => r.status === 'open'), 'open prediction');
    section('Closed', rows.filter(r => r.status !== 'open'), 'closed prediction');
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
  // markdown-lite: blank-line-separated blocks. "- " blocks \u2192 bullet lists,
  // "#"-prefixed lines \u2192 headings, everything else \u2192 paragraphs. All text
  // nodes \u2014 zero innerHTML. A [risk_on]-style tag anywhere in a bullet
  // becomes a small plain-English pill instead of raw shorthand.
  //
  // Headings and bold were added in v10.3.0: the agent writes markdown, and
  // without a parser for it the notebook rendered literal "## Posture:" and
  // "**never**" down the page \u2014 its own structure showing as punctuation.
  const out = [];
  for (const block of String(body || '').split(/\n\s*\n/)) {
    const lines = block.split('\n').map(l => l.trim()).filter(Boolean);
    if (!lines.length) continue;
    if (lines.every(l => l.startsWith('- '))) {
      out.push(h('ul', { class: 'desk-wiki-list' },
        ...lines.map(l => {
          const item = h('li', {});
          let text = dashes(l.slice(2));
          const m = text.match(/\s*\[(risk_on|risk_off|neutral)\]\s*/);
          if (m) {
            text = (text.slice(0, m.index) + ' '
              + text.slice(m.index + m[0].length)).trim();
            item.append(...inlineMd(text), ' ',
              h('span', { class: 'c-pill neutral desk-wiki-tag', text: REGIME_TAGS[m[1]] }));
          } else {
            item.append(...inlineMd(text));
          }
          return item;
        })));
      continue;
    }
    // Headings can lead a block that also carries prose, so walk the lines
    // rather than classifying the block as a whole.
    let para = [];
    const flush = () => {
      if (!para.length) return;
      out.push(h('p', { class: 'desk-wiki-p' }, ...inlineMd(dashes(para.join(' ')))));
      para = [];
    };
    for (const line of lines) {
      const head = line.match(/^(#{1,4})\s+(.*)$/);
      if (head) {
        flush();
        out.push(h('div', { class: 'desk-wiki-h h' + Math.min(head[1].length, 3) },
          ...inlineMd(dashes(head[2]))));
      } else {
        para.push(line);
      }
    }
    flush();
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
          h('span', { text: fmtDateTimeET(j.t) + ' \u2014 the AI ' }),
          h('span', { class: 'desk-diary-kind' + (j.kind === 'pivot' ? ' pivot' : ''),
            text: DIARY_KIND[j.kind] || DIARY_KIND.note })),
        h('div', { class: 'desk-diary-title', text: j.title }),
        j.body ? h('p', { class: 'desk-diary-body t-dim', text: dashes(j.body) }) : null));
    }
    return;
  }
  const pages = wikiCache.pages || [];
  if (metaEl) metaEl.textContent = pages.length ? pages.length + ' page(s)' : '';
  if (!pages.length) {
    renderEmpty(el, 'The notebook is empty \u2014 lessons appear once real results come in.');
    return;
  }
  const built = pages.map(p => h('div', { class: 'desk-wiki-page' },
    h('div', { class: 'desk-wiki-head' },
      h('span', { class: 'desk-wiki-title',
        text: p.title || WIKI_TITLES[p.slug] || p.slug }),
      h('span', { class: 't-dim',
        title: 'Rewritten ' + p.revision
          + (p.revision === 1 ? ' time' : ' times') + ' as real results came in',
        text: timeAgo(p.updated_at) })),
    ...wikiBlocks(p.body)));
  for (const page of built) el.append(page);
  const more = capList(built, LESSONS_CAP, 'lesson page', 'lesson pages');
  if (more) el.append(more);
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

/* ── Claims registry: the structured facts behind the notebook, plus the
   owner-approval queue. Tier carries AUTHORITY: established/experimental
   claims may justify trades; candidates/observations are watch-only. Stats
   are recorded sample sizes — there is no confidence score to render, by
   design. ── */
const CLAIM_TIER = {
  established: { label: 'in force', pill: 'up' },
  candidate: { label: 'candidate — watch-only', pill: 'neutral' },
  observation: { label: 'observation — watch-only', pill: 'neutral' },
  digest: { label: 'digest', pill: 'neutral' },
};
const CLAIM_CLASS = {
  risk_rule: 'risk rule', market_strategy: 'market pattern',
  system_mechanics: 'system fact', operational: 'ops incident',
};

function claimStatsText(stats) {
  const bits = [];
  if (typeof stats.n === 'number') bits.push('n=' + stats.n);
  if (typeof stats.wins === 'number' && typeof stats.losses === 'number'
      && (stats.wins + stats.losses) > 0) {
    bits.push(stats.wins + 'W/' + stats.losses + 'L');
  }
  if (typeof stats.avg_alpha_pct === 'number') {
    bits.push('avg alpha ' + fmtPct(stats.avg_alpha_pct));
  }
  return bits.join(' · ');
}

async function loadClaims() {
  const el = document.getElementById('desk-claims');
  const metaEl = document.getElementById('desk-claims-meta');
  if (!el) return;
  skeleton(el);
  try {
    const [c, p] = await Promise.all([
      apiGet('/api/desk/claims').catch(() => null),
      apiGet('/api/desk/proposals').catch(() => null),
    ]);
    clear(el);
    const rows = (c && c.claims) || [];
    const pending = (p && p.proposals || []).filter(x => x.status === 'pending');
    if (metaEl) {
      const s = (c && c.summary) || {};
      metaEl.textContent = s.active
        ? s.active + ' active' + (pending.length
          ? ' · ' + pending.length + ' awaiting owner' : '')
        : '';
    }
    if (pending.length) {
      const strip = h('div', { class: 'desk-claims-approvals' },
        h('span', { class: 'c-pill warn', text: 'awaiting your approval' }));
      for (const pr of pending) {
        strip.append(h('span', { class: 'desk-claims-approval-item',
          text: pr.ref + ' [' + pr.change_kind + '] ' + pr.title }));
      }
      el.append(strip);
    }
    if (!rows.length) {
      renderEmpty(el, 'No claims registered yet — they appear as the AI turns measured results into structured facts.');
      return;
    }
    const cards = rows.map(r => {
      const tier = CLAIM_TIER[r.tier] || CLAIM_TIER.digest;
      const head = h('div', { class: 'desk-claim-head' },
        h('span', { class: 'desk-claim-cite t-dim', text: r.cite }),
        h('span', { class: 'c-pill ' + (r.experimental ? 'warn' : tier.pill),
          text: r.experimental ? 'experimental — size-capped' : tier.label }),
        h('span', { class: 'c-pill neutral',
          text: CLAIM_CLASS[r.kclass] || r.kclass }));
      if (r.expires_at) {
        head.append(h('span', { class: 't-dim desk-claim-exp',
          text: 'expires ' + r.expires_at + ' unless renewed' }));
      }
      const stats = claimStatsText(r.stats || {});
      const n = r.evidence_count;
      const foot = h('div', { class: 'desk-claim-foot t-dim' },
        h('span', { text: (stats ? stats + ' · ' : '')
          + n + (n === 1 ? ' piece of evidence' : ' pieces of evidence') }));
      return h('div', { class: 'desk-claim' },
        head,
        h('p', { class: 'desk-claim-statement', text: dashes(r.statement) }),
        foot);
    });
    for (const c of cards) el.append(c);
    const more = capList(cards, CLAIMS_CAP, 'claim', 'claims');
    if (more) el.append(more);
  } catch (err) { renderError(el, err, loadClaims); }
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

  // Honesty gate (v9.21.2). The server states `connected` on EVERY frame and
  // keeps emitting ~1/s even while the SIP socket is down — so frame arrival
  // alone proves nothing, and the arrival-based watchdog below never fired.
  // A dead tape read as LIVE for 13 days (2026-07-16 → 07-29) because this
  // function used to `return` on an empty quote set BEFORE touching the pill.
  // Decide the pill from the DATA, not from the fact a frame showed up.
  if (snap.connected === false || !syms.length) {
    setLivePill('delayed');
    return;
  }

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
/* Era-1 rows whose "fill" is a bookkeeping adjustment rather than an
   execution. An expiry settlement is NOT one of these — it happens at a real
   moment and keeps its stamp. */
const CORP_ACTION_SRC = new Set(['dividend', 'split_adjustment']);

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
        h('th', { text: 'When',
          title: 'When the fill executed, on the New York market clock (US/Eastern), to the second.' }),
        h('th', { text: 'Stock' }),
        h('th', { text: 'Side' }), h('th', { class: 'num', text: 'Shares' }),
        h('th', { class: 'num', text: 'Fill price' }),
        h('th', { class: 'num', text: 'Receipt', title: 'Era-2 fills are executed by the broker against the live market — the decision run is the receipt. Era-1 fills carry the old ledger\'s stamped bid/ask, session, and fee receipts.' }),
        h('th', { class: 'num', text: 'Value' }),
        h('th', { text: 'Why', title: 'The AI\'s stated reason for this trade at the time (era-1 fills; era-2 reasoning lives on the linked decision)' }))),
      h('tbody', {}, ...rows.map(r => {
        const q = r.fill_quote || {};
        const isOpt = OCC_RE_F.test(r.symbol);
        const buy = (r.side || '').toUpperCase() === 'BUY';
        const quoteCell = h('td', { class: 'num' },
          (q.bid != null && q.ask != null)
            ? h('span', { class: 't-dim', text: fmtPrice(q.bid) + ' / ' + fmtPrice(q.ask) })
            : h('span', { class: 't-dim', text: r.era === 2 ? 'broker fill' : '—' }));
        // Receipt extras: era-1 fills keep the old ledger's stamped session/
        // fee receipts; era-2 fills tag stops and the frozen era.
        const extras = [];
        if (r.era === 1) {
          extras.push(pill('ERA 1', 'neutral',
            'A fill from the pre-migration book (the hand-rolled ledger, frozen at cutover).'));
        }
        if (r.era === 2 && r.kind === 'stop') {
          extras.push(pill('STOP', 'warn',
            'This fill came from a resting protective stop on the broker’s book.'));
        }
        if (q.session && q.session !== 'regular') {
          extras.push(pill(String(q.session).toUpperCase(), 'neutral',
            'This fill booked outside regular trading hours.'));
        }
        if (q.fee && q.fee.total) {
          extras.push(h('span', { class: 't-dim',
            title: q.fee.contracts + ' contract(s) × ' + fmtPrice(q.fee.per_contract) + ' per-contract fee, included in the value',
            text: 'fee ' + fmtPrice(q.fee.total) }));
        }
        if (extras.length) quoteCell.append(h('div', { class: 'desk-fill-extra' }, ...extras));
        const why = (r.rationale || '').trim();
        const whyCell = why
          ? fillWhyCell(why)
          : (r.era === 2 && r.run_id
              ? h('td', { class: 'desk-fills-why t-dim',
                  title: 'The reasoning lives on the decision this fill belongs to — see “The latest decision” card’s History view.',
                  text: 'run ' + r.run_id })
              : h('td', { class: 't-dim', text: '—' }));
        // The ET clock time leads, with the session date and the relative
        // age beneath it. "2h ago" alone cannot be checked against anything;
        // the exact stamp is what lets a reader pull up the tape and confirm
        // the fill happened when the desk says it did. The era-1 corp-action
        // rows have no execution to stamp — their only instant is when the
        // old ledger booked the adjustment (same rule as /trades).
        const booked = CORP_ACTION_SRC.has(q.src);
        return h('tr', {},
          h('td', { class: 'desk-fill-when' },
            h('div', { text: booked ? '—' : fmtTimeET(r.t) }),
            h('div', { class: 'desk-fill-when-sub',
              text: fmtDateET(r.t) + ' · ' + timeAgo(r.t) })),
          h('td', {}, h('a', { href: '/symbol/' + (isOpt ? r.symbol.match(/^[A-Z]+/)[0] : r.symbol), class: 'c-link', text: r.symbol })),
          h('td', {}, pill((r.side || '').toUpperCase() || '—', buy ? 'up' : 'down')),
          h('td', { class: 'num', text: fmtNum(Math.abs(r.shares), isOpt ? 0 : 2) }),
          h('td', { class: 'num', text: fmtPrice(r.price) }),
          quoteCell,
          h('td', { class: 'num', text: fmtDollar(Math.abs(r.dollars)) }),
          whyCell);
      })));
    el.append(table);
    const more = capList([...table.querySelectorAll('tbody > tr')], FILLS_CAP,
      'fill', 'fills');
    if (more) el.append(more);
    // /trades is the real archive — every fill, both eras, with realized P&L
    // per row. This card is a recent-activity window, so say where the rest is
    // rather than growing to 200 rows pretending to be the ledger.
    el.append(h('p', { class: 'desk-fills-allnote t-dim' },
      h('a', { href: '/trades', class: 'c-link', text: 'The full trade history' }),
      h('span', { text: ' — every fill with the profit it realized.' })));
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
    loadLab(), loadOpenOrders(), loadPredictions(), loadClaims(),
  ]);
  refreshPeeks();
}

/* ── collapsed-card peek ──
   A collapsed card that shows only its title has not solved anything — it has
   moved the information behind a tap the reader has no reason to make. Each
   collapsed card therefore carries a one-line count of what is inside, so
   "Claims registry · 109 claims" is legible without opening it.

   The selector per card is EXPLICIT. A generic "count rows, else list items,
   else children" fallback was tried first and produced "1 wires", "1 picks",
   "1 notes" — it counted each renderer's single wrapper div — plus "2 marks"
   for an equity card whose own header already says 249. A confidently wrong
   count is worse than no count, so a card with no spec simply gets no peek. */
const PEEK_SPEC = {
  positions: { sel: 'tbody > tr', one: 'holding', many: 'holdings' },
  decision: { sel: '.desk-pick', one: 'pick', many: 'picks' },
  thinking: { sel: '.desk-feed-line', one: 'note', many: 'notes' },
  lab: { sel: '.desk-lab-entry', one: 'entry', many: 'entries' },
  // NOT here on purpose: equity, watch, predictions, wiki, claims and fills
  // each already populate a `#desk-<key>-meta` span in their header — "109
  // active", "35 open · 9 closed · 50% hit rate", "57 most recent". Those are
  // richer than any count this could compute, and adding a second number
  // wrapped a redundant line into every header, costing the exact vertical
  // space collapsing is meant to reclaim. The guard below enforces that
  // generally, so a card that GAINS a meta later silently stops duplicating.
};

function refreshPeeks() {
  for (const card of document.querySelectorAll('.desk-card[data-collapse-key]')) {
    const key = card.getAttribute('data-collapse-key');
    const spec = PEEK_SPEC[key];
    const header = card.querySelector('.c-card-header');
    const body = card.querySelector('.c-card-body');
    let peek = header && header.querySelector('.desk-card-peek');
    const drop = () => { if (peek) peek.remove(); };
    if (!spec || !header || !body) { drop(); continue; }
    // already says what is inside → no second opinion
    const meta = header.querySelector('[id$="-meta"]');
    if (meta && meta.textContent.trim()) { drop(); continue; }
    const n = body.querySelectorAll(spec.sel).length;
    if (!n) { drop(); continue; }
    if (!peek) {
      peek = h('span', { class: 'desk-card-peek' });
      header.append(peek);
    }
    peek.textContent = n + ' ' + (n === 1 ? spec.one : spec.many);
  }
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
/* Cards that arrive COLLAPSED on a phone.

   Much shorter than it was. Tabs (v10.3.0) already keep the long-form
   corpora — claims, notebook, lab, predictions — off the landing view
   entirely, and every list is now capped; collapsing a card the reader
   deliberately switched tabs to reach would hide it twice for no gain.

   What remains is the two cards on the Now tab that a phone glance does not
   need open: `orders`, whose header meta ("3 resting · 2 stops") already
   answers the only question a glance asks — is the protection armed — and
   `thinking`, which is prose.

   Left OPEN on purpose: decision (what it just did), equity (the curve),
   positions (the book) — what a phone glance is actually for. */
const MOBILE_COLLAPSED = ['orders', 'thinking'];
const MOBILE_Q = '(max-width: 768px)';

/* Rather than branch the collapse logic on viewport, stamp the SAME
   data-collapsed="1" attribute the template uses for opt-in defaults. Every
   downstream path — the persisted-preference check below, and the toggle
   handler's `'!' + key` "user explicitly opened this" record — then works
   unchanged. A parallel mobile code path would have had to re-implement both,
   and would have re-collapsed a card the reader had deliberately opened. */
function applyMobileCollapseDefaults() {
  if (!window.matchMedia(MOBILE_Q).matches) return;
  for (const key of MOBILE_COLLAPSED) {
    const card = document.querySelector(
      '.desk-card[data-collapse-key="' + key + '"]');
    if (card && !card.hasAttribute('data-collapsed')) {
      card.setAttribute('data-collapsed', '1');
    }
  }
}

/* Inject the ⓘ disclosure for each panel's plain-English explainer. Added in
   JS rather than the template so all ten card headers stay untouched and the
   button only ever exists where there is actually a .c-card-sub to reveal.
   CSS keeps it display:none above 768px, so desktop is unaffected. */
function wireInfoToggles() {
  for (const card of document.querySelectorAll('.desk-card')) {
    const header = card.querySelector('.c-card-header');
    const sub = card.querySelector('.c-card-sub');
    if (!header || !sub || header.querySelector('.desk-info-btn')) continue;
    const btn = h('button', {
      class: 'desk-info-btn', type: 'button', text: 'ⓘ',
      title: 'What is this panel?', 'aria-expanded': 'false',
      'aria-label': 'Explain this panel',
    });
    btn.addEventListener('click', ev => {
      ev.stopPropagation();
      const open = card.classList.toggle('desk-info-open');
      btn.setAttribute('aria-expanded', open ? 'true' : 'false');
    });
    // Sit next to the TITLE, not at the end of the header. Appended last it
    // landed after the meta text, and since the header wraps on a phone that
    // pushed a second 44px-tall line into every one of the ten card headers.
    const title = header.querySelector('.c-card-title, h3, h4');
    if (title) title.after(btn); else header.append(btn);
  }
}

function wireCollapse() {
  applyMobileCollapseDefaults();
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

/* ── tabs: one panel renders at a time ──
   These replaced anchor links in v10.3.0. The zones were always there, but
   scroll-jumping between them left the page 40,971px tall and put the nav
   itself 3.6 screens down. Panels stay POPULATED while hidden — every loader
   already runs on the shared interval — so switching is instant and costs no
   request. ── */
const TAB_KEY = 'ef-desk-tab-v1';
/* Old #zone-* deep links still land somewhere sensible. */
const LEGACY_HASH = {
  'desk-hero': 'panel-now',
  'zone-reasoning': 'panel-now',
  'zone-history': 'panel-learned',
};

function wireTabs() {
  const bar = document.getElementById('desk-tabs');
  if (!bar) return;
  const tabs = [...bar.querySelectorAll('.desk-tab')];
  if (!tabs.length) return;

  const show = (panelId, { focus = false, remember = true } = {}) => {
    let matched = false;
    for (const t of tabs) {
      const on = t.dataset.panel === panelId;
      matched = matched || on;
      t.classList.toggle('active', on);
      t.setAttribute('aria-selected', on ? 'true' : 'false');
      t.tabIndex = on ? 0 : -1;
      const panel = document.getElementById(t.dataset.panel);
      if (panel) panel.hidden = !on;
      if (on && focus) t.focus();
    }
    if (!matched) return false;
    if (remember) {
      try { localStorage.setItem(TAB_KEY, panelId); } catch (e) { /* private mode */ }
      history.replaceState(null, '', '#' + panelId);
    }
    // A chart or treemap built inside a hidden panel measured 0px wide.
    // Charts re-measure through their own ResizeObserver; the holdings card
    // is laid out imperatively and skips its repaint while hidden, so it
    // needs one now — with whatever the tape has folded in since.
    window.dispatchEvent(new Event('resize'));
    repaintPositions();
    redrawAllocation();
    return true;
  };

  bar.addEventListener('click', ev => {
    const t = ev.target.closest('.desk-tab');
    if (t) show(t.dataset.panel);
  });
  // Left/right arrows move between tabs, per the ARIA tablist pattern.
  bar.addEventListener('keydown', ev => {
    if (ev.key !== 'ArrowRight' && ev.key !== 'ArrowLeft') return;
    const i = tabs.findIndex(t => t.classList.contains('active'));
    if (i < 0) return;
    ev.preventDefault();
    const next = (i + (ev.key === 'ArrowRight' ? 1 : tabs.length - 1)) % tabs.length;
    show(tabs[next].dataset.panel, { focus: true });
  });

  const fromHash = (location.hash || '').slice(1);
  const wanted = LEGACY_HASH[fromHash] || fromHash;
  let stored = null;
  try { stored = localStorage.getItem(TAB_KEY); } catch (e) { /* private mode */ }
  // A hash wins over the remembered tab: it is an explicit request.
  if (!show(wanted, { remember: false })) show(stored || tabs[0].dataset.panel, { remember: false });
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
wireInfoToggles();
wireTabs();
wireSegs();
loadAll();
startTape();
// refresh the live panels periodically (the agent updates several times/day)
setInterval(() => {
  loadHeader(); loadThinking(); loadDecision(); loadWhatsNew(); loadWiki();
  loadLab(); loadOpenOrders(); loadPredictions(); loadClaims();
  // The BOOK moves intraday too — fills land, stops arm, the account read
  // changes. A tab left open across a trading hour must not keep serving
  // the load-time book: re-pull the positions reference (live ticks fold
  // into whatever book this cached), the fills card, and the equity curve
  // (the dashed live tip anchors to the curve's last real mark, so a stale
  // curve pins the tip to stale history).
  loadPositions(); loadFills(); loadEquity();
  // counts move as the agent works — a stale peek on a collapsed card is a
  // quietly wrong number, which is worse than no number
  setTimeout(refreshPeeks, 1500);
}, 60_000);

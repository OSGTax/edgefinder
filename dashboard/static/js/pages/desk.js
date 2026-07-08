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
let equitySeries = null;
let lastEquityData = [];

const ACTION_CLASS = { buy: 'up', add: 'up', hold: 'neutral', trim: 'warn', exit: 'down', sell: 'down' };

function pill(text, cls) {
  return h('span', { class: 'c-pill ' + (cls || 'neutral'), text });
}

function statCard(label, value, cls) {
  return h('div', { class: 'c-stat' },
    h('div', { class: 'label', text: label }),
    h('div', { class: 'value ' + (cls || ''), text: value }));
}

/* ── stats + strategy + regime headers ── */
async function loadHeader() {
  const statsEl = document.getElementById('desk-stats');
  const stratEl = document.getElementById('desk-strategy');
  const regimeEl = document.getElementById('desk-regime');
  skeleton(statsEl);
  try {
    const [pf, strat, regime] = await Promise.all([
      apiGet('/api/desk/portfolio'),
      apiGet('/api/desk/strategy'),
      apiGet('/api/desk/regime').catch(() => null),
    ]);

    clear(statsEl);
    const pnlCls = pf.total_pnl >= 0 ? 't-up' : 't-down';
    statsEl.append(
      statCard('Account value', fmtDollar(pf.equity)),
      statCard('Profit / loss', fmtPnl(pf.total_pnl), pnlCls),
      statCard('Return', fmtPct(pf.total_return_pct / 100, { signed: true }), pnlCls),
      statCard('Cash on hand', fmtDollar(pf.cash)),
      statCard('Investments', String((pf.positions || []).length)),
    );

    clear(stratEl);
    if (strat.current) {
      stratEl.append(h('span', {
        class: 'c-pill info',
        title: strat.current.thesis || 'The strategy the AI is currently running.',
        text: 'Strategy v' + strat.current.version + ' · ' + strat.current.name,
      }));
    }

    clear(regimeEl);
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
      regimeEl.append(h('span', { class: 'c-pill ' + cls, title: tip, text: label }));
    }
  } catch (err) {
    renderError(statsEl, err, loadHeader);
  }
}

/* ── equity curve ── */
function ensureEquityChart() {
  if (equityChart) return;
  const el = document.getElementById('desk-equity-chart');
  equityChart = createChart(el, { height: 320 });
  const c = colors();
  equitySeries = equityChart.addAreaSeries({
    lineColor: c.accent, topColor: c.accent + '55', bottomColor: c.accent + '08',
    lineWidth: 2, priceLineVisible: false,
  });
  onThemeChange(() => {
    const cc = colors();
    equitySeries.applyOptions({ lineColor: cc.accent, topColor: cc.accent + '55', bottomColor: cc.accent + '08' });
  });
}

async function loadEquity() {
  const metaEl = document.getElementById('desk-equity-meta');
  try {
    const series = await apiGet('/api/desk/equity?limit=2000');
    if (!series.length) {
      metaEl.textContent = 'no marks yet';
      return;
    }
    ensureEquityChart();
    lastEquityData = series.map(p => ({ time: toEpochSec(p.t), value: p.equity }))
      .filter(p => p.time);
    // de-dup identical timestamps (chart requires strictly increasing time)
    const seen = new Set();
    const data = [];
    for (const p of lastEquityData) { if (!seen.has(p.time)) { seen.add(p.time); data.push(p); } }
    equitySeries.setData(data);
    equityChart.timeScale().fitContent();
    const last = series[series.length - 1];
    metaEl.textContent = `${fmtDollar(last.equity)} · ${series.length} marks`;
  } catch (err) {
    metaEl.textContent = 'error loading curve';
  }
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
      h('td', {}, h('a', { href: '/symbol/' + p.symbol, class: 'c-link', text: p.symbol })),
      h('td', { class: 'num', text: fmtNum(p.shares, 2) }),
      h('td', { class: 'num', text: fmtPrice(p.avg_price) }),
      h('td', { class: 'num', text: fmtPrice(p.last_price) }),
      dayChangeCell(stats[p.symbol]),
      trendCell(stats[p.symbol]),
      h('td', { class: 'num', text: fmtDollar(p.market_value) }),
      h('td', { class: 'num', text: fmtPct(p.weight) }),
      h('td', { class: 'num ' + (p.unrealized_pnl >= 0 ? 't-up' : 't-down'),
        text: fmtPnl(p.unrealized_pnl) })))));
}

function optionsTable(rows) {
  return h('table', { class: 'c-table' },
    h('thead', {}, h('tr', {},
      h('th', { text: 'Option contract', title: 'e.g. "NVDA $200C 2027-01-16" = the right to buy NVDA at $200 until Jan 16 2027' }),
      h('th', { text: 'Side', title: 'LONG = the AI bought it. SHORT = the AI sold it (always backed by shares, cash, or another option — never naked)' }),
      h('th', { class: 'num', text: 'Qty', title: 'Contracts held; each contract covers 100 shares' }),
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

async function loadPositions() {
  const el = document.getElementById('desk-positions');
  skeleton(el);
  try {
    const [pf, hs] = await Promise.all([
      apiGet('/api/desk/portfolio'),
      apiGet('/api/desk/holding-stats').catch(() => null),
    ]);
    if (!pf.positions.length) { renderEmpty(el, 'All cash — no open positions.'); return; }
    clear(el);
    const stats = (hs && hs.symbols) || {};
    const eqs = pf.positions.filter(p => !occParse(p.symbol));
    const opts = pf.positions.filter(p => occParse(p.symbol));
    const donut = allocation(pf);
    if (donut) el.append(donut);
    if (eqs.length) el.append(equitiesTable(eqs, stats));
    if (opts.length) {
      el.append(h('div', { class: 'desk-subhead', text: 'Options' }), optionsTable(opts));
    }
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
    const feed = h('div', { class: 'desk-feed' });
    for (const line of data.lines) {
      feed.append(h('div', { class: 'desk-feed-line' },
        h('span', { class: 'desk-feed-phase', text: line.phase || '·' }),
        h('span', { class: 'desk-feed-text', text: line.text }),
        h('span', { class: 'desk-feed-time t-dim', text: timeAgo(line.t) })));
    }
    el.append(feed);
  } catch (err) { renderError(el, err, loadThinking); }
}

/* ── latest decision: picks + watchlist ── */
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
      for (const p of d.picks) {
        const action = (p.action || '').toLowerCase();
        const card = h('div', { class: 'desk-pick c-card' },
          h('div', { class: 'desk-pick-head' },
            h('a', { href: '/symbol/' + p.symbol, class: 'desk-pick-sym', text: p.symbol }),
            pill((p.action || '').toUpperCase() || '—', ACTION_CLASS[action] || 'neutral'),
            p.why_now ? h('span', { class: 'desk-pick-why t-dim', text: p.why_now }) : null),
          p.rationale ? h('p', { class: 'desk-pick-rationale', text: p.rationale }) : null);
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
        picksEl.append(card);
      }
    }

    clear(wlEl);
    if (d.watchlist && d.watchlist.length) {
      const chips = h('div', { class: 'c-chips' },
        h('span', { class: 't-dim', text: 'Watchlist: ' }),
        ...d.watchlist.map(w => h('span', { class: 'c-chip' },
          h('a', { href: '/symbol/' + (w.symbol || w), class: 'c-link', text: (w.symbol || w) }),
          w.note ? h('span', { class: 't-dim', text: ' — ' + w.note }) : null)));
      wlEl.append(chips);
    }
  } catch (err) { renderError(picksEl, err, loadDecision); }
}

/* ── backtest evidence ── */
async function loadBacktests() {
  const el = document.getElementById('desk-backtests');
  skeleton(el);
  try {
    const rows = await apiGet('/api/desk/backtests?limit=15');
    if (!rows.length) { renderEmpty(el, 'No backtests run yet.'); return; }
    clear(el);
    const table = h('table', { class: 'c-table' },
      h('thead', {}, h('tr', {},
        h('th', { text: 'Idea tested' }),
        h('th', { class: 'num', text: 'Return', title: 'What the idea would have made over the test period' }),
        h('th', { class: 'num', text: 'vs the market', title: 'How much better (+) or worse (−) than simply buying the S&P 500' }),
        h('th', { class: 'num', text: 'Smoothness', title: 'Sharpe ratio: return relative to how bumpy the ride was — higher is better' }),
        h('th', { class: 'num', text: 'Worst dip', title: 'The deepest peak-to-bottom drop along the way' }))),
      h('tbody', {}, ...rows.map(r => {
        const res = r.result || {};
        const ex = res.excess_return_pct;
        return h('tr', {},
          h('td', {}, h('div', { text: r.label }), h('div', { class: 't-dim desk-bt-when', text: timeAgo(r.t) })),
          h('td', { class: 'num', text: res.return_pct != null ? fmtNum(res.return_pct, 1) + '%' : '—' }),
          h('td', { class: 'num ' + (ex >= 0 ? 't-up' : 't-down'), text: ex != null ? (ex >= 0 ? '+' : '') + fmtNum(ex, 1) + '%' : '—' }),
          h('td', { class: 'num', text: res.sharpe != null ? fmtNum(res.sharpe, 2) : '—' }),
          h('td', { class: 'num', text: res.max_drawdown_pct != null ? fmtNum(res.max_drawdown_pct, 1) + '%' : '—' }));
      })));
    el.append(table);
  } catch (err) { renderError(el, err, loadBacktests); }
}

/* ── strategy journal ── */
/* ── lessons wiki: what the AI has learned (curated, size-capped) ── */
const WIKI_TITLES = {
  playbook: 'Playbook', lessons: 'Lessons', mistakes: 'Mistakes',
  'market-notes': 'Market notes',
};

function wikiBlocks(body) {
  // markdown-lite: blank-line-separated blocks; "- " blocks → bullet lists,
  // everything else → paragraphs. All text nodes — zero innerHTML.
  const out = [];
  for (const block of String(body || '').split(/\n\s*\n/)) {
    const lines = block.split('\n').map(l => l.trim()).filter(Boolean);
    if (!lines.length) continue;
    if (lines.every(l => l.startsWith('- '))) {
      out.push(h('ul', { class: 'desk-wiki-list' },
        ...lines.map(l => h('li', { text: l.slice(2) }))));
    } else {
      out.push(h('p', { class: 'desk-wiki-p', text: lines.join(' ') }));
    }
  }
  return out;
}

async function loadWiki() {
  const el = document.getElementById('desk-wiki');
  const metaEl = document.getElementById('desk-wiki-meta');
  if (!el) return;
  skeleton(el);
  try {
    const data = await apiGet('/api/desk/wiki');
    const pages = data.pages || [];
    if (!pages.length) {
      renderEmpty(el, 'The notebook is empty — lessons appear once real results come in.');
      metaEl.textContent = '';
      return;
    }
    metaEl.textContent = pages.length + ' page(s)';
    clear(el);
    for (const p of pages) {
      el.append(h('div', { class: 'desk-wiki-page' },
        h('div', { class: 'desk-wiki-head' },
          h('span', { class: 'desk-wiki-title',
            text: p.title || WIKI_TITLES[p.slug] || p.slug }),
          pill('rev ' + p.revision, 'info'),
          h('span', { class: 't-dim', text: timeAgo(p.updated_at) })),
        ...wikiBlocks(p.body)));
    }
  } catch (err) { renderError(el, err, loadWiki); }
}

async function loadJournal() {
  const el = document.getElementById('desk-journal');
  skeleton(el);
  try {
    const data = await apiGet('/api/desk/strategy');
    const journal = data.journal || [];
    if (!journal.length) { renderEmpty(el, 'No pivots yet — strategy is fresh.'); return; }
    clear(el);
    const list = h('div', { class: 'desk-journal' });
    for (const j of journal) {
      list.append(h('div', { class: 'desk-journal-entry' },
        h('div', { class: 'desk-journal-head' },
          pill(j.kind, j.kind === 'pivot' ? 'warn' : 'info'),
          h('span', { class: 'desk-journal-title', text: j.title }),
          h('span', { class: 't-dim', text: timeAgo(j.t) })),
        j.body ? h('p', { class: 'desk-journal-body t-dim', text: j.body }) : null));
    }
    el.append(list);
  } catch (err) { renderError(el, err, loadJournal); }
}

/* ── live tape: real-time SIP quotes over SSE ── */
let tapePrev = {};        // symbol -> last mid (for up/down tick coloring)
let tapeLastEvent = 0;    // client-side staleness watchdog

function renderTape(snap) {
  const el = document.getElementById('desk-tape');
  const statusEl = document.getElementById('desk-tape-status');
  const liveDot = document.getElementById('desk-tape-live');
  const banner = document.getElementById('desk-tape-banner');
  const quotes = snap.quotes || {};
  const syms = Object.keys(quotes).sort();
  if (!syms.length) {
    renderEmpty(el, 'No live quotes yet — streamer warming up.');
    statusEl.textContent = snap.connected ? 'connected' : 'waiting for stream';
    liveDot.hidden = true;
    return;
  }
  const anyFresh = syms.some(s => !quotes[s].stale);
  liveDot.hidden = !(snap.connected && anyFresh);
  statusEl.textContent = snap.connected
    ? `${syms.length} symbols · SIP live`
    : 'stream reconnecting — quotes may be stale';
  banner.hidden = snap.connected || anyFresh;
  if (!banner.hidden) {
    clear(banner);
    banner.append(h('span', { text: '⚠ Live stream interrupted — prices below are stale and nothing will trade off them. Reconnecting…' }));
  }

  clear(el);
  fillOptSelect(syms);
  const table = h('table', { class: 'c-table desk-tape-table' },
    h('thead', {}, h('tr', {},
      h('th', { text: 'Symbol' }), h('th', { class: 'num', text: 'Bid' }),
      h('th', { class: 'num', text: 'Ask' }), h('th', { class: 'num', text: 'Mid' }),
      h('th', { class: 'num', text: 'Last' }), h('th', { class: 'num', text: 'Age' }))),
    h('tbody', {}, ...syms.map(s => {
      const q = quotes[s];
      const dir = q.mid != null && tapePrev[s] != null
        ? (q.mid > tapePrev[s] ? 'up' : q.mid < tapePrev[s] ? 'down' : '') : '';
      if (q.mid != null) tapePrev[s] = q.mid;
      const ageTxt = q.age_secs == null ? '—' : q.age_secs < 2 ? 'live' : `${Math.round(q.age_secs)}s`;
      return h('tr', { class: q.stale ? 'desk-tape-stale' : '' },
        h('td', {}, h('a', { href: '/symbol/' + s, class: 'c-link', text: s })),
        h('td', { class: 'num', text: q.bid != null ? fmtPrice(q.bid) : '—' }),
        h('td', { class: 'num', text: q.ask != null ? fmtPrice(q.ask) : '—' }),
        h('td', { class: 'num ' + (dir === 'up' ? 't-up' : dir === 'down' ? 't-down' : ''),
          text: q.mid != null ? fmtPrice(q.mid) : '—' }),
        h('td', { class: 'num', text: q.last != null ? fmtPrice(q.last) : '—' }),
        h('td', { class: 'num ' + (q.stale ? 't-down' : 't-dim'), text: q.stale ? `stale ${ageTxt}` : ageTxt }));
    })));
  el.append(table);
}

function startTape() {
  const statusEl = document.getElementById('desk-tape-status');
  let es;
  const connect = () => {
    es = new EventSource('/api/desk/stream');
    es.addEventListener('quotes', ev => {
      tapeLastEvent = Date.now();
      try { renderTape(JSON.parse(ev.data)); } catch (e) { /* skip bad frame */ }
    });
    es.onerror = () => { statusEl.textContent = 'stream lost — reconnecting…'; };
  };
  connect();
  // client-side watchdog: EventSource auto-reconnects, but surface the gap
  setInterval(() => {
    if (tapeLastEvent && Date.now() - tapeLastEvent > 6000) {
      statusEl.textContent = 'stream lost — reconnecting…';
      document.getElementById('desk-tape-live').hidden = true;
    }
  }, 3000);
}

/* ── options radar: live chain summary + the growing IV bank ── */
let optSymbol = localStorage.getItem('ef-opt-symbol') || 'SPY';
let optSelectFilled = false;

function fillOptSelect(tapeSymbols) {
  // free-text picker: ANY optionable symbol works (the endpoint is on-demand
  // REST); the tape just seeds the suggestion list
  if (optSelectFilled || !tapeSymbols.length) return;
  const input = document.getElementById('desk-opt-symbol');
  const list = document.getElementById('desk-opt-list');
  if (!input || !list) return;
  clear(list);
  for (const s of tapeSymbols.filter(s => !OCC_RE.test(s)).sort()) {
    list.append(h('option', { value: s }));
  }
  input.value = optSymbol;
  input.addEventListener('change', () => {
    const sym = input.value.trim().toUpperCase();
    if (!/^[A-Z]{1,6}$/.test(sym)) { input.value = optSymbol; return; }
    input.value = sym;
    optSymbol = sym;
    localStorage.setItem('ef-opt-symbol', optSymbol);
    loadOptions();
  });
  optSelectFilled = true;
}

function chainRows(summary, maxRows = 21) {
  // merge calls/puts by strike, then keep the strikes nearest ATM (dense
  // chains like SPY have 130+ strikes in ±10% — a skyscraper, not a table)
  const byStrike = new Map();
  for (const c of summary.calls_table || []) byStrike.set(c.strike, { c });
  for (const p of summary.puts_table || []) {
    byStrike.set(p.strike, { ...(byStrike.get(p.strike) || {}), p });
  }
  const atm = summary.atm_strike ?? summary.spot;
  return [...byStrike.entries()]
    .sort((a, b) => Math.abs(a[0] - atm) - Math.abs(b[0] - atm))
    .slice(0, maxRows)
    .sort((a, b) => a[0] - b[0]);
}

function optStat(label, value, cls, tip) {
  return h('div', { class: 'c-stat', title: tip || null },
    h('div', { class: 'label' + (tip ? ' desk-hint' : ''), text: label }),
    h('div', { class: 'value ' + (cls || ''), text: value }));
}

async function loadOptions() {
  const el = document.getElementById('desk-options');
  const metaEl = document.getElementById('desk-opt-meta');
  if (!el) return;
  skeleton(el);
  try {
    const [s, hist] = await Promise.all([
      apiGet('/api/desk/options/' + optSymbol),
      apiGet('/api/desk/options/' + optSymbol + '/history').catch(() => null),
    ]);
    clear(el);
    if (!s.available) {
      renderEmpty(el, 'Options data unavailable' + (s.error ? ` (${s.error})` : '') + '.');
      metaEl.textContent = '';
      return;
    }
    metaEl.textContent = `expiry ${s.expiry} · ${s.dte} DTE`;
    const em = s.expected_move_pct != null
      ? `±${fmtNum(s.expected_move_pct, 1)}% ($${fmtNum(s.expected_move_dollars, 2)})` : '—';
    const stats = h('div', { class: 'pf-grid mb-16' },
      optStat('Stock price', fmtPrice(s.spot),
        '', 'The current market price of the stock itself'),
      optStat('Implied volatility', s.atm_iv != null ? fmtNum(s.atm_iv * 100, 1) + '%' : '—',
        '', 'How jumpy the options market expects this stock to be — higher = bigger expected swings (and pricier options)'),
      optStat('Expected move', em,
        '', `The size of the price swing (up OR down) that options traders are pricing in between now and ${s.expiry}`),
      optStat('Fear gauge', s.skew_25d != null
        ? (s.skew_25d >= 0 ? '+' : '') + fmtNum(s.skew_25d * 100, 1) + '%'
        : '—', s.skew_25d > 0.02 ? 't-down' : '',
        'Put/call skew: positive = traders paying extra for downside protection (nervous); near zero = relaxed'),
      optStat('History collected', `${(hist && hist.series ? hist.series.length : 0)} day(s)`,
        '', 'Days of volatility history saved so far — charts get richer as this grows'));
    el.append(stats);

    const rows = chainRows(s);
    if (rows.length) {
      el.append(h('div', { class: 'desk-opt-chainwrap' },
        h('table', { class: 'c-table desk-opt-chain' },
          h('thead', {}, h('tr', {},
            h('th', { class: 'num', text: 'C bid' }), h('th', { class: 'num', text: 'C ask' }),
            h('th', { class: 'num', text: 'C IV' }), h('th', { class: 'num', text: 'CΔ' }),
            h('th', { class: 'num desk-opt-strike', text: 'Strike' }),
            h('th', { class: 'num', text: 'PΔ' }), h('th', { class: 'num', text: 'P IV' }),
            h('th', { class: 'num', text: 'P bid' }), h('th', { class: 'num', text: 'P ask' }))),
          h('tbody', {}, ...rows.map(([strike, { c, p }]) => {
            const atm = strike === s.atm_strike;
            const f = (v, d = 2) => v != null ? fmtNum(v, d) : '—';
            return h('tr', { class: atm ? 'desk-opt-atm' : '' },
              h('td', { class: 'num', text: f(c && c.bid) }),
              h('td', { class: 'num', text: f(c && c.ask) }),
              h('td', { class: 'num', text: c && c.iv != null ? f(c.iv * 100, 1) + '%' : '—' }),
              h('td', { class: 'num t-dim', text: f(c && c.delta) }),
              h('td', { class: 'num desk-opt-strike', text: fmtNum(strike, strike % 1 ? 2 : 0) }),
              h('td', { class: 'num t-dim', text: f(p && p.delta) }),
              h('td', { class: 'num', text: p && p.iv != null ? f(p.iv * 100, 1) + '%' : '—' }),
              h('td', { class: 'num', text: f(p && p.bid) }),
              h('td', { class: 'num', text: f(p && p.ask) }));
          })))));
    }
  } catch (err) { renderError(el, err, loadOptions); }
}

/* ── market movers: gainers / losers / most-active (last close) ── */
function moverColumn(title, rows) {
  const col = h('div', { class: 'desk-mv-col' },
    h('div', { class: 'desk-mv-col-head', text: title }));
  if (!rows || !rows.length) {
    col.append(h('div', { class: 't-dim desk-mv-empty', text: '—' }));
    return col;
  }
  for (const r of rows) {
    const chg = r.change_pct;
    const cls = chg == null ? 't-dim' : chg >= 0 ? 't-up' : 't-down';
    const chgTxt = chg == null ? '—' : (chg >= 0 ? '+' : '') + fmtNum(chg, 2) + '%';
    col.append(h('div', { class: 'desk-mv-row' },
      h('a', { href: '/symbol/' + r.symbol, class: 'c-link desk-mv-sym', text: r.symbol }),
      h('span', { class: 'desk-mv-price t-dim', text: fmtPrice(r.close) }),
      h('span', { class: 'desk-mv-chg ' + cls, text: chgTxt })));
  }
  return col;
}

async function loadMovers() {
  const el = document.getElementById('desk-movers');
  const metaEl = document.getElementById('desk-movers-meta');
  skeleton(el);
  try {
    const d = await apiGet('/api/desk/movers?top=5');
    const empty = !d.as_of || (!(d.gainers || []).length && !(d.losers || []).length && !(d.most_active || []).length);
    if (empty) { renderEmpty(el, 'No market data yet.'); if (metaEl) metaEl.textContent = ''; return; }
    if (metaEl) metaEl.textContent = 'as of ' + d.as_of + ' close';
    clear(el);
    el.append(h('div', { class: 'desk-mv-grid' },
      moverColumn('Top gainers', d.gainers),
      moverColumn('Top losers', d.losers),
      moverColumn('Most active', d.most_active)));
  } catch (err) { renderError(el, err, loadMovers); }
}

/* ── recent fills: each trade + the live bid/ask it priced off ── */
const OCC_RE_F = /^[A-Z]{1,6}\d{6}[CP]\d{8}$/;

async function loadFills() {
  const el = document.getElementById('desk-fills');
  const metaEl = document.getElementById('desk-fills-meta');
  if (!el) return;
  skeleton(el);
  try {
    const rows = await apiGet('/api/desk/trades?limit=20');
    if (!rows.length) { renderEmpty(el, 'No trades yet.'); if (metaEl) metaEl.textContent = ''; return; }
    if (metaEl) metaEl.textContent = rows.length + ' most recent';
    clear(el);
    const table = h('table', { class: 'c-table' },
      h('thead', {}, h('tr', {},
        h('th', { text: 'When' }), h('th', { text: 'Stock' }),
        h('th', { text: 'Side' }), h('th', { class: 'num', text: 'Shares' }),
        h('th', { class: 'num', text: 'Fill price' }),
        h('th', { class: 'num', text: 'Live bid / ask', title: 'The real-time bid and ask the fill priced against at that moment' }),
        h('th', { class: 'num', text: 'Value' }))),
      h('tbody', {}, ...rows.map(r => {
        const q = r.fill_quote || {};
        const isOpt = OCC_RE_F.test(r.symbol);
        const buy = (r.side || '').toUpperCase() === 'BUY';
        const quote = (q.bid != null && q.ask != null)
          ? h('span', { class: 't-dim', text: fmtPrice(q.bid) + ' / ' + fmtPrice(q.ask) })
          : h('span', { class: 't-dim', text: '—' });
        return h('tr', {},
          h('td', { class: 't-dim', text: timeAgo(r.t) }),
          h('td', {}, h('a', { href: '/symbol/' + (isOpt ? r.symbol.match(/^[A-Z]+/)[0] : r.symbol), class: 'c-link', text: r.symbol })),
          h('td', {}, pill((r.side || '').toUpperCase() || '—', buy ? 'up' : 'down')),
          h('td', { class: 'num', text: fmtNum(Math.abs(r.shares), isOpt ? 0 : 2) }),
          h('td', { class: 'num', text: fmtPrice(r.price) }),
          h('td', { class: 'num' }, quote),
          h('td', { class: 'num', text: fmtDollar(Math.abs(r.dollars)) }));
      })));
    el.append(table);
  } catch (err) { renderError(el, err, loadFills); }
}

/* ── dividend calendar: per-holding ex-dates + amounts ── */
async function loadDividends() {
  const el = document.getElementById('desk-dividends');
  const metaEl = document.getElementById('desk-dividends-meta');
  if (!el) return;
  skeleton(el);
  try {
    const d = await apiGet('/api/desk/dividends');
    const payers = (d.holdings || []).filter(x => x.has_dividend);
    if (!payers.length) {
      renderEmpty(el, 'None of the current holdings pay a dividend.');
      if (metaEl) metaEl.textContent = '';
      return;
    }
    if (metaEl) metaEl.textContent = payers.length + ' of ' + (d.holdings || []).length + ' holdings pay';
    clear(el);
    const table = h('table', { class: 'c-table' },
      h('thead', {}, h('tr', {},
        h('th', { text: 'Stock' }),
        h('th', { class: 'num', text: 'Last paid', title: 'The most recent ex-dividend date — the cutoff to own the stock and receive that payment' }),
        h('th', { class: 'num', text: 'Amount', title: 'Cash per share of the most recent dividend' }),
        h('th', { class: 'num', text: 'Next', title: 'The next scheduled ex-dividend date, if one has been announced' }),
        h('th', { class: 'num', text: 'Est. yearly', title: 'Estimated payout per share over a year (sum of the last four dividends)' }))),
      h('tbody', {}, ...payers.map(x => h('tr', {},
        h('td', {}, h('a', { href: '/symbol/' + x.symbol, class: 'c-link', text: x.symbol })),
        h('td', { class: 'num', text: x.last_ex_date || '—' }),
        h('td', { class: 'num', text: x.last_amount != null ? fmtPrice(x.last_amount) : '—' }),
        h('td', { class: 'num ' + (x.next_ex_date ? 't-up' : 't-dim'), text: x.next_ex_date || '—' }),
        h('td', { class: 'num', text: x.ttm_amount ? fmtPrice(x.ttm_amount) : '—' })))));
    el.append(table);
  } catch (err) { renderError(el, err, loadDividends); }
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
    loadDecision(), loadBacktests(), loadJournal(), loadWhatsNew(),
    loadOptions(), loadMovers(), loadDividends(), loadFills(), loadWiki(),
  ]);
}

loadAll();
startTape();
// refresh the live panels periodically (the agent updates several times/day)
setInterval(() => { loadHeader(); loadThinking(); loadDecision(); loadWhatsNew(); loadOptions(); loadWiki(); }, 60_000);

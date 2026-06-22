/* Trading Desk — the autonomous agent's window.
   Four panels: (stats + equity curve), holdings, live thinking feed,
   the latest decision (picks with why-now / rationale / news + watchlist),
   and the evidence (backtests it ran) + strategy journal. Reads the new
   /api/desk/* endpoints; reuses the core charts/dom/fmt/net modules. */

import { apiGet } from '../core/net.js';
import { toEpochSec, fmtDollar, fmtPnl, fmtPct, fmtPrice, fmtNum, timeAgo, fmtDateTimeET }
  from '../core/fmt.js';
import { h, clear, skeleton, renderEmpty, renderError } from '../core/dom.js';
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
      statCard('Equity', fmtDollar(pf.equity)),
      statCard('Total P&L', fmtPnl(pf.total_pnl), pnlCls),
      statCard('Return', fmtPct(pf.total_return_pct / 100, { signed: true }), pnlCls),
      statCard('Cash', fmtDollar(pf.cash)),
      statCard('Positions', String((pf.positions || []).length)),
    );

    clear(stratEl);
    if (strat.current) {
      stratEl.append(
        pill('v' + strat.current.version + ' · ' + strat.current.name, 'info'));
    }

    clear(regimeEl);
    if (regime && regime.tag) {
      const cls = regime.tag === 'risk_on' ? 'up' : regime.tag === 'risk_off' ? 'down' : 'neutral';
      regimeEl.append(pill('Regime: ' + regime.tag.replace('_', ' '), cls));
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

/* ── holdings ── */
async function loadPositions() {
  const el = document.getElementById('desk-positions');
  skeleton(el);
  try {
    const pf = await apiGet('/api/desk/portfolio');
    if (!pf.positions.length) { renderEmpty(el, 'All cash — no open positions.'); return; }
    clear(el);
    const table = h('table', { class: 'c-table' },
      h('thead', {}, h('tr', {},
        h('th', { text: 'Symbol' }), h('th', { class: 'num', text: 'Shares' }),
        h('th', { class: 'num', text: 'Avg' }), h('th', { class: 'num', text: 'Last' }),
        h('th', { class: 'num', text: 'Value' }), h('th', { class: 'num', text: 'Wt' }),
        h('th', { class: 'num', text: 'Unreal. P&L' }))),
      h('tbody', {}, ...pf.positions.map(p => h('tr', {},
        h('td', {}, h('a', { href: '/symbol/' + p.symbol, class: 'c-link', text: p.symbol })),
        h('td', { class: 'num', text: fmtNum(p.shares, 0) }),
        h('td', { class: 'num', text: fmtPrice(p.avg_price) }),
        h('td', { class: 'num', text: fmtPrice(p.last_price) }),
        h('td', { class: 'num', text: fmtDollar(p.market_value) }),
        h('td', { class: 'num', text: fmtPct(p.weight) }),
        h('td', { class: 'num ' + (p.unrealized_pnl >= 0 ? 't-up' : 't-down'),
          text: fmtPnl(p.unrealized_pnl) })))));
    el.append(table);
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
        h('th', { text: 'Idea' }), h('th', { class: 'num', text: 'Return' }),
        h('th', { class: 'num', text: 'vs SPY' }), h('th', { class: 'num', text: 'Sharpe' }),
        h('th', { class: 'num', text: 'MaxDD' }))),
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

async function loadAll() {
  await Promise.all([
    loadHeader(), loadEquity(), loadPositions(), loadThinking(),
    loadDecision(), loadBacktests(), loadJournal(),
  ]);
}

loadAll();
// refresh the live panels periodically (the agent updates several times/day)
setInterval(() => { loadHeader(); loadThinking(); loadDecision(); }, 60_000);

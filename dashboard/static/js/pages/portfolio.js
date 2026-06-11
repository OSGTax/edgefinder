/* Portfolio command center — hero stats from the SERVER-computed /summary
   (no fabricated capital, ever), equity + comparison charts on epoch time,
   P&L calendar heatmap, live-proof scorecard, open positions, market strip,
   ops badge. Everything is v2 now (the old arena lane was retired). */

import { apiGet } from '../core/net.js';
import { fmtDollar, fmtPnl, fmtPct, fmtNum, fmtPrice, fmtInt, upDownClass, fmtDate } from '../core/fmt.js';
import { h, clear, renderError, renderEmpty, panel } from '../core/dom.js';
import { createChart, colors } from '../core/charts.js';
import { onThemeChange } from '../core/theme.js';
import { calendarHeatmap } from '../components/heatmap.js';
import { poller } from '../core/poll.js';

const state = { eqDays: 90, meta: new Map() };
let eqChart = null;
let cmpChart = null;

/* ── hero stats (server summary only) ── */
async function loadHero() {
  const el = document.getElementById('pf-hero');
  await panel(el, () => apiGet('/api/strategies/summary'), (host, lanes) => {
    const L = lanes.all;
    const stat = (label, value, cls = '', sub = null) => h('div', { class: 'c-stat' },
      h('div', { class: 'label', text: label }),
      h('div', { class: `value hero num ${cls}`, text: value }),
      sub != null ? h('div', { class: 'delta t-dim', text: sub }) : null);
    clear(host).append(
      stat('Total equity', fmtDollar(L.total_equity), '',
           `${L.strategies} strategies · ${fmtDollar(L.starting_capital)} start`),
      stat('Total P&L', fmtPnl(L.total_pnl), upDownClass(L.total_pnl),
           L.starting_capital ? fmtPct(L.total_pnl / L.starting_capital * 100) : null),
      stat('Day P&L', L.day_pnl != null ? fmtPnl(L.day_pnl) : '—', upDownClass(L.day_pnl)),
      stat('Open / Win rate', `${fmtInt(L.open_positions)} · ${L.win_rate != null ? L.win_rate + '%' : '—'}`),
    );
  });
}

/* ── equity chart (sum of all strategies, epoch time) ── */
async function loadEquity() {
  const host = document.getElementById('pf-equity');
  host.replaceChildren();
  let data;
  try {
    data = await apiGet(`/api/strategies/equity-curve?days=${state.eqDays}`);
  } catch (err) { renderError(host, err, loadEquity); return; }

  const names = Object.keys(data);
  const byTime = new Map();
  for (const name of names) {
    for (const p of data[name] || []) {
      byTime.set(p.time, (byTime.get(p.time) || 0) + (p.total_equity || 0));
    }
  }
  const points = [...byTime.entries()].sort((a, b) => a[0] - b[0])
    .map(([time, value]) => ({ time, value }));
  if (!points.length) { renderEmpty(host, 'No equity snapshots yet'); return; }

  eqChart?.__efDestroy?.();
  const c = colors();
  eqChart = createChart(host);
  const area = eqChart.addAreaSeries({
    lineColor: c.accent, topColor: c.accent + '33', bottomColor: 'transparent', lineWidth: 2,
  });
  area.setData(points);
  eqChart.timeScale().fitContent();
}

/* ── comparison chart (% return per strategy + SPY) ── */
async function loadComparison() {
  const host = document.getElementById('pf-compare');
  host.replaceChildren();
  let curves, bench;
  try {
    [curves, bench] = await Promise.all([
      apiGet(`/api/strategies/equity-curve?days=${state.eqDays}`),
      apiGet(`/api/benchmarks/comparison?days=${state.eqDays}`),
    ]);
  } catch (err) { renderError(host, err, loadComparison); return; }

  cmpChart?.__efDestroy?.();
  const c = colors();
  cmpChart = createChart(host);
  const legend = document.getElementById('pf-cmp-legend');
  legend.replaceChildren();

  let plotted = 0;
  for (const [name, pts] of Object.entries(curves)) {
    if (!pts?.length) continue;
    const base = pts[0].total_equity;
    if (!base) continue;
    const slot = state.meta.get(name)?.color_slot ?? plotted;
    const color = c.series[slot % 8];
    const line = cmpChart.addLineSeries({ color, lineWidth: 2, priceLineVisible: false });
    line.setData(pts.map(p => ({ time: p.time, value: (p.total_equity / base - 1) * 100 })));
    legend.append(h('span', { class: 'c-chip', text: name }));
    plotted++;
  }
  const times = bench.times || [];
  const spy = (bench.indices || {}).SPY || [];
  if (times.length && spy.length === times.length) {
    const line = cmpChart.addLineSeries({
      color: c.benchmark, lineWidth: 1, lineStyle: 2, priceLineVisible: false,
    });
    line.setData(times.map((t, i) => ({ time: t, value: spy[i] })));
    legend.append(h('span', { class: 'c-chip', text: 'SPY' }));
  }
  if (!plotted) { renderEmpty(host, 'No strategy curves yet'); return; }
  cmpChart.timeScale().fitContent();
}

/* ── P&L calendar from closed trades ── */
async function loadCalendar() {
  const el = document.getElementById('pf-calendar');
  await panel(el, () => apiGet('/api/trades?status=CLOSED&limit=500'), (host, trades) => {
    const daily = {};
    for (const t of trades) {
      if (!t.exit_time || t.pnl_dollars == null) continue;
      const day = t.exit_time.slice(0, 10);
      daily[day] = (daily[day] || 0) + t.pnl_dollars;
    }
    if (!Object.keys(daily).length) { renderEmpty(host, 'No closed trades yet'); return; }
    clear(host).append(calendarHeatmap(daily));
  });
}

/* ── live proof scorecard ── */
async function loadProof() {
  const el = document.getElementById('pf-proof');
  await panel(el, () => apiGet('/api/strategies/scorecard?days=90'), (host, cards) => {
    if (!cards?.length) { renderEmpty(host, 'No scorecard data'); return; }
    const wrap = h('div', { class: 'flex-col gap-8' });
    for (const cdt of cards) {
      const crit = cdt.criteria || {};
      /* a strategy without enough live history hasn't FAILED anything —
         it just hasn't been graded yet */
      const pill = (ok) => cdt.status === 'insufficient_data'
        ? h('span', { class: 'c-pill neutral', text: 'PENDING' })
        : h('span', { class: `c-pill ${ok ? 'up' : 'down'}`, text: ok ? 'PASS' : 'FAIL' });
      wrap.append(h('div', { class: 'flex items-center gap-8 flex-wrap' },
        h('span', { class: 'num flex-1', text: cdt.strategy_name }),
        h('span', { class: 'num t-2', text: `Sharpe ${fmtNum(cdt.sharpe)}` }),
        h('span', { class: 'num ' + upDownClass(cdt.excess_vs_spy_pct),
                    text: cdt.excess_vs_spy_pct != null ? fmtPct(cdt.excess_vs_spy_pct) + ' vs SPY' : '—' }),
        pill(crit.all_met)));
    }
    wrap.append(h('div', { class: 't-dim', text:
      'Live = the validation bar on real-time data. Costs: live ~5bps/side; lab 2bps flat unless costed.' }));
    clear(host).append(wrap);
  });
}

/* ── open positions ── */
async function loadPositions() {
  const el = document.getElementById('pf-positions');
  await panel(el, () => apiGet('/api/trades?status=OPEN&limit=200'), (host, trades) => {
    const rows = trades;
    document.getElementById('pf-pos-count').textContent = `${rows.length} open`;
    if (!rows.length) { renderEmpty(host, 'No open positions'); return; }
    const tbody = h('tbody');
    for (const t of rows) {
      const upnl = t.unrealized_pnl;
      tbody.append(h('tr', {
        class: 'clickable',
        onclick: () => { window.location.href = `/symbol/${t.symbol}?markers=trades,dividends,splits`; },
      },
        h('td', { class: 'num', text: t.symbol }),
        h('td', { text: t.strategy_name }),
        h('td', { class: 'num', text: fmtInt(t.shares) }),
        h('td', { class: 'num', text: fmtPrice(t.entry_price) }),
        h('td', { class: 'num', text: t.current_price ? fmtPrice(t.current_price) : '—' }),
        h('td', { class: 'num ' + upDownClass(upnl), text: upnl != null ? fmtPnl(upnl) : '—' }),
        h('td', { class: 'num t-dim', text: fmtDate(t.entry_time) }),
      ));
    }
    clear(host).append(h('table', { class: 'c-table' },
      h('thead', {}, h('tr', {},
        h('th', { text: 'Symbol' }), h('th', { text: 'Strategy' }),
        h('th', { class: 'num', text: 'Shares' }), h('th', { class: 'num', text: 'Entry' }),
        h('th', { class: 'num', text: 'Now' }), h('th', { class: 'num', text: 'Unrlzd P&L' }),
        h('th', { text: 'Opened' }))),
      tbody));
  });
}

/* ── market strip + ops badge ── */
async function loadMarket() {
  const el = document.getElementById('pf-market');
  await panel(el, () => apiGet('/api/market/regime?limit=1'), (host, data) => {
    const m = data.latest || data;
    if (!m || m.spy_price == null) { renderEmpty(host, 'No market snapshot yet'); return; }
    const idx = (label, px, chg) => h('div', { class: 'c-stat' },
      h('div', { class: 'label', text: label }),
      h('div', { class: 'value num', text: fmtPrice(px) }),
      h('div', { class: 'delta num ' + upDownClass(chg), text: fmtPct(chg, { decimals: 2 }) }));
    clear(host).append(h('div', { class: 'grid-4' },
      idx('SPY', m.spy_price, m.spy_change_pct),
      idx('QQQ', m.qqq_price, m.qqq_change_pct),
      idx('IWM', m.iwm_price, m.iwm_change_pct),
      h('div', { class: 'c-stat' },
        h('div', { class: 'label', text: 'VIX · Regime' }),
        h('div', { class: 'value num', text: fmtNum(m.vix_level, 1) }),
        h('div', { class: 'delta t-2', text: m.market_regime || '' })),
    ));
  });
}

async function loadOpsBadge() {
  const el = document.getElementById('pf-ops-badge');
  try {
    const ops = await apiGet('/api/ops/health');
    const counts = ops.observation_counts || {};
    const critical = counts.critical || 0;
    const stale = (ops.heartbeats || []).some(hb => hb.ok === false);
    el.replaceChildren(h('a', {
      href: '/ops',
      class: `c-pill ${critical ? 'down' : stale ? 'warn' : 'up'}`,
      text: critical ? `${critical} CRITICAL` : stale ? 'DEGRADED' : 'SYSTEMS OK',
    }));
  } catch {
    el.replaceChildren(h('a', { href: '/ops', class: 'c-pill neutral', text: 'OPS ?' }));
  }
}

/* ── boot ── */
async function refreshAll() {
  await Promise.allSettled([
    loadHero(), loadEquity(), loadComparison(), loadCalendar(),
    loadProof(), loadPositions(), loadMarket(), loadOpsBadge(),
  ]);
}

async function init() {
  try {
    const meta = await apiGet('/api/strategies/meta');
    for (const m of meta) state.meta.set(m.name, m);
  } catch { /* meta optional */ }

  const ranges = [{ label: '30D', value: 30 }, { label: '90D', value: 90 },
                  { label: '180D', value: 180 }, { label: '1Y', value: 365 }];
  const rangeEl = document.getElementById('pf-eq-ranges');
  rangeEl.classList.add('c-chips', 'scroll');
  rangeEl.replaceChildren(...ranges.map(r => {
    const b = h('button', { class: 'c-chip' + (r.value === state.eqDays ? ' active' : ''), text: r.label });
    b.addEventListener('click', () => {
      for (const x of rangeEl.children) x.classList.remove('active');
      b.classList.add('active');
      state.eqDays = r.value;
      loadEquity(); loadComparison();
    });
    return b;
  }));

  onThemeChange(() => { loadEquity(); loadComparison(); });
  await refreshAll();
  poller(refreshAll, { intervalMs: 60000, maxFailures: 10 });
}

document.addEventListener('DOMContentLoaded', init);

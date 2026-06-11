/* Portfolio — PER-STRATEGY account tracking (v5.52 rebuild).
   The centerpiece is a multi-line equity chart: one series per isolated
   $100k account plotting total_return_pct (the comparable axis), with a
   toggleable legend (All / None / alt-click = solo, persisted in
   localStorage), a crosshair legend showing name + return% + equity $,
   and an account-card grid sorted by total return. The old aggregate
   hero panels and P&L calendar are gone — the only fleet rollup left is
   the slim strip at the top (server-computed /summary, never fabricated).

   Color/style policy for >8 series on the 8-slot palette: color comes
   from meta.color_slot (fallback: roster index mod 8); each full trip
   through the palette switches line style — solid, then dashed, then
   dotted — and the legend dot goes hollow ("ring") so same-color pairs
   stay distinguishable up to 24 accounts. */

import { apiGet } from '../core/net.js';
import { fmtDollar, fmtPnl, fmtPct, fmtNum, fmtPrice, fmtInt, upDownClass } from '../core/fmt.js';
import { h, clear, renderError, renderEmpty, panel } from '../core/dom.js';
import { createChart, colors, rangeSwitcher } from '../core/charts.js';
import { onThemeChange } from '../core/theme.js';
import { sparkline } from '../components/sparkline.js';
import { poller } from '../core/poll.js';

const LWC = window.LightweightCharts;
const HIDDEN_KEY = 'ef-pf-hidden'; // hidden set (not visible set) — new strategies default to visible
const RANGES = [
  { label: '1D', value: 1 }, { label: '1W', value: 7 },
  { label: '1M', value: 30 }, { label: '3M', value: 90 },
  { label: 'ALL', value: 365 }, // the API caps the window at 365d
];

const state = {
  days: 90,
  meta: new Map(),   // name -> {display_name, color_slot, tier, ...}
  names: [],         // sorted roster (accounts ∪ curves) — drives styles
  curves: {},        // name -> [{time, total_equity, total_return_pct}]
  accounts: [],
  positions: {},     // name -> [open trades]
  hidden: loadHidden(),
};

let chart = null;
const seriesMap = new Map(); // name -> line series

function loadHidden() {
  try { return new Set(JSON.parse(localStorage.getItem(HIDDEN_KEY) || '[]')); }
  catch { return new Set(); }
}
function saveHidden() {
  try { localStorage.setItem(HIDDEN_KEY, JSON.stringify([...state.hidden])); }
  catch { /* private mode etc. — visibility just won't persist */ }
}

const isVisible = (name) => !state.hidden.has(name);
const displayName = (name) => state.meta.get(name)?.display_name || name;

function styleFor(name) {
  const idx = Math.max(0, state.names.indexOf(name));
  const slot = ((state.meta.get(name)?.color_slot ?? idx) % 8 + 8) % 8;
  const cycle = Math.floor(idx / 8);
  const lineStyle = cycle === 0 ? LWC.LineStyle.Solid
    : cycle === 1 ? LWC.LineStyle.Dashed : LWC.LineStyle.Dotted;
  return { slot, cycle, lineStyle };
}

function dot(st) {
  return h('span', { class: `c-dot s${st.slot}${st.cycle ? ' ring' : ''}` });
}

/* ── main chart ── */
function destroyChart() {
  chart?.__efDestroy?.();
  chart = null;
  seriesMap.clear();
}

function pctPoints(name) {
  return (state.curves[name] || []).map(p => ({ time: p.time, value: p.total_return_pct ?? 0 }));
}

function buildChart() {
  const host = document.getElementById('pf-chart');
  destroyChart();
  clear(host);
  const withData = state.names.filter(n => (state.curves[n] || []).length);
  if (!withData.length) {
    renderEmpty(host, 'No equity snapshots yet');
    renderXLegend(null);
    return;
  }
  const c = colors();
  chart = createChart(host);
  // % axis; per-series last-value/price-line labels stay off — with 14
  // series they shingle into noise; the crosshair legend is the readout
  chart.applyOptions({ localization: { priceFormatter: (p) => p.toFixed(1) + '%' } });
  for (const name of withData) {
    const st = styleFor(name);
    const series = chart.addLineSeries({
      color: c.series[st.slot], lineWidth: 2, lineStyle: st.lineStyle,
      priceLineVisible: false, lastValueVisible: false,
      visible: isVisible(name),
    });
    series.setData(pctPoints(name));
    seriesMap.set(name, series);
  }
  chart.timeScale().fitContent();
  chart.subscribeCrosshairMove((param) => renderXLegend(param?.time ?? null));
  renderXLegend(null);
}

function refreshSeriesData() {
  // same roster -> update in place (preserves the user's pan/zoom)
  for (const [name, series] of seriesMap) series.setData(pctPoints(name));
  renderXLegend(null);
}

/* ── crosshair legend (name + return% + equity $) ── */
function pointAt(pts, time) {
  if (!pts?.length) return null;
  if (time == null) return pts[pts.length - 1];
  let lo = 0, hi = pts.length - 1, best = null;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (pts[mid].time <= time) { best = pts[mid]; lo = mid + 1; } else { hi = mid - 1; }
  }
  return best;
}

function renderXLegend(time) {
  const el = document.getElementById('pf-xlegend');
  const entries = [];
  for (const name of state.names) {
    if (!isVisible(name)) continue;
    const p = pointAt(state.curves[name], time);
    if (p) entries.push({ name, p });
  }
  entries.sort((a, b) => (b.p.total_return_pct ?? 0) - (a.p.total_return_pct ?? 0));
  el.replaceChildren(...entries.map(({ name, p }) => h('span', { class: 'e' },
    dot(styleFor(name)),
    h('b', { text: displayName(name) }),
    h('span', { class: 'num ' + upDownClass(p.total_return_pct), text: fmtPct(p.total_return_pct) }),
    h('span', { class: 'num t-dim', text: fmtDollar(p.total_equity) }),
  )));
}

/* ── visibility: toggle / solo / all / none ── */
function toggle(name) {
  if (state.hidden.has(name)) state.hidden.delete(name);
  else state.hidden.add(name);
  applyVisibility();
}

function solo(name) {
  // re-soloing the soloed strategy restores everything
  const alreadySolo = isVisible(name) && state.hidden.size === state.names.length - 1;
  state.hidden = alreadySolo ? new Set() : new Set(state.names.filter(n => n !== name));
  applyVisibility();
}

function setAll(visibleAll) {
  state.hidden = visibleAll ? new Set() : new Set(state.names);
  applyVisibility();
}

function applyVisibility() {
  saveHidden();
  // invisible series drop out of price-scale autoscale -> chart rescales
  for (const [name, series] of seriesMap) series.applyOptions({ visible: isVisible(name) });
  renderLegend();
  renderXLegend(null);
  for (const card of document.querySelectorAll('#pf-grid .pf-acct')) {
    card.classList.toggle('dim', state.hidden.has(card.dataset.name));
  }
}

/* ── toggle legend chips ── */
function renderLegend() {
  const el = document.getElementById('pf-legend');
  el.replaceChildren(...state.names.map(name => {
    const st = styleFor(name);
    const hasData = (state.curves[name] || []).length > 0;
    const chip = h('button', {
      class: 'c-chip' + (isVisible(name) ? '' : ' off'),
      title: (hasData ? '' : 'no snapshots yet — ') + 'click: toggle · alt-click: solo',
    }, dot(st), h('span', { text: displayName(name) }));
    chip.addEventListener('click', (ev) => (ev.altKey ? solo(name) : toggle(name)));
    return chip;
  }));
}

/* ── account grid (one card per isolated paper account) ── */
function renderGrid() {
  const host = document.getElementById('pf-grid');
  if (!state.accounts.length) { renderEmpty(host, 'No strategy accounts yet'); return; }
  const ret = (a) => a.starting_capital ? (a.total_equity / a.starting_capital - 1) * 100 : -Infinity;
  const rows = [...state.accounts].sort((a, b) => ret(b) - ret(a));
  const c = colors();
  clear(host);
  for (const a of rows) {
    const name = a.strategy_name;
    const m = state.meta.get(name) || {};
    const st = styleFor(name);
    const pnl = (a.total_equity || 0) - (a.starting_capital || 0);
    const pct = a.starting_capital ? (pnl / a.starting_capital) * 100 : null;
    const curve = state.curves[name] || [];
    const openCount = (state.positions[name] || []).length;
    const details = h('a', { class: 'c-chip', href: '/strategies', text: 'Details →' });
    details.addEventListener('click', (ev) => ev.stopPropagation());
    host.append(h('div', {
      class: 'c-card pf-acct' + (state.hidden.has(name) ? ' dim' : ''),
      dataset: { name },
      onclick: () => solo(name),
    },
      h('div', { class: 'c-card-body' },
        h('div', { class: 'flex items-center gap-8 mb-8' },
          dot(st),
          h('span', { class: 'nm num flex-1', text: displayName(name) }),
          m.tier ? h('span', { class: `c-pill ${m.tier === 'validated' ? 'info' : 'neutral'}`, text: m.tier }) : null,
          a.pending ? h('span', { class: 'c-pill warn', text: 'PENDING' }) : null,
          a.is_paused ? h('span', { class: 'c-pill warn', text: 'PAUSED' }) : null),
        h('div', { class: 'flex items-end justify-between gap-8' },
          h('div', {},
            h('div', { class: 'eq num', text: fmtDollar(a.total_equity) }),
            h('div', { class: 'num ' + upDownClass(pnl),
                       text: fmtPnl(pnl) + (pct != null ? ` (${fmtPct(pct)})` : '') })),
          curve.length >= 2
            ? sparkline(curve.map(p => p.total_equity),
                        { size: 'lg', color: c.series[st.slot], baseline: a.starting_capital })
            : h('span', { class: 't-dim meta', text: 'no curve yet' })),
        h('div', { class: 'flex items-center gap-12 mt-8 meta' },
          h('span', { class: 't-dim num', text: `DD ${fmtPct(a.drawdown_pct, { signed: false })}` }),
          h('span', { class: 't-dim num', text: `${fmtInt(openCount)} open` }),
          h('span', { class: 'flex-1' }),
          details))));
  }
}

/* ── the one data load that feeds chart + legend + grid ── */
async function loadAccounts({ rebuild = false } = {}) {
  const host = document.getElementById('pf-chart');
  let accounts, positions, curves;
  try {
    [accounts, positions, curves] = await Promise.all([
      apiGet('/api/strategies/accounts'),
      apiGet('/api/strategies/positions'),
      apiGet(`/api/strategies/equity-curve?days=${state.days}`),
    ]);
  } catch (err) {
    destroyChart();
    renderError(host, err, () => loadAccounts({ rebuild: true }));
    return;
  }

  state.accounts = accounts;
  state.positions = positions;
  state.curves = curves;
  const names = new Set([...accounts.map(a => a.strategy_name), ...Object.keys(curves)]);
  const roster = [...names].sort();
  const rosterChanged = roster.join('|') !== state.names.join('|');
  state.names = roster;
  for (const hid of [...state.hidden]) if (!names.has(hid)) state.hidden.delete(hid);

  const needsBuild = rebuild || rosterChanged || !chart ||
    roster.some(n => (curves[n] || []).length && !seriesMap.has(n));
  if (needsBuild) buildChart();
  else refreshSeriesData();
  renderLegend();
  renderGrid();
}

/* ── slim fleet strip (the only aggregate left on this page) ── */
async function loadFleet() {
  const el = document.getElementById('pf-fleet');
  await panel(el, () => apiGet('/api/strategies/summary'), (host, lanes) => {
    const L = lanes.all || {};
    clear(host).append(
      h('span', {}, 'Fleet ', h('b', { class: 'num', text: fmtDollar(L.total_equity) })),
      h('span', { class: 'num ' + upDownClass(L.day_pnl),
                  text: (L.day_pnl != null ? fmtPnl(L.day_pnl) : '—') + ' today' }),
      h('span', { class: 'num', text: `${fmtInt(L.strategies)} accounts` }),
    );
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
    loadAccounts(), loadFleet(), loadProof(), loadMarket(), loadOpsBadge(),
  ]);
}

async function init() {
  try {
    const meta = await apiGet('/api/strategies/meta');
    for (const m of meta) state.meta.set(m.name, m);
  } catch { /* meta optional — roster-index colors and raw names */ }

  rangeSwitcher(document.getElementById('pf-ranges'), RANGES, state.days, (v) => {
    state.days = v;
    loadAccounts({ rebuild: true }); // refetch window + fitContent
  });

  document.getElementById('pf-legend-actions').append(
    h('button', { class: 'c-chip', text: 'All', onclick: () => setAll(true) }),
    h('button', { class: 'c-chip', text: 'None', onclick: () => setAll(false) }),
  );

  // chart series + sparkline strokes capture token colors at draw time
  onThemeChange(() => { if (state.names.length) { buildChart(); renderGrid(); } });

  await refreshAll();
  poller(refreshAll, { intervalMs: 60000, maxFailures: 10 });
}

document.addEventListener('DOMContentLoaded', init);

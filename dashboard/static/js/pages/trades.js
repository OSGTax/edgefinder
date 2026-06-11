/* Trades journal — filters, stats, sortable table, expandable reasoning
   rows (indicators at entry/exit), symbol deep-links. Sort state resets
   on filter change (the old page's stale-sort bug). */

import { apiGet } from '../core/net.js';
import { fmtDateTimeET, fmtPnl, fmtPct, fmtPrice, fmtNum, fmtInt, fmtDate, fmtDuration, upDownClass } from '../core/fmt.js';
import { h, clear, renderEmpty, panel } from '../core/dom.js';

const state = { strategy: '', status: '', symbol: '', sortCol: 'entry_time', sortDir: -1, expanded: null, trades: [] };

const COLS = [
  ['symbol', 'Symbol', t => t.symbol, false],
  ['strategy_name', 'Strategy', t => t.strategy_name, false],
  ['shares', 'Shares', t => fmtInt(t.shares), true],
  ['entry_price', 'Entry', t => fmtPrice(t.entry_price), true],
  ['exit_price', 'Exit', t => t.exit_price ? fmtPrice(t.exit_price) : '—', true],
  ['pnl_dollars', 'P&L', t => t.pnl_dollars != null ? fmtPnl(t.pnl_dollars) : (t.unrealized_pnl != null ? fmtPnl(t.unrealized_pnl) + '*' : '—'), true],
  ['pnl_percent', 'P&L %', t => t.pnl_percent != null ? fmtPct(t.pnl_percent) : '—', true],
  ['r_multiple', 'R', t => fmtNum(t.r_multiple, 1), true],
  ['hold_duration_hours', 'Held', t => fmtDuration(t.hold_duration_hours), true],
  ['status', 'Status', t => t.status, false],
  ['entry_time', 'Opened', t => fmtDateTimeET(t.entry_time), true],
];

function filtered() {
  let rows = state.trades;
  if (state.status === 'OPEN') rows = rows.filter(t => t.status === 'OPEN');
  else if (state.status === 'wins') rows = rows.filter(t => t.status === 'CLOSED' && (t.pnl_dollars || 0) > 0);
  else if (state.status === 'losses') rows = rows.filter(t => t.status === 'CLOSED' && (t.pnl_dollars || 0) <= 0);
  if (state.symbol) rows = rows.filter(t => t.symbol.includes(state.symbol.toUpperCase()));
  const dir = state.sortDir;
  const col = state.sortCol;
  return [...rows].sort((a, b) => {
    const av = a[col], bv = b[col];
    if (av == null) return 1;
    if (bv == null) return -1;
    return (av > bv ? 1 : av < bv ? -1 : 0) * dir;
  });
}

function indicatorChips(ind) {
  if (!ind || typeof ind !== 'object') return h('span', { class: 't-dim', text: 'no indicator data' });
  const wrap = h('div', { class: 'c-chips' });
  for (const k of ['rsi', 'macd_line', 'ema_21', 'ema_50', 'close', 'volume_ratio']) {
    if (ind[k] == null) continue;
    wrap.append(h('span', { class: 'c-chip', text: `${k}: ${fmtNum(ind[k], 2)}` }));
  }
  return wrap;
}

function expandRow(t) {
  const td = h('td', { colspan: String(COLS.length) });
  td.append(
    h('div', { class: 'grid-2 gap-12' },
      h('div', {},
        h('h4', { text: `Entry — ${fmtDateTimeET(t.entry_time)} @ ${fmtPrice(t.entry_price)}` }),
        h('p', { class: 't-2', text: t.entry_reasoning || 'no reasoning recorded' }),
        indicatorChips(t.indicators_at_entry)),
      h('div', {},
        h('h4', { text: t.exit_time ? `Exit — ${fmtDateTimeET(t.exit_time)} @ ${fmtPrice(t.exit_price)} (${t.exit_reason || '—'})` : 'Still open' }),
        h('p', { class: 't-2', text: t.exit_reasoning || (t.exit_time ? 'no reasoning recorded' : '') }),
        t.exit_time ? indicatorChips(t.indicators_at_exit) : null),
    ),
    h('a', { class: 'c-btn ghost mt-8', href: `/symbol/${t.symbol}?markers=trades`, text: 'Open chart →' }),
  );
  return h('tr', { class: 'row-expand' }, td);
}

function renderTable() {
  const host = document.getElementById('tr-table');
  const rows = filtered();
  if (!rows.length) { renderEmpty(host, 'No trades match'); return; }

  const thead = h('tr', {}, ...COLS.map(([key, label, , numeric]) => {
    const th = h('th', { class: (numeric ? 'num ' : '') + 'sortable', text: label });
    if (state.sortCol === key) {
      th.append(h('span', { class: 'sort-arrow', text: state.sortDir > 0 ? '▲' : '▼' }));
    }
    th.addEventListener('click', () => {
      if (state.sortCol === key) state.sortDir *= -1;
      else { state.sortCol = key; state.sortDir = -1; }
      renderTable();
    });
    return th;
  }));

  const tbody = h('tbody');
  for (const t of rows) {
    const tr = h('tr', { class: 'clickable' }, ...COLS.map(([key, , get, numeric]) => {
      const cls = (numeric ? 'num ' : '') +
        (key.startsWith('pnl') || key === 'r_multiple'
          ? upDownClass(t[key] ?? t.unrealized_pnl) : '');
      return h('td', { class: cls.trim(), text: String(get(t)) });
    }));
    tr.addEventListener('click', () => {
      state.expanded = state.expanded === t.trade_id ? null : t.trade_id;
      renderTable();
    });
    tbody.append(tr);
    if (state.expanded === t.trade_id) tbody.append(expandRow(t));
  }
  clear(host).append(h('table', { class: 'c-table' }, h('thead', {}, thead), tbody));
}

async function loadStats() {
  const el = document.getElementById('tr-stats');
  const q = state.strategy ? `?strategy=${state.strategy}` : '';
  await panel(el, () => apiGet(`/api/trades/stats${q}`), (host, s) => {
    const stat = (label, value, cls = '') => h('div', { class: 'c-stat' },
      h('div', { class: 'label', text: label }),
      h('div', { class: `value num ${cls}`, text: value }));
    clear(host).append(
      stat('Trades', `${fmtInt(s.total_trades)} (${fmtInt(s.open_trades)} open)`),
      stat('Win rate', s.win_rate != null ? fmtPct(s.win_rate, { signed: false }) : '—'),
      stat('Total P&L', fmtPnl(s.total_pnl), upDownClass(s.total_pnl)),
      stat('Profit factor', fmtNum(s.profit_factor)),
    );
  });
}

async function loadTrades() {
  const host = document.getElementById('tr-table');
  const q = new URLSearchParams({ limit: 500 });
  if (state.strategy) q.set('strategy', state.strategy);
  await panel(host, () => apiGet(`/api/trades?${q}`), (el, trades) => {
    state.trades = trades;
    state.sortCol = 'entry_time'; state.sortDir = -1; state.expanded = null;
    renderTable();
  });
}

async function init() {
  try {
    const meta = await apiGet('/api/strategies/meta');
    const sel = document.getElementById('tr-strategy');
    for (const m of meta) sel.append(h('option', { value: m.name, text: `${m.name} (${m.lane})` }));
  } catch { /* dropdown stays generic */ }

  document.getElementById('tr-strategy').addEventListener('change', (e) => {
    state.strategy = e.target.value;
    loadStats(); loadTrades();
  });
  document.getElementById('tr-status').addEventListener('click', (e) => {
    const b = e.target.closest('button');
    if (!b) return;
    for (const x of document.querySelectorAll('#tr-status button')) x.classList.remove('active');
    b.classList.add('active');
    state.status = b.dataset.status;
    renderTable();
  });
  let t = null;
  document.getElementById('tr-symbol').addEventListener('input', (e) => {
    clearTimeout(t);
    t = setTimeout(() => { state.symbol = e.target.value.trim(); renderTable(); }, 200);
  });

  try {
    const integ = await apiGet('/api/trades/integrity');
    const ok = integ.ok ?? integ.valid ?? (integ.verified === integ.total);
    document.getElementById('tr-integrity').replaceChildren(
      h('span', { class: `c-pill ${ok ? 'up' : 'warn'}`,
                  text: ok ? 'CHAIN VERIFIED' : 'CHAIN: see /ops' }));
  } catch { /* badge optional */ }

  loadStats();
  loadTrades();
}

document.addEventListener('DOMContentLoaded', init);

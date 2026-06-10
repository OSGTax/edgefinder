/* Screener — hand-rolled sector treemap (Chart.js retired), the scanner's
   ranked qualification watchlist, and the active-universe table with
   sector/strategy/search filtering and symbol deep-links. */

import { apiGet } from '../core/net.js';
import { fmtPrice, fmtPct, fmtNum, fmtCompact, upDownClass } from '../core/fmt.js';
import { h, clear, renderEmpty, panel } from '../core/dom.js';
import { treemap } from '../components/treemap.js';

const state = { stocks: [], sector: '', strategy: '', search: '', sortCol: 'market_cap', sortDir: -1 };

async function loadTreemap() {
  const el = document.getElementById('sc-treemap');
  await panel(el, async () => {
    const [sectors, stocks] = await Promise.all([
      apiGet('/api/benchmarks/sectors').catch(() => ({ sectors: [] })),
      state.stocks.length ? Promise.resolve(state.stocks) : apiGet('/api/research/active'),
    ]);
    state.stocks = stocks;
    return sectors;
  }, (host, sectorData) => {
    const counts = {};
    for (const st of state.stocks) {
      const sec = st.sector || 'Unknown';
      counts[sec] = (counts[sec] || 0) + 1;
    }
    const quadrants = {};
    for (const s of (sectorData.sectors || sectorData || [])) {
      if (s && s.name) quadrants[s.name] = s.quadrant;
    }
    const nodes = Object.entries(counts).map(([name, count]) => ({
      name, count, value: count, quadrant: quadrants[name] || 'unknown',
    }));
    if (!nodes.length) { renderEmpty(host, 'No universe data yet — run a scan'); return; }
    treemap(host, nodes, {
      height: 200,
      onClick: (name) => {
        state.sector = state.sector === name ? '' : name;
        renderClear();
        renderTable();
      },
    });
    renderClear();
  });
}

function renderClear() {
  const el = document.getElementById('sc-sector-clear');
  el.replaceChildren(state.sector
    ? h('button', { class: 'c-chip active', text: `${state.sector} ✕`,
                    onclick: () => { state.sector = ''; renderClear(); renderTable(); } })
    : h('span', { class: 't-dim', text: 'click a sector to filter' }));
}

async function loadWatchlist() {
  const el = document.getElementById('sc-watchlist');
  const q = state.strategy ? `&strategy=${state.strategy}` : '';
  await panel(el, () => apiGet(`/api/research/qualifications?qualified=true&limit=50${q}`), (host, rows) => {
    if (!rows.length) { renderEmpty(host, 'No qualified tickers — scanner runs nightly'); return; }
    const tbody = h('tbody');
    for (const r of rows) {
      tbody.append(h('tr', {
        class: 'clickable',
        onclick: () => { window.location.href = `/symbol/${r.symbol}`; },
      },
        h('td', { class: 'num', text: r.symbol }),
        h('td', { text: r.strategy_name }),
        h('td', { class: 'num', text: r.score != null ? fmtNum(r.score, 0) : '—' })));
    }
    clear(host).append(h('table', { class: 'c-table' },
      h('thead', {}, h('tr', {},
        h('th', { text: 'Symbol' }), h('th', { text: 'Strategy' }),
        h('th', { class: 'num', text: 'Score' }))),
      tbody));
  });
}

const COLS = [
  ['symbol', 'Symbol', s => s.symbol, false],
  ['company_name', 'Company', s => s.company_name || '—', false],
  ['sector', 'Sector', s => s.sector || '—', false],
  ['last_price', 'Price', s => fmtPrice(s.last_price), true],
  ['market_cap', 'Mkt cap', s => s.market_cap ? '$' + fmtCompact(s.market_cap) : '—', true],
  ['earnings_growth', 'Earn gr', s => s.earnings_growth != null ? fmtPct(s.earnings_growth * 100) : '—', true],
  ['price_to_earnings', 'P/E', s => fmtNum(s.price_to_earnings, 1), true],
  ['rsi_14', 'RSI', s => fmtNum(s.rsi_14, 0), true],
  ['short_interest', 'Short %', s => s.short_interest != null ? fmtPct(s.short_interest * 100, { signed: false }) : '—', true],
];

function filtered() {
  let rows = state.stocks;
  if (state.sector) rows = rows.filter(s => (s.sector || 'Unknown') === state.sector);
  if (state.strategy) rows = rows.filter(s => (s.qualifying_strategies || []).includes(state.strategy));
  if (state.search) {
    const q = state.search.toLowerCase();
    rows = rows.filter(s => s.symbol.toLowerCase().includes(q)
      || (s.company_name || '').toLowerCase().includes(q));
  }
  const { sortCol, sortDir } = state;
  return [...rows].sort((a, b) => {
    const av = a[sortCol], bv = b[sortCol];
    if (av == null) return 1;
    if (bv == null) return -1;
    return (av > bv ? 1 : av < bv ? -1 : 0) * sortDir;
  });
}

function renderTable() {
  const host = document.getElementById('sc-table');
  const rows = filtered();
  if (!rows.length) { renderEmpty(host, 'No stocks match'); return; }
  const thead = h('tr', {}, ...COLS.map(([key, label, , numeric]) => {
    const th = h('th', { class: (numeric ? 'num ' : '') + 'sortable', text: label });
    if (state.sortCol === key) th.append(h('span', { class: 'sort-arrow', text: state.sortDir > 0 ? '▲' : '▼' }));
    th.addEventListener('click', () => {
      if (state.sortCol === key) state.sortDir *= -1;
      else { state.sortCol = key; state.sortDir = -1; }
      renderTable();
    });
    return th;
  }));
  const tbody = h('tbody');
  for (const s of rows.slice(0, 300)) {
    tbody.append(h('tr', {
      class: 'clickable',
      onclick: () => { window.location.href = `/symbol/${s.symbol}`; },
    }, ...COLS.map(([key, , get, numeric]) => h('td', {
      class: (numeric ? 'num ' : '') +
        (key === 'earnings_growth' ? upDownClass(s[key]) : ''),
      text: String(get(s)),
    }))));
  }
  clear(host).append(h('table', { class: 'c-table' }, h('thead', {}, thead), tbody));
}

async function loadStocks() {
  const host = document.getElementById('sc-table');
  await panel(host, () => apiGet('/api/research/active'), (el, stocks) => {
    state.stocks = stocks;
    renderTable();
  });
}

async function init() {
  try {
    const meta = await apiGet('/api/strategies/meta');
    const sel = document.getElementById('sc-strategy');
    for (const m of meta.filter(m => m.lane === 'arena')) {
      sel.append(h('option', { value: m.name, text: m.name }));
    }
  } catch { /* dropdown stays generic */ }

  document.getElementById('sc-strategy').addEventListener('change', (e) => {
    state.strategy = e.target.value;
    loadWatchlist();
    renderTable();
  });
  let t = null;
  document.getElementById('sc-search').addEventListener('input', (e) => {
    clearTimeout(t);
    t = setTimeout(() => { state.search = e.target.value.trim(); renderTable(); }, 200);
  });

  await loadStocks();
  loadTreemap();
  loadWatchlist();
}

document.addEventListener('DOMContentLoaded', init);

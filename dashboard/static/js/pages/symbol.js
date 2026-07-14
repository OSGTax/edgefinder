/* Symbol Workstation — the trading-terminal centerpiece.
   Candlestick + volume + toggleable EMA/BB overlays, synced RSI/MACD
   panes, trade/dividend/split/news markers, symbol search, full R2
   history via range=max, URL-shareable state, trade-detail drawer. */

import { apiGet } from '../core/net.js';
import { toEpochSec, fmtPrice, fmtPct, fmtPnl, fmtCompact, fmtDate, fmtNum, upDownClass } from '../core/fmt.js';
import { h, clear, skeleton, renderError, renderEmpty, debounce } from '../core/dom.js';
import { createChart, colors, syncPanes, rangeSwitcher, fullscreenButton, eventMarkers, mergeMarkers } from '../core/charts.js';
import { onThemeChange } from '../core/theme.js';
import { sparkline } from '../components/sparkline.js';

/* ── state ── */
const RANGES = [
  { label: '1M', value: '1m' }, { label: '3M', value: '3m' },
  { label: '6M', value: '6m' }, { label: '1Y', value: '1y' },
  { label: '2Y', value: '2y' }, { label: '5Y', value: '5y' },
  { label: 'MAX', value: 'max' },
];
const IND_DEFS = [
  { key: 'ema9', label: 'EMA 9', series: ['ema_9'] },
  { key: 'ema21', label: 'EMA 21', series: ['ema_21'] },
  { key: 'ema50', label: 'EMA 50', series: ['ema_50'] },
  { key: 'ema200', label: 'EMA 200', series: ['ema_200'] },
  { key: 'bb', label: 'Bollinger', series: ['bb_upper', 'bb_middle', 'bb_lower'] },
  { key: 'rsi', label: 'RSI', pane: 'rsi' },
  { key: 'macd', label: 'MACD', pane: 'macd' },
];
const MARKER_DEFS = [
  { key: 'trades', label: 'Trades' },
  { key: 'dividends', label: 'Dividends' },
  { key: 'splits', label: 'Splits' },
  { key: 'news', label: 'News' },
];

const state = {
  symbol: null,
  range: '1y',
  inds: new Set(['ema21', 'ema200', 'rsi']),
  markers: new Set(['trades', 'dividends', 'splits']),
  bars: null,        // /api/symbols response
  events: null,
  trades: [],
  profile: null,
};

let charts = null;   // { price, candle, volume, overlays:{}, rsi, macd }

/* ── URL state ── */
function parseUrl() {
  const m = window.location.pathname.match(/^\/symbol\/([A-Za-z.\-]+)/);
  const q = new URLSearchParams(window.location.search);
  state.symbol = m ? m[1].toUpperCase() : (q.get('ticker') || 'SPY').toUpperCase();
  if (q.get('range') && RANGES.some(r => r.value === q.get('range'))) state.range = q.get('range');
  if (q.get('ind') != null) state.inds = new Set(q.get('ind').split(',').filter(Boolean));
  if (q.get('markers') != null) state.markers = new Set(q.get('markers').split(',').filter(Boolean));
}

function writeUrl() {
  const q = new URLSearchParams();
  if (state.range !== '1y') q.set('range', state.range);
  q.set('ind', [...state.inds].join(','));
  q.set('markers', [...state.markers].join(','));
  history.replaceState(null, '', `/symbol/${state.symbol}?${q}`);
}

/* ── data loading ── */
async function loadSymbol() {
  writeUrl();
  document.title = `${state.symbol} — EdgeFinder`;
  document.getElementById('sym-name').textContent = state.symbol;

  const shell = document.getElementById('pane-price');
  shell.classList.add('loading');
  const status = document.getElementById('sym-chart-status');
  status.classList.add('hidden');

  const [bars, events, trades, profile] = await Promise.allSettled([
    apiGet(`/api/symbols/${state.symbol}/bars?range=${state.range}&indicators=true`, { timeout: 30000 }),
    apiGet(`/api/symbols/${state.symbol}/events`),
    apiGet(`/api/desk/trades?limit=500`),
    apiGet(`/api/symbols/${state.symbol}/fundamentals`),
  ]);
  shell.classList.remove('loading');

  if (bars.status === 'rejected') {
    destroyCharts();
    status.classList.remove('hidden');
    renderError(status, bars.reason, () => loadSymbol());
    return;
  }
  state.bars = bars.value;
  state.events = events.status === 'fulfilled' ? events.value : { dividends: [], splits: [], news: [] };
  // The AI's own fills for this symbol (the desk ledger is the only trades
  // source since the workbench was retired). Option fills on the same
  // underlying count too — their OCC symbol is underlying + 6-digit date +
  // C/P + 8-digit strike.
  const isMine = t => t.symbol === state.symbol ||
    (t.symbol && t.symbol.startsWith(state.symbol) &&
     /^\d{6}[CP]\d{8}$/.test(t.symbol.slice(state.symbol.length)));
  state.trades = trades.status === 'fulfilled'
    ? (trades.value || []).filter(isMine)
    : [];
  state.profile = profile.status === 'fulfilled' ? profile.value : null;

  renderHeader();
  buildCharts();
  renderRail();
}

/* ── header ── */
function renderHeader() {
  const bars = state.bars.bars;
  const last = bars[bars.length - 1];
  const prev = bars[bars.length - 2];
  const priceEl = document.getElementById('sym-price');
  const chgEl = document.getElementById('sym-change');
  priceEl.textContent = fmtPrice(last?.close);
  if (last && prev && prev.close) {
    const pct = (last.close / prev.close - 1) * 100;
    chgEl.textContent = `${fmtPnl(last.close - prev.close).replace('$', '')} (${fmtPct(pct, { decimals: 2 })})`;
    chgEl.className = 'num ' + upDownClass(pct);
  } else {
    chgEl.textContent = '';
  }
  const src = document.getElementById('sym-source');
  src.classList.remove('hidden');
  const firstYear = bars.length ? new Date(bars[0].time * 1000).getUTCFullYear() : '';
  src.textContent = state.bars.source === 'r2'
    ? `R2 · full history (since ${firstYear})`
    : state.bars.truncated ? 'DB · truncated (R2 unavailable)' : `DB · ${state.range.toUpperCase()}`;
  src.className = 'c-pill ' + (state.bars.truncated ? 'warn' : 'neutral');
}

/* ── chart construction ── */
function destroyCharts() {
  if (!charts) return;
  for (const c of [charts.price, charts.rsi, charts.macd]) c?.__efDestroy?.();
  charts = null;
}

function buildCharts() {
  destroyCharts();
  const c = colors();
  const priceEl = document.getElementById('pane-price');
  const rsiEl = document.getElementById('pane-rsi');
  const macdEl = document.getElementById('pane-macd');

  const price = createChart(priceEl);
  const candle = price.addCandlestickSeries({
    upColor: c.up, downColor: c.down, wickUpColor: c.up, wickDownColor: c.down,
    borderVisible: false,
  });
  candle.setData(state.bars.bars);

  const volume = price.addHistogramSeries({
    priceScaleId: 'vol', priceFormat: { type: 'volume' }, lastValueVisible: false,
    priceLineVisible: false,
  });
  price.priceScale('vol').applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } });
  volume.setData(state.bars.bars.map(b => ({
    time: b.time, value: b.volume || 0,
    color: b.close >= b.open ? c.up + '55' : c.down + '55',
  })));

  charts = { price, candle, volume, overlays: {}, rsi: null, macd: null, rsiSeries: null, macdSeries: null };

  // EMA / BB overlays
  const ind = state.bars.indicators || {};
  const overlayColor = { ema_9: c.series[3], ema_21: c.series[1], ema_50: c.series[2], ema_200: c.series[4],
                         bb_upper: c.benchmark, bb_middle: c.series[1], bb_lower: c.benchmark };
  for (const def of IND_DEFS) {
    if (def.pane || !state.inds.has(def.key)) continue;
    for (const sname of def.series) {
      if (!ind[sname]) continue;
      const line = price.addLineSeries({
        color: overlayColor[sname] || c.info, lineWidth: 1,
        priceLineVisible: false, lastValueVisible: false,
        lineStyle: sname.startsWith('bb_') && sname !== 'bb_middle' ? 2 : 0,
      });
      line.setData(ind[sname]);
      charts.overlays[sname] = line;
    }
  }

  // sub-panes
  const panes = [{ chart: price, series: candle }];
  rsiEl.classList.toggle('hidden', !state.inds.has('rsi'));
  if (state.inds.has('rsi') && ind.rsi) {
    const rsi = createChart(rsiEl);
    const line = rsi.addLineSeries({ color: c.series[3], lineWidth: 1.5, priceLineVisible: false });
    line.setData(ind.rsi);
    line.createPriceLine({ price: 70, color: c.down, lineWidth: 1, lineStyle: 3, axisLabelVisible: false });
    line.createPriceLine({ price: 30, color: c.up, lineWidth: 1, lineStyle: 3, axisLabelVisible: false });
    charts.rsi = rsi; charts.rsiSeries = line;
    panes.push({ chart: rsi, series: line });
  }
  macdEl.classList.toggle('hidden', !state.inds.has('macd'));
  if (state.inds.has('macd') && ind.macd_line) {
    const macd = createChart(macdEl);
    if (ind.macd_histogram) {
      const hist = macd.addHistogramSeries({ priceLineVisible: false, lastValueVisible: false });
      hist.setData(ind.macd_histogram.map(p => ({
        ...p, color: p.value >= 0 ? c.up + '88' : c.down + '88',
      })));
    }
    const ml = macd.addLineSeries({ color: c.series[1], lineWidth: 1.5, priceLineVisible: false });
    ml.setData(ind.macd_line);
    if (ind.macd_signal) {
      const sig = macd.addLineSeries({ color: c.series[2], lineWidth: 1, priceLineVisible: false });
      sig.setData(ind.macd_signal);
    }
    charts.macd = macd; charts.macdSeries = ml;
    panes.push({ chart: macd, series: ml });
  }
  if (panes.length > 1) syncPanes(panes);

  applyMarkers();
  initLegend(price, candle);
  price.timeScale().fitContent();
  price.subscribeClick(onChartClick);
}

function applyMarkers() {
  if (!charts) return;
  const lists = [];
  if (state.markers.has('trades')) lists.push(fillMarkers(state.trades));
  lists.push(eventMarkers(state.events, state.markers));
  charts.candle.setMarkers(mergeMarkers(...lists));
}

function initLegend(price, candle) {
  const legend = document.getElementById('sym-legend');
  const c = colors();
  const render = (bar) => {
    if (!bar) { legend.replaceChildren(); return; }
    const dir = bar.close >= bar.open ? 't-up' : 't-down';
    legend.replaceChildren(
      h('span', {}, 'O ', h('b', { class: dir, text: fmtPrice(bar.open) })),
      h('span', {}, 'H ', h('b', { class: dir, text: fmtPrice(bar.high) })),
      h('span', {}, 'L ', h('b', { class: dir, text: fmtPrice(bar.low) })),
      h('span', {}, 'C ', h('b', { class: dir, text: fmtPrice(bar.close) })),
      h('span', {}, 'V ', h('b', { text: fmtCompact(bar.volume) })),
    );
  };
  const bars = state.bars.bars;
  render(bars[bars.length - 1]);
  price.subscribeCrosshairMove(param => {
    const d = param.seriesData && param.seriesData.get(candle);
    render(d && d.open != null ? { ...d, volume: barVolumeAt(param.time) } : bars[bars.length - 1]);
  });
}

const volByTime = () => {
  const m = new Map();
  for (const b of state.bars.bars) m.set(b.time, b.volume);
  return m;
};
let _volMap = null;
function barVolumeAt(time) {
  if (!_volMap || _volMap._src !== state.bars) {
    _volMap = volByTime();
    _volMap._src = state.bars;
  }
  return _volMap.get(time);
}

/* ── fills → chart markers ──
   The desk ledger records individual fills (one row per buy/sell), not
   entry/exit round trips — each fill gets its own arrow. */
function fillMarkers(fills) {
  const c = colors();
  const out = [];
  fills.forEach((t, i) => {
    const time = toEpochSec((t.t || '').slice(0, 10));
    if (!time) return;
    const buy = (t.side || '').toLowerCase() === 'buy';
    out.push({
      time,
      position: buy ? 'belowBar' : 'aboveBar',
      shape: buy ? 'arrowUp' : 'arrowDown',
      color: buy ? c.accent : c.down,
      text: buy ? 'B' : 'S',
      id: `fill:${i}`,
    });
  });
  return out.sort((a, b) => a.time - b.time);
}

/* ── chart click -> fill drawer ── */
function onChartClick(param) {
  const id = param.hoveredObjectId;
  if (!id || typeof id !== 'string') return;
  const m = id.match(/^fill:(\d+)$/);
  if (!m) return;
  const fill = state.trades[Number(m[1])];
  if (fill) openTradeDrawer(fill);
}

function openTradeDrawer(t) {
  const drawer = document.getElementById('trade-drawer');
  document.getElementById('trade-drawer-title').textContent =
    `${t.symbol} · ${(t.side || '').toUpperCase()}`;
  const body = document.getElementById('trade-drawer-body');
  const q = t.fill_quote || null;
  const rows = [
    ['When', fmtDate(t.t)],
    ['Side', (t.side || '').toUpperCase()],
    [t.symbol && t.symbol.length > 6 ? 'Contracts' : 'Shares', t.shares],
    ['Fill price', fmtPrice(t.price)],
    ['Dollars', t.dollars != null ? fmtPnl(Math.abs(t.dollars)) : '—'],
    ['Quote at fill', q && q.bid != null && q.ask != null
      ? `${fmtPrice(q.bid)} / ${fmtPrice(q.ask)} (bid/ask)` : '—'],
    ['Run', t.run_id || '—'],
  ];
  const kv = h('dl', { class: 'c-kv' });
  for (const [k, v] of rows) kv.append(h('dt', { text: k }), h('dd', { text: String(v ?? '—') }));
  clear(body).append(kv);
  if (t.rationale) {
    body.append(h('h4', { class: 'mt-16', text: 'Why the AI did it' }),
                h('p', { class: 't-2', text: t.rationale }));
  }
  drawer.classList.add('open');
}

/* ── right rail ── */
function renderRail() {
  const body = document.getElementById('sym-rail-body');
  const active = document.querySelector('#sym-rail-tabs button.active')?.dataset.tab || 'profile';
  if (active === 'profile') renderProfile(body);
  else if (active === 'options') renderRailOptions(body);
  else if (active === 'trades') renderRailTrades(body);
  else renderRailEvents(body);
}

/* ── options tab: live chain summary + the IV bank ── */
async function renderRailOptions(body) {
  skeleton(body);
  try {
    const [s, hist] = await Promise.all([
      apiGet('/api/desk/options/' + state.symbol),
      apiGet('/api/desk/options/' + state.symbol + '/history').catch(() => null),
    ]);
    clear(body);
    if (!s.available) {
      renderEmpty(body, 'No options data for ' + state.symbol +
        (s.error ? ` (${s.error})` : '') + '.');
      return;
    }
    const f = (v, d = 2) => v != null ? fmtNum(v, d) : '—';
    const kv = (label, value, cls) => h('div', { class: 'sym-opt-kv' },
      h('span', { class: 't-dim', text: label }),
      h('span', { class: 'num ' + (cls || ''), text: value }));
    body.append(
      h('div', { class: 'sym-opt-head' },
        h('span', { text: `Nearest expiry ${s.expiry}` }),
        h('span', { class: 't-dim', text: `${s.dte} DTE` })),
      kv('ATM IV', s.atm_iv != null ? f(s.atm_iv * 100, 1) + '%' : '—'),
      kv('Expected move', s.expected_move_pct != null
        ? `±${f(s.expected_move_pct, 1)}% ($${f(s.expected_move_dollars)})` : '—'),
      kv('25Δ skew (put−call)', s.skew_25d != null
        ? (s.skew_25d >= 0 ? '+' : '') + f(s.skew_25d * 100, 1) + '%' : '—',
        s.skew_25d > 0.02 ? 't-down' : ''),
      kv('Expiries ≤45d', String((s.expiries || []).length)));

    const series = (hist && hist.series) || [];
    body.append(kv('IV bank', `${series.length} day(s) collected`));
    if (series.length >= 2) {
      const prev = series[series.length - 2], cur = series[series.length - 1];
      if (prev.atm_iv != null && cur.atm_iv != null) {
        const d = (cur.atm_iv - prev.atm_iv) * 100;
        body.append(kv('ATM IV vs prior day',
          (d >= 0 ? '+' : '') + f(d, 1) + ' pts', d > 0 ? 't-down' : 't-up'));
      }
    }

    // compact chain around the money: strike | call mid | put mid | IV
    const mids = new Map();
    for (const c of s.calls_table || []) mids.set(c.strike, { c });
    for (const p of s.puts_table || []) mids.set(p.strike, { ...(mids.get(p.strike) || {}), p });
    const mid = r => r && r.bid && r.ask ? (r.bid + r.ask) / 2 : null;
    const atm = s.atm_strike ?? s.spot;
    const rows = [...mids.entries()]
      .sort((a, b) => Math.abs(a[0] - atm) - Math.abs(b[0] - atm))
      .slice(0, 15)                       // rail is narrow — nearest 15 strikes
      .sort((a, b) => a[0] - b[0]);
    if (rows.length) {
      body.append(h('h4', { class: 'mt-16', text: 'Chain (±10% of spot)' }),
        h('table', { class: 'c-table sym-opt-chain' },
          h('thead', {}, h('tr', {},
            h('th', { class: 'num', text: 'Call' }),
            h('th', { class: 'num', text: 'Strike' }),
            h('th', { class: 'num', text: 'Put' }),
            h('th', { class: 'num', text: 'IV' }))),
          h('tbody', {}, ...rows.map(([strike, { c, p }]) => {
            const iv = (c && c.iv) || (p && p.iv);
            return h('tr', { class: strike === s.atm_strike ? 'sym-opt-atm' : '' },
              h('td', { class: 'num', text: f(mid(c)) }),
              h('td', { class: 'num', text: fmtNum(strike, strike % 1 ? 2 : 0) }),
              h('td', { class: 'num', text: f(mid(p)) }),
              h('td', { class: 'num t-dim', text: iv != null ? f(iv * 100, 1) + '%' : '—' }));
          }))));
    }
  } catch (err) { renderError(body, err, () => renderRailOptions(body)); }
}

/* ── Research tab: SEC EDGAR point-in-time fundamentals (public domain) ──
   Plain-English labels; the technical term rides in the tooltip. Every
   number's provenance is stated: which filing, and which price. */
const RESEARCH_FIELDS = [
  // [key, label, formatter, tooltip]
  ['market_cap', 'Company value', v => '$' + fmtCompact(v),
    'Market capitalization: share price × shares outstanding'],
  ['price_to_earnings', 'Price vs profit', v => v.toFixed(1) + '×',
    'P/E ratio: how many dollars you pay per dollar of yearly profit (no figure when the company runs a loss)'],
  ['price_to_sales', 'Price vs sales', v => v.toFixed(2) + '×',
    'P/S ratio: company value relative to a year of revenue'],
  ['price_to_book', 'Price vs net worth', v => v.toFixed(2) + '×',
    'P/B ratio: price relative to accounting net worth (book value)'],
  ['ev_to_ebitda', 'Value vs core earnings', v => v.toFixed(1) + '×',
    'EV/EBITDA: total company value (incl. debt) vs earnings before interest, tax, depreciation'],
  ['fcf_yield', 'Cash generated', v => fmtPct(v * 100, { signed: false }),
    'Free-cash-flow yield: spare cash produced per year as a share of company value'],
  ['earnings_per_share', 'Profit per share', v => '$' + v.toFixed(2),
    'Trailing-twelve-month net income divided by shares outstanding'],
  ['return_on_equity', 'Profit on net worth', v => fmtPct(v * 100, { signed: false }),
    'ROE: yearly profit as a share of the owners’ stake'],
  ['revenue_growth', 'Sales growth (yr)', v => fmtPct(v * 100),
    'Trailing 12 months of revenue vs the 12 months before'],
  ['earnings_growth', 'Profit growth (yr)', v => fmtPct(v * 100),
    'Trailing 12 months of profit vs the 12 months before'],
  ['debt_to_equity', 'Debt load', v => v.toFixed(2) + '×',
    'Total debt relative to the owners’ stake — blank means the company reports no debt'],
  ['current_ratio', 'Bills coverage', v => v.toFixed(2) + '×',
    'Current ratio: short-term assets vs bills due within a year'],
];

function renderProfile(body) {
  const p = state.profile;
  if (!p) { renderEmpty(body, 'Research data unavailable right now.'); return; }
  if (!p.covered) { renderEmpty(body, p.note || 'No company filings for this symbol.'); return; }
  clear(body);

  const s = p.snapshot || {};
  const kv = h('dl', { class: 'c-kv' });
  for (const [key, label, fmt, tip] of RESEARCH_FIELDS) {
    const v = s[key];
    if (v == null) continue;
    kv.append(h('dt', { text: label, title: tip }),
              h('dd', { class: 'num', text: String(fmt(v)) }));
  }
  body.append(kv);

  // Trend lines across every filing since 2009: sales and profit-per-share.
  const revs = (p.series || []).map(r => r._revenue_ttm).filter(v => v != null);
  const eps = (p.series || []).map(r => r.earnings_per_share).filter(v => v != null);
  const trend = (label, values, tip) => values.length >= 4
    ? h('div', { class: 'sym-research-trend', title: tip },
        h('span', { class: 't-dim', text: label }), sparkline(values, { size: 'lg' }))
    : null;
  const t1 = trend('Yearly sales since ’09', revs,
    'Trailing-twelve-month revenue at each SEC filing');
  const t2 = trend('Profit/share since ’09', eps,
    'Trailing-twelve-month earnings per share at each SEC filing (split-affected eras show level shifts)');
  if (t1) body.append(t1);
  if (t2) body.append(t2);

  body.append(h('p', { class: 't-dim sym-research-src', text:
    `Source: ${p.filings} SEC filings (point-in-time), ` +
    `latest ${p.latest_form || 'filing'} on ${p.latest_filed}` +
    (p.price_used ? ` · priced at $${fmtNum(p.price_used, 2)} (${p.price_as_of})` : '') }));
}

function renderRailTrades(body) {
  if (!state.trades.length) {
    renderEmpty(body, 'The AI hasn’t traded this symbol yet.');
    return;
  }
  const list = h('div', { class: 'flex-col gap-4' });
  for (const t of state.trades) {  // endpoint returns newest first
    const buy = (t.side || '').toLowerCase() === 'buy';
    const unit = t.symbol && t.symbol.length > 6 ? 'ct' : 'sh';
    const row = h('div', {
      class: 'c-tl clickable',
      onclick: () => {
        openTradeDrawer(t);
        const e = toEpochSec((t.t || '').slice(0, 10));
        if (e && charts) {
          charts.price.timeScale().setVisibleRange({ from: e - 45 * 86400, to: e + 45 * 86400 });
        }
      },
    },
      h('div', { class: 'when', text: fmtDate(t.t) }),
      h('div', { class: 'what' },
        h('span', { class: 'num ' + (buy ? 't-up' : 't-down'), text: buy ? 'BUY ' : 'SELL ' }),
        h('span', { class: 'num', text: `${t.shares} ${unit} @ ${fmtPrice(t.price)}` }),
      ),
    );
    list.append(row);
  }
  clear(body).append(list);
}

function renderRailEvents(body) {
  const ev = state.events;
  const list = h('div', { class: 'flex-col gap-4' });
  const items = [
    ...(ev.dividends || []).map(d => ({ time: d.time, label: `Dividend ${fmtPrice(d.cash_amount)}` })),
    ...(ev.splits || []).map(s => ({ time: s.time, label: `Split ${s.ratio}` })),
    ...(ev.news || []).map(n => ({ time: n.time, label: n.title, url: n.url })),
  ].sort((a, b) => b.time - a.time).slice(0, 60);
  if (!items.length) { renderEmpty(body, 'No events'); return; }
  for (const it of items) {
    list.append(h('div', { class: 'c-tl' },
      h('div', { class: 'when', text: fmtDate(it.time) }),
      h('div', { class: 'what' },
        it.url ? h('a', { href: it.url, target: '_blank', rel: 'noopener', text: it.label })
               : h('span', { text: it.label }),
      ),
    ));
  }
  clear(body).append(list);
}

/* ── toggles + search ── */
function renderToggles() {
  const indEl = document.getElementById('sym-ind-toggles');
  indEl.replaceChildren(...IND_DEFS.map(def => h('button', {
    class: 'c-chip' + (state.inds.has(def.key) ? ' active' : ''),
    text: def.label,
    onclick: (e) => {
      state.inds.has(def.key) ? state.inds.delete(def.key) : state.inds.add(def.key);
      e.target.classList.toggle('active');
      writeUrl();
      buildCharts();
    },
  })));
  const mkEl = document.getElementById('sym-marker-toggles');
  mkEl.replaceChildren(...MARKER_DEFS.map(def => h('button', {
    class: 'c-chip' + (state.markers.has(def.key) ? ' active' : ''),
    text: def.label,
    onclick: (e) => {
      state.markers.has(def.key) ? state.markers.delete(def.key) : state.markers.add(def.key);
      e.target.classList.toggle('active');
      writeUrl();
      applyMarkers();
    },
  })));
}

function initSearch() {
  const input = document.getElementById('sym-search-input');
  const results = document.getElementById('sym-search-results');
  const recents = () => JSON.parse(localStorage.getItem('ef-recent-symbols') || '[]');

  const show = (items, hint) => {
    results.classList.remove('hidden');
    results.replaceChildren(
      ...(hint ? [h('div', { class: 'hint', text: hint })] : []),
      ...items.map(it => h('div', {
        class: 'item',
        onclick: () => pick(it.symbol),
      },
        h('span', { class: 's', text: it.symbol }),
        h('span', { class: 'n', text: it.company_name || '' }),
      )),
    );
  };
  const hide = () => results.classList.add('hidden');

  const pick = (symbol) => {
    hide();
    input.value = '';
    const r = [symbol, ...recents().filter(s => s !== symbol)].slice(0, 8);
    localStorage.setItem('ef-recent-symbols', JSON.stringify(r));
    state.symbol = symbol.toUpperCase();
    loadSymbol();
  };

  const search = debounce(async (q) => {
    try {
      const data = await apiGet(`/api/symbols/search?q=${encodeURIComponent(q)}&limit=10`);
      const items = data.results || data || [];
      show(items.length ? items : [], items.length ? null : 'No matches — Enter to load anyway');
    } catch { hide(); }
  }, 250);

  input.addEventListener('input', () => {
    const q = input.value.trim();
    if (q.length >= 2) search(q);
    else if (!q) show(recents().map(s => ({ symbol: s })), recents().length ? 'Recent' : 'Type to search');
    else hide();
  });
  input.addEventListener('focus', () => {
    if (!input.value.trim()) show(recents().map(s => ({ symbol: s })), recents().length ? 'Recent' : 'Type to search');
  });
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && input.value.trim()) pick(input.value.trim().toUpperCase());
    if (e.key === 'Escape') hide();
  });
  document.addEventListener('click', (e) => {
    if (!e.target.closest('.sym-search')) hide();
  });
}

/* ── boot ── */
function init() {
  parseUrl();
  renderToggles();
  initSearch();

  rangeSwitcher(document.getElementById('sym-ranges'), RANGES, state.range, (value) => {
    state.range = value;
    loadSymbol();
  });
  fullscreenButton(document.getElementById('sym-fullscreen'), document.getElementById('sym-shell'));

  document.getElementById('sym-rail-tabs').addEventListener('click', (e) => {
    const btn = e.target.closest('button');
    if (!btn) return;
    for (const b of document.querySelectorAll('#sym-rail-tabs button')) b.classList.remove('active');
    btn.classList.add('active');
    renderRail();
  });
  document.getElementById('trade-drawer-close').addEventListener('click', () =>
    document.getElementById('trade-drawer').classList.remove('open'));

  onThemeChange(() => { if (state.bars) buildCharts(); });

  skeleton(document.getElementById('sym-rail-body'));
  loadSymbol();
}

document.addEventListener('DOMContentLoaded', init);

/* Picks — the AI analyst account's daily report, chart-forward.
   Each pick leads with its price chart + entry setup (EMA200 overlay,
   today's entry marker), then the proof (rule track records), the why-now
   metrics, the rationale, and news. Reads the persisted decision from
   /api/picks; charts come live from the existing /api/symbols bars endpoint. */

import { apiGet, apiPost } from '../core/net.js';
import { toEpochSec, fmtPct, fmtNum, fmtPrice, fmtDate } from '../core/fmt.js';
import { h, clear, skeleton, renderError, renderEmpty } from '../core/dom.js';
import { createChart, colors } from '../core/charts.js';

const STRATEGY = 'ai_analyst';
const _charts = [];

const ACTION_CLASS = { buy: 't-up', add: 't-up', hold: 't-dim', trim: 't-warn', sell: 't-down' };
const ACTION_LABEL = { buy: 'BUY', add: 'ADD', hold: 'HOLD', trim: 'TRIM', sell: 'SELL' };

function destroyCharts() {
  for (const c of _charts) c?.__efDestroy?.();
  _charts.length = 0;
}

function pill(text, cls) {
  return h('span', { class: 'c-pill ' + (cls || 'neutral'), text });
}

/* ── load + render ── */
async function load() {
  const sumEl = document.getElementById('picks-summary');
  const listEl = document.getElementById('picks-list');
  const dateEl = document.getElementById('picks-date');
  skeleton(sumEl);
  skeleton(listEl);
  destroyCharts();

  let data;
  try {
    data = await apiGet(`/api/picks/latest?strategy=${STRATEGY}`);
  } catch (err) {
    renderError(sumEl, err, load);
    clear(listEl);
    return;
  }

  if (!data.decision_date) {
    dateEl.textContent = 'no decision yet';
    renderEmpty(sumEl, 'The analyst has not produced a decision yet — click “Run research now”.');
    clear(listEl);
    return;
  }

  dateEl.textContent = `as of ${fmtDate(data.decision_date)}`;
  renderSummary(sumEl, data);
  renderPicks(listEl, data);
}

function renderSummary(el, data) {
  const c = data.counts || {};
  clear(el).append(
    h('div', { class: 'flex items-center gap-12 flex-wrap' },
      pill(`${c.holdings || 0} held`),
      pill(`${c.new || 0} new`, 't-up'),
      pill(`${c.sells || 0} sells`, 't-down'),
      data.model ? pill(data.model, 'neutral') : null,
    ),
    h('p', { class: 't-2 mt-8', text: data.summary || '' }),
  );
}

function renderPicks(listEl, data) {
  clear(listEl);
  const picks = data.picks || [];
  if (!picks.length) {
    renderEmpty(listEl, 'No picks in this decision (all cash).');
    return;
  }
  for (const p of picks) listEl.append(card(p, data.decision_date));
}

function card(p, decisionDate) {
  const chartHost = h('div', { class: 'ch-pane mini' });
  const el = h('div', { class: 'c-card' },
    h('div', { class: 'ch-shell' }, chartHost),
    h('div', { class: 'c-card-body' }, cardBody(p)),
  );
  buildPickChart(chartHost, p.symbol, decisionDate, p.action);
  return el;
}

function cardBody(p) {
  const frag = h('div', {},
    h('div', { class: 'flex items-center gap-8 flex-wrap' },
      h('strong', { class: 'num', text: p.symbol }),
      pill(ACTION_LABEL[p.action] || p.action, ACTION_CLASS[p.action]),
      p.composite != null ? h('span', { class: 't-dim', text: `score ${fmtNum(p.composite, 2)}` }) : null,
      p.target_weight != null ? h('span', { class: 't-dim', text: `· ${fmtPct(p.target_weight * 100, { signed: false })}` }) : null,
    ),
    h('p', { class: 't-2 mt-8', text: p.rationale || '' }),
  );

  // firing rules
  const rules = p.rules || [];
  if (rules.length) {
    frag.append(h('div', { class: 'flex gap-4 flex-wrap mt-8' },
      ...rules.map(r => h('span', { class: 'c-chip', text: r }))));
  }

  // proof: each firing rule's backtested track record
  const proof = p.proof || {};
  const proofRules = Object.keys(proof);
  if (proofRules.length) {
    const kv = h('dl', { class: 'c-kv mt-8' });
    for (const r of proofRules) {
      const t = proof[r];
      kv.append(
        h('dt', { text: `${r} (backtest)` }),
        h('dd', { text: `${fmtPct(t.return_pct)} · Sharpe ${fmtNum(t.sharpe, 2)} · maxDD ${fmtPct(t.max_drawdown_pct, { signed: false })}` }),
      );
    }
    frag.append(h('div', { class: 'mt-8' },
      h('div', { class: 't-dim', text: 'Proof — buy when this rule fires, hold, monthly:' }), kv));
  }

  // why-now metrics
  const m = p.metrics || {};
  const metricRows = [
    ['Price', m.price != null ? fmtPrice(m.price) : null],
    ['6-mo return', m.ret_126 != null ? fmtPct(m.ret_126 * 100) : null],
    ['1-mo return', m.ret_20 != null ? fmtPct(m.ret_20 * 100) : null],
    ['RSI', m.rsi != null ? fmtNum(m.rsi, 0) : null],
    ['vs 200-EMA', m.price_over_ema200 != null ? fmtPct((m.price_over_ema200 - 1) * 100) : null],
    ['vs 52w high', m.high_ratio != null ? fmtPct((m.high_ratio - 1) * 100) : null],
  ].filter(r => r[1] != null);
  if (metricRows.length) {
    const kv = h('dl', { class: 'c-kv mt-8' });
    for (const [k, v] of metricRows) kv.append(h('dt', { text: k }), h('dd', { text: v }));
    frag.append(kv);
  }

  // news
  const news = p.news || [];
  if (news.length) {
    const list = h('div', { class: 'flex-col gap-4 mt-8' });
    for (const n of news.slice(0, 3)) {
      list.append(h('div', { class: 'c-tl' },
        h('div', { class: 'what' }, h('span', { class: 't-2', text: n.title || '' }))));
    }
    frag.append(h('div', { class: 'mt-8' },
      h('div', { class: 't-dim', text: 'Recent news' }), list));
  }

  return frag;
}

async function buildPickChart(host, symbol, decisionDate, action) {
  try {
    const data = await apiGet(`/api/symbols/${symbol}/bars?range=6m&indicators=true`, { timeout: 20000 });
    if (!data.bars || !data.bars.length) { renderEmpty(host, 'No chart data'); return; }
    const c = colors();
    const chart = createChart(host);
    _charts.push(chart);
    const candle = chart.addCandlestickSeries({
      upColor: c.up, downColor: c.down, wickUpColor: c.up, wickDownColor: c.down,
      borderVisible: false,
    });
    candle.setData(data.bars);
    const ind = data.indicators || {};
    if (ind.ema_200) {
      const line = chart.addLineSeries({
        color: c.series[4], lineWidth: 1, priceLineVisible: false, lastValueVisible: false,
      });
      line.setData(ind.ema_200);
    }
    const t = toEpochSec(decisionDate);
    if (t != null) {
      const sell = action === 'sell';
      candle.setMarkers([{
        time: t,
        position: sell ? 'aboveBar' : 'belowBar',
        shape: sell ? 'arrowDown' : 'arrowUp',
        color: sell ? c.down : c.accent,
        text: (ACTION_LABEL[action] || action || '').toString(),
      }]);
    }
    chart.timeScale().fitContent();
  } catch (err) {
    renderError(host, err);
  }
}

/* ── run-now ── */
function initRunButton() {
  const btn = document.getElementById('picks-run');
  btn.addEventListener('click', async () => {
    btn.disabled = true;
    const label = btn.textContent;
    btn.textContent = 'Research started…';
    try {
      await apiPost('/api/picks/run', {});
    } catch (err) {
      console.error(err);
    }
    // the research runs in a background thread; re-check shortly
    setTimeout(() => { btn.disabled = false; btn.textContent = label; load(); }, 30000);
  });
}

function init() {
  initRunButton();
  load();
}

document.addEventListener('DOMContentLoaded', init);

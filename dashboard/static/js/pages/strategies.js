/* Strategies — DB-driven cards (zero hardcoded names/risk), detail drawer:
   equity curve, scorecard, dividend-credit ledger, parameter audit,
   validation linkage. All accounts are v2 now (arena lane retired). */

import { apiGet } from '../core/net.js';
import { fmtDollar, fmtPnl, fmtPct, fmtNum, fmtInt, fmtDate, upDownClass } from '../core/fmt.js';
import { h, clear, skeleton, renderError, renderEmpty, panel } from '../core/dom.js';
import { createChart, colors } from '../core/charts.js';

const state = { meta: new Map(), accounts: [] };
let detailChart = null;

async function loadCards() {
  const el = document.getElementById('st-cards');
  await panel(el, () => apiGet('/api/strategies/accounts'), (host, accounts) => {
    state.accounts = accounts;
    const rows = accounts;
    if (!rows.length) { renderEmpty(host, 'No strategies yet'); return; }
    clear(host);
    for (const a of rows) {
      const name = a.strategy_name || a.name;
      const m = state.meta.get(name) || {};
      const pnl = (a.total_equity || 0) - (a.starting_capital || 0);
      host.append(h('div', { class: 'c-card clickable', onclick: () => openDetail(name) },
        h('div', { class: 'c-card-header' },
          h('span', { class: `c-dot s${(m.color_slot ?? 7) % 8}` }),
          h('span', { class: 'num', text: name }),
          h('span', { class: `c-pill ${m.tier === 'validated' ? 'info' : 'neutral'}`, text: m.tier || 'v2' }),
          a.pending ? h('span', { class: 'c-pill neutral', text: 'AWAITING FIRST CYCLE' }) : null,
          a.is_paused ? h('span', { class: 'c-pill warn', text: 'PAUSED' }) : null,
          h('span', { class: 'spacer' })),
        h('div', { class: 'c-card-body' },
          h('div', { class: 'flex items-end gap-12' },
            h('div', { class: 'c-stat flex-1' },
              h('div', { class: 'label', text: 'Equity' }),
              h('div', { class: 'value num', text: fmtDollar(a.total_equity) }),
              h('div', { class: 'delta num ' + upDownClass(pnl), text: fmtPnl(pnl) })),
            h('div', { class: 'c-stat flex-1' },
              h('div', { class: 'label', text: 'Cash' }),
              h('div', { class: 'value num', text: fmtDollar(a.cash ?? a.cash_balance) })),
            h('div', { class: 'c-stat flex-1' },
              h('div', { class: 'label', text: 'Drawdown' }),
              h('div', { class: 'value num ' + ((a.drawdown_pct || 0) > 10 ? 't-down' : ''), text: fmtPct(a.drawdown_pct, { signed: false }) })),
          ))));
    }
  });
}

async function openDetail(name) {
  const drawer = document.getElementById('st-drawer');
  const body = document.getElementById('st-drawer-body');
  document.getElementById('st-drawer-title').textContent = name;
  drawer.classList.add('open');
  skeleton(body, 'chart');

  const [curve, score, stats, divs, params, validation] = await Promise.allSettled([
    apiGet(`/api/strategies/equity-curve?days=90&strategy=${name}`),
    apiGet(`/api/strategies/scorecard?days=90&strategy=${name}`),
    apiGet(`/api/trades/stats?strategy=${name}`),
    apiGet(`/api/strategies/dividends?strategy=${name}`),
    apiGet(`/api/strategies/params?strategy=${name}`),
    apiGet('/api/strategies/validation'),
  ]);
  clear(body);

  // equity chart
  const chartHost = h('div', { class: 'ch-pane mini mb-16' });
  body.append(chartHost);
  detailChart?.__efDestroy?.();
  if (curve.status === 'fulfilled' && (curve.value[name] || []).length) {
    const c = colors();
    const m = state.meta.get(name) || {};
    detailChart = createChart(chartHost, { height: 160 });
    const line = detailChart.addAreaSeries({
      lineColor: c.series[(m.color_slot ?? 0) % 8],
      topColor: c.series[(m.color_slot ?? 0) % 8] + '33',
      bottomColor: 'transparent', lineWidth: 2,
    });
    line.setData(curve.value[name].map(p => ({ time: p.time, value: p.total_equity })));
    detailChart.timeScale().fitContent();
  } else {
    renderEmpty(chartHost, 'No equity snapshots');
  }

  // trade stats + scorecard
  if (stats.status === 'fulfilled') {
    const s = stats.value;
    const kv = h('dl', { class: 'c-kv mb-16' });
    for (const [k, v] of [
      ['Trades', `${fmtInt(s.total_trades)} (${fmtInt(s.open_trades)} open)`],
      ['Win rate', s.win_rate != null ? fmtPct(s.win_rate, { signed: false }) : '—'],
      ['Total P&L', fmtPnl(s.total_pnl)],
      ['Profit factor', fmtNum(s.profit_factor)],
      ['Avg R', fmtNum(s.avg_r_multiple, 2)],
    ]) kv.append(h('dt', { text: k }), h('dd', { text: String(v) }));
    body.append(h('h4', { text: 'Trading' }), kv);
  }
  if (score.status === 'fulfilled' && score.value[0]) {
    const cdt = score.value[0];
    const crit = cdt.criteria || {};
    body.append(h('h4', { text: 'Live proof (90d)' }),
      h('div', { class: 'flex items-center gap-8 mb-16 flex-wrap' },
        h('span', { class: 'num', text: `Sharpe ${fmtNum(cdt.sharpe)}` }),
        h('span', { class: 'num ' + upDownClass(cdt.excess_vs_spy_pct),
                    text: cdt.excess_vs_spy_pct != null ? fmtPct(cdt.excess_vs_spy_pct) + ' vs SPY' : '—' }),
        h('span', { class: `c-pill ${crit.all_met ? 'up' : 'down'}`, text: crit.all_met ? 'PASS' : 'FAIL' })));
  }

  // offline validation linkage
  if (validation.status === 'fulfilled') {
    const v = validation.value.find(x => x.strategy_name === name || x.strategy_name?.includes(name));
    if (v) {
      body.append(h('h4', { text: 'Offline validation' }),
        h('div', { class: 'flex items-center gap-8 mb-16 flex-wrap' },
          h('span', { class: 't-2', text: `${fmtDate(v.run_at)} · ${v.universe || ''}` }),
          h('span', { class: `c-pill ${v.validated ? 'up' : 'neutral'}`,
                      text: v.validated ? 'VALIDATED' : v.verdict || '—' }),
          h('a', { class: 'c-chip', href: '/lab?tab=runs', text: 'Lab →' })));
    }
  }

  // dividend ledger (v2)
  if (divs.status === 'fulfilled' && divs.value.length) {
    const tbody = h('tbody');
    for (const d of divs.value.slice(0, 20)) {
      tbody.append(h('tr', {},
        h('td', { class: 'num', text: d.ex_date }),
        h('td', { class: 'num', text: d.symbol }),
        h('td', { class: 'num', text: fmtInt(d.shares) }),
        h('td', { class: 'num t-up', text: fmtPnl(d.amount) })));
    }
    body.append(h('h4', { text: 'Dividend credits' }),
      h('div', { class: 'c-table-wrap mb-16' }, h('table', { class: 'c-table' },
        h('thead', {}, h('tr', {},
          h('th', { text: 'Ex-date' }), h('th', { text: 'Symbol' }),
          h('th', { class: 'num', text: 'Shares' }), h('th', { class: 'num', text: 'Credit' }))),
        tbody)));
  }

  // parameter audit
  if (params.status === 'fulfilled' && params.value.length) {
    const list = h('div', { class: 'flex-col gap-4' });
    for (const p of params.value.slice(0, 15)) {
      list.append(h('div', { class: 'c-tl' },
        h('div', { class: 'when', text: fmtDate(p.changed_at) }),
        h('div', { class: 'what' },
          h('span', { class: 'num', text: `${p.param_name}: ${p.old_value} → ${p.new_value}` }),
          h('div', { class: 'sub', text: `by ${p.changed_by}` }))));
    }
    body.append(h('h4', { text: 'Parameter changes' }), list);
  }
}

async function init() {
  try {
    const meta = await apiGet('/api/strategies/meta');
    for (const m of meta) state.meta.set(m.name, m);
  } catch { /* cards render without slots */ }

  document.getElementById('st-drawer-close').addEventListener('click', () =>
    document.getElementById('st-drawer').classList.remove('open'));

  loadCards();
}

document.addEventListener('DOMContentLoaded', init);

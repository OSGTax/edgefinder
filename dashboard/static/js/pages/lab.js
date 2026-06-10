/* Hunt/Lab Explorer — scoreboard toward the 10-finalist goal, the full
   validation_runs browser (filters, detail drawer, compare ≤4), and the
   legacy quick-backtest tab rebuilt on net.js/poll.js (bounded polling). */

import { apiGet, apiPost } from '../core/net.js';
import { fmtPct, fmtNum, fmtInt, fmtDate, fmtDollar, upDownClass, toEpochSec } from '../core/fmt.js';
import { h, clear, skeleton, renderError, renderEmpty, panel } from '../core/dom.js';
import { createChart, colors } from '../core/charts.js';
import { poller } from '../core/poll.js';

/* ════ tabs ════ */
function activeTab() {
  return new URLSearchParams(location.search).get('tab') || 'scoreboard';
}

function switchTab(tab) {
  for (const b of document.querySelectorAll('#lab-tabs button')) {
    b.classList.toggle('active', b.dataset.tab === tab);
  }
  for (const id of ['scoreboard', 'runs', 'backtest']) {
    document.getElementById(`tab-${id}`).classList.toggle('hidden', id !== tab);
  }
  const q = new URLSearchParams(location.search);
  q.set('tab', tab);
  history.replaceState(null, '', `/lab?${q}`);
  if (tab === 'scoreboard') loadScoreboard();
  if (tab === 'runs') loadRuns();
}

/* ════ scoreboard ════ */
function holdoutPill(status) {
  if (status === 'pass') return h('span', { class: 'c-pill up', text: 'HOLDOUT PASS' });
  if (status === 'fail') return h('span', { class: 'c-pill down', text: 'HOLDOUT FAIL' });
  return h('span', { class: 'c-pill sealed', text: '🔒 SEALED' });
}

function fracBar(frac) {
  const m = /^(\d+)\s*\/\s*(\d+)$/.exec(frac || '');
  if (!m) return h('span', { class: 'num', text: frac || '—' });
  const pct = (+m[1] / +m[2]) * 100;
  const bar = h('span', { class: 'c-frac' },
    h('span', { class: 'num', text: frac }),
    h('span', { class: 'bar' }, h('i', { class: `w${Math.round(pct / 10) * 10}` })));
  return bar;
}

async function loadScoreboard() {
  const progressEl = document.getElementById('sb-progress-body');
  const gridEl = document.getElementById('sb-finalists');
  await panel(progressEl, () => apiGet('/api/lab/scoreboard'), (el, sb) => {
    const solid = sb.counts.holdout_passed;
    const hatch = sb.counts.criteria_passing - solid;
    const bar = h('div', { class: 'c-progress mt-8' });
    const seg = (cls, n) => {
      if (n <= 0) return null;
      const s = h('i', { class: cls });
      s.classList.add(`w${Math.round((n / sb.target) * 100 / 10) * 10}`);
      return s;
    };
    [seg('seg-solid', solid), seg('seg-hatch', hatch)].forEach(x => x && bar.append(x));
    clear(el).append(
      h('div', { class: 'flex items-center gap-12 flex-wrap' },
        h('div', { class: 'c-stat' },
          h('div', { class: 'label', text: 'Finalists toward goal' }),
          h('div', { class: 'value hero num', text: `${sb.counts.criteria_passing} / ${sb.target}` })),
        h('div', { class: 'c-stat' },
          h('div', { class: 'label', text: 'Holdout passed' }),
          h('div', { class: 'value num', text: String(sb.counts.holdout_passed) })),
        h('div', { class: 'c-stat' },
          h('div', { class: 'label', text: 'Paper trading' }),
          h('div', { class: 'value num', text: String(sb.counts.promoted) })),
      ),
      bar,
      h('div', { class: 't-dim mt-8', text:
        'Finalist = newest run passes all criteria. Solid = sealed holdout also passed; hatched = holdout still sealed (one look, owner-authorized).' }),
    );

    clear(gridEl);
    if (!sb.finalists.length) { renderEmpty(gridEl, 'No criteria-passing runs yet — the hunt continues.'); return; }
    for (const f of sb.finalists) {
      gridEl.append(h('div', { class: 'c-card' },
        h('div', { class: 'c-card-header' },
          h('span', { class: 'num', text: f.strategy_name }),
          f.promoted ? h('span', { class: 'c-pill info', text: `PAPER · ${f.tier || ''}` }) : null,
          h('span', { class: 'spacer' }),
          holdoutPill(f.holdout_status)),
        h('div', { class: 'c-card-body' },
          h('dl', { class: 'c-kv' },
            h('dt', { text: 'Excess vs SPY / fold' }),
            h('dd', { class: upDownClass(f.mean_excess_vs_spy_pct), text: fmtPct(f.mean_excess_vs_spy_pct) }),
            h('dt', { text: 'Folds beating SPY' }), h('dd', { text: f.folds_beating_spy || '—' }),
            h('dt', { text: 'Mean Sharpe' }), h('dd', { text: fmtNum(f.mean_sharpe) }),
            h('dt', { text: 'Universe' }), h('dd', { text: f.universe || '—' }),
            h('dt', { text: 'Run' }), h('dd', { text: `${fmtDate(f.run_at)} · ${f.git_sha || ''}` }),
          ),
          h('button', { class: 'c-btn ghost mt-8', text: 'Open run', onclick: () => openRun(f.id) }),
        )));
    }
  }, { skeletonKind: 'block' });
}

/* ════ runs browser ════ */
const runsState = { offset: 0, limit: 50, compare: new Set() };

function runFilters() {
  const q = new URLSearchParams();
  const label = document.getElementById('rf-label').value;
  const strategy = document.getElementById('rf-strategy').value.trim();
  const verdict = document.getElementById('rf-verdict').value;
  const holdout = document.getElementById('rf-holdout').value;
  if (label) q.set('label', label);
  if (strategy) q.set('strategy', strategy);
  if (verdict) q.set('verdict', verdict);
  if (holdout) q.set('holdout', holdout);
  q.set('limit', runsState.limit);
  q.set('offset', runsState.offset);
  return q;
}

async function loadRuns() {
  const tableEl = document.getElementById('runs-table');
  await panel(tableEl, () => apiGet(`/api/lab/runs?${runFilters()}`), (el, data) => {
    if (!data.runs.length) { renderEmpty(el, 'No runs match'); return; }
    const thead = h('tr', {},
      h('th', { text: '' }),
      h('th', { text: 'Run' }), h('th', { text: 'Strategy' }),
      h('th', { text: 'Universe' }), h('th', { text: 'Costs' }),
      h('th', { class: 'num', text: 'Excess/fold' }),
      h('th', { class: 'num', text: 'Folds>SPY' }),
      h('th', { class: 'num', text: 'ExSharpe' }),
      h('th', { class: 'num', text: 'Trades' }),
      h('th', { text: 'Verdict' }), h('th', { text: 'Holdout' }));
    const tbody = h('tbody');
    for (const r of data.runs) {
      const cb = h('input', { type: 'checkbox' });
      cb.checked = runsState.compare.has(r.id);
      cb.addEventListener('change', () => {
        if (cb.checked) {
          if (runsState.compare.size >= 4) { cb.checked = false; return; }
          runsState.compare.add(r.id);
        } else runsState.compare.delete(r.id);
        updateCompareBtn();
      });
      const row = h('tr', { class: 'clickable' },
        h('td', {}, cb),
        h('td', { class: 'num', text: fmtDate(r.run_at) }),
        h('td', { class: 'num', text: r.strategy_name }),
        h('td', { text: r.universe || '—' }),
        h('td', { text: r.cost_label || '—' }),
        h('td', { class: 'num ' + upDownClass(r.mean_excess_vs_spy_pct), text: fmtPct(r.mean_excess_vs_spy_pct) }),
        h('td', { class: 'num', text: r.folds_beating_spy || '—' }),
        h('td', { class: 'num ' + upDownClass(r.mean_excess_sharpe), text: fmtNum(r.mean_excess_sharpe) }),
        h('td', { class: 'num', text: fmtInt(r.total_trades) }),
        h('td', {}, h('span', { class: `c-pill ${r.verdict === 'PASS' ? 'up' : 'down'}`, text: r.verdict })),
        h('td', {}, holdoutPill(r.holdout_status)),
      );
      row.addEventListener('click', (e) => {
        if (e.target.tagName !== 'INPUT') openRun(r.id);
      });
      tbody.append(row);
    }
    clear(el).append(h('table', { class: 'c-table' }, h('thead', {}, thead), tbody));

    const pager = document.getElementById('runs-pager');
    clear(pager).append(
      h('button', {
        class: 'c-btn ghost', text: '← Prev', disabled: runsState.offset === 0 ? '' : null,
        onclick: () => { runsState.offset = Math.max(0, runsState.offset - runsState.limit); loadRuns(); },
      }),
      h('span', { class: 't-dim num', text: `${runsState.offset + 1}–${Math.min(runsState.offset + runsState.limit, data.total)} of ${data.total}` }),
      h('button', {
        class: 'c-btn ghost', text: 'Next →',
        disabled: runsState.offset + runsState.limit >= data.total ? '' : null,
        onclick: () => { runsState.offset += runsState.limit; loadRuns(); },
      }),
    );
  });
}

function updateCompareBtn() {
  const btn = document.getElementById('rf-compare');
  btn.textContent = `Compare (${runsState.compare.size})`;
  btn.classList.toggle('hidden', runsState.compare.size < 2);
}

/* ── run detail drawer ── */
const CRITERIA_LABELS = {
  sharpe_beats_spy: 'Beats SPY Sharpe (mean)',
  majority_folds_higher_sharpe: 'Majority folds higher Sharpe',
  lower_drawdown_than_spy: 'Lower drawdown than SPY',
  traded: 'Traded (≥3 fills)',
  oos_sharpe_positive: 'OOS Sharpe positive',
  beats_spy_majority_folds: 'Beats SPY in majority of folds',
  mean_excess_positive: 'Mean excess return positive',
  min_trades_met: 'Minimum trades met',
};

async function openRun(id) {
  const drawer = document.getElementById('run-drawer');
  const body = document.getElementById('run-drawer-body');
  drawer.classList.add('open');
  skeleton(body);
  let d;
  try {
    d = await apiGet(`/api/lab/runs/${id}`);
  } catch (err) {
    renderError(body, err, () => openRun(id));
    return;
  }
  document.getElementById('run-drawer-title').textContent =
    `${d.strategy_name} · #${d.id}`;
  clear(body);

  // criteria
  const crit = d.criteria || {};
  const critList = h('div', { class: 'flex-col gap-4 mb-16' });
  critList.append(h('div', { class: 't-dim', text: `Criteria mode: ${crit.mode || '—'}` }));
  for (const [k, v] of Object.entries(crit)) {
    if (typeof v !== 'boolean' || k === 'all_met') continue;
    critList.append(h('div', { class: 'flex items-center gap-8' },
      h('span', { class: `c-pill ${v ? 'up' : 'down'}`, text: v ? 'PASS' : 'FAIL' }),
      h('span', { text: CRITERIA_LABELS[k] || k })));
  }
  critList.append(h('div', { class: 'flex items-center gap-8 mt-8' },
    h('span', { class: `c-pill ${crit.all_met ? 'up' : 'down'}`, text: crit.all_met ? 'ALL MET' : 'NOT MET' }),
    holdoutPill(d.holdout_status)));
  body.append(h('h4', { text: 'Criteria' }), critList);

  // OOS stats
  const oos = d.oos || {};
  const kv = h('dl', { class: 'c-kv mb-16' });
  const rows = [
    ['Mean excess vs SPY / fold', fmtPct(oos.mean_excess_vs_spy_pct)],
    ['Folds beating SPY', oos.folds_beating_spy],
    ['Mean excess Sharpe', fmtNum(oos.mean_excess_sharpe)],
    ['Folds higher Sharpe', oos.folds_higher_sharpe],
    ['Mean fold Sharpe', fmtNum(oos.mean_sharpe)],
    ['Compounded OOS return', fmtPct(oos.total_return_pct)],
    ['Drawdown cut vs SPY', fmtPct(oos.mean_drawdown_reduction_pct)],
    ['Total fills', fmtInt(oos.total_trades)],
  ];
  for (const [k, v] of rows) kv.append(h('dt', { text: k }), h('dd', { text: String(v ?? '—') }));
  body.append(h('h4', { text: 'Out-of-sample' }), kv);

  // folds
  body.append(h('h4', { text: 'Folds' }));
  if (d.folds && d.folds.length) {
    const chartEl = h('div', { class: 'ch-pane mini mb-8' });
    body.append(chartEl);
    const c = colors();
    const chart = createChart(chartEl, { height: 150 });
    const hist = chart.addHistogramSeries({ priceLineVisible: false, lastValueVisible: false });
    hist.setData(d.folds.map((f, i) => ({
      time: i + 1,
      value: f.excess_vs_spy_pct ?? 0,
      color: (f.excess_vs_spy_pct ?? 0) >= 0 ? c.up : c.down,
    })));
    chart.timeScale().fitContent();

    const tbody = h('tbody');
    for (const f of d.folds) {
      tbody.append(h('tr', {},
        h('td', { class: 'num', text: f.window }),
        h('td', { text: f.regime || '—' }),
        h('td', { class: 'num ' + upDownClass(f.excess_vs_spy_pct), text: fmtPct(f.excess_vs_spy_pct) }),
        h('td', { class: 'num', text: fmtNum(f.sharpe) }),
        h('td', { class: 'num', text: fmtPct(f.max_drawdown_pct, { signed: false }) }),
        h('td', { class: 'num', text: fmtInt(f.trades) })));
    }
    body.append(h('div', { class: 'c-table-wrap' }, h('table', { class: 'c-table' },
      h('thead', {}, h('tr', {},
        h('th', { text: 'Window' }), h('th', { text: 'Regime' }),
        h('th', { class: 'num', text: 'Excess' }), h('th', { class: 'num', text: 'Sharpe' }),
        h('th', { class: 'num', text: 'MaxDD' }), h('th', { class: 'num', text: 'Fills' }))),
      tbody)));
  } else {
    body.append(h('div', { class: 'c-empty', text: 'Fold detail not recorded for this run (pre-v5.40 record format).' }));
  }

  // holdout
  body.append(h('h4', { class: 'mt-16', text: 'Sealed holdout' }));
  if (d.holdout && d.holdout.passes != null) {
    const hk = h('dl', { class: 'c-kv' });
    for (const [k, v] of Object.entries(d.holdout)) {
      if (v == null || typeof v === 'object') continue;
      hk.append(h('dt', { text: k }), h('dd', { text: String(v) }));
    }
    body.append(hk);
  } else {
    body.append(h('div', { class: 't-dim', text:
      `SEALED — ${(d.config || {}).holdout_window || 'window pinned'} · burning the one look requires owner sign-off (--burn-holdout).` }));
  }

  // config disclosure
  body.append(h('h4', { class: 'mt-16', text: 'Config (full disclosure)' }));
  const ck = h('dl', { class: 'c-kv' });
  for (const [k, v] of Object.entries(d.config || {})) {
    ck.append(h('dt', { text: k }),
              h('dd', { text: typeof v === 'object' ? JSON.stringify(v) : String(v) }));
  }
  body.append(ck);
}

/* ── compare view ── */
async function openCompare() {
  const ids = [...runsState.compare];
  const drawer = document.getElementById('run-drawer');
  const body = document.getElementById('run-drawer-body');
  document.getElementById('run-drawer-title').textContent = `Compare ${ids.length} runs`;
  drawer.classList.add('open');
  skeleton(body);
  let runs;
  try {
    runs = await Promise.all(ids.map(id => apiGet(`/api/lab/runs/${id}`)));
  } catch (err) {
    renderError(body, err, openCompare);
    return;
  }
  clear(body);
  const metrics = [
    ['Excess vs SPY / fold', r => r.oos?.mean_excess_vs_spy_pct, fmtPct, true],
    ['Folds beating SPY', r => r.oos?.folds_beating_spy, v => v, false],
    ['Mean excess Sharpe', r => r.oos?.mean_excess_sharpe, fmtNum, true],
    ['Mean Sharpe', r => r.oos?.mean_sharpe, fmtNum, true],
    ['Drawdown cut', r => r.oos?.mean_drawdown_reduction_pct, fmtPct, true],
    ['Fills', r => r.oos?.total_trades, fmtInt, false],
    ['All criteria met', r => r.criteria?.all_met ? 'YES' : 'no', v => v, false],
    ['Holdout', r => r.holdout_status, v => v, false],
    ['Universe', r => r.universe, v => v, false],
  ];
  const thead = h('tr', {}, h('th', { text: 'Metric' }),
    ...runs.map(r => h('th', { class: 'num', text: `${r.strategy_name} #${r.id}` })));
  const tbody = h('tbody');
  for (const [label, get, fmt, numeric] of metrics) {
    const vals = runs.map(get);
    let bestIdx = -1;
    if (numeric) {
      const nums = vals.map(v => (typeof v === 'number' ? v : -Infinity));
      bestIdx = nums.indexOf(Math.max(...nums));
    }
    tbody.append(h('tr', {}, h('td', { text: label }),
      ...vals.map((v, i) => h('td', {
        class: 'num' + (i === bestIdx ? ' t-up' : ''),
        text: String(v == null ? '—' : fmt(v)),
      }))));
  }
  body.append(h('div', { class: 'c-table-wrap' },
    h('table', { class: 'c-table' }, h('thead', {}, thead), tbody)));
}

/* ════ backtest tab (rebuilt on net/poll) ════ */
let btPoller = null;

async function initBacktest() {
  const sel = document.getElementById('bt-strategy');
  try {
    const strategies = await apiGet('/api/strategies');
    sel.replaceChildren(...strategies.map(s => h('option', { value: s.name, text: s.name })));
  } catch { sel.replaceChildren(h('option', { text: 'coward' })); }

  const modeSel = document.getElementById('bt-mode');
  const sync = () => {
    document.getElementById('bt-symbols-wrap').classList.toggle('hidden', modeSel.value !== 'symbols');
    document.getElementById('bt-topn-wrap').classList.toggle('hidden', modeSel.value !== 'top');
  };
  modeSel.addEventListener('change', sync);
  sync();

  document.getElementById('bt-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    btPoller?.stop();
    const results = document.getElementById('bt-results');
    const body = {
      strategy: sel.value,
      mode: modeSel.value,
      symbols: document.getElementById('bt-symbols').value.split(',').map(s => s.trim()).filter(Boolean),
      top_n: parseInt(document.getElementById('bt-topn').value, 10) || 100,
      starting_cash: parseFloat(document.getElementById('bt-cash').value) || 10000,
    };
    const start = document.getElementById('bt-start').value;
    const end = document.getElementById('bt-end').value;
    if (start) body.start = start;
    if (end) body.end = end;
    if (start && end && start > end) {
      renderError(results, { message: 'start date is after end date' });
      return;
    }
    skeleton(results, 'chart');
    let jobId;
    try {
      jobId = (await apiPost('/api/backtest/jobs', body)).job_id;
    } catch (err) {
      renderError(results, err);
      return;
    }
    btPoller = poller(async () => {
      const job = await apiGet(`/api/backtest/jobs/${jobId}`);
      if (job.status === 'done') { renderBtResult(job.result); return false; }
      if (job.status === 'error') {
        renderError(results, { message: job.error || 'backtest failed' });
        return false;
      }
      const p = job.progress || {};
      clear(results).append(h('div', { class: 'c-empty' },
        h('div', { class: 'c-skel mb-8' }),
        h('div', { text: `${p.phase || job.status}… ${p.done ?? ''}${p.total ? ' / ' + p.total : ''}` })));
      return true;
    }, {
      intervalMs: 1500, maxFailures: 5,
      onGiveUp: () => renderError(results,
        { message: 'job lost — backtest jobs are in-memory and reset on restart' },
        () => document.getElementById('bt-form').requestSubmit()),
    });
  });
}

function renderBtResult(d) {
  const results = document.getElementById('bt-results');
  const s = d.stats || {};
  const stat = (label, value, cls = '') => h('div', { class: 'c-stat' },
    h('div', { class: 'label', text: label }),
    h('div', { class: `value num ${cls}`, text: value }));
  clear(results).append(
    h('div', { class: 't-dim mb-8', text: `${d.strategy} · ${d.universe_mode} · ${fmtDollar(d.starting_cash)} start` }),
    h('div', { class: 'grid-4 mb-16' },
      stat('Return', fmtPct(s.return_pct), upDownClass(s.return_pct)),
      stat('Sharpe', fmtNum(s.sharpe)),
      stat('Max drawdown', fmtPct(s.max_drawdown_pct, { signed: false }), 't-down'),
      stat('Win rate', s.win_rate != null ? fmtPct(s.win_rate, { signed: false }) : '—'),
      stat('vs ' + (s.benchmark_symbol || 'benchmark'), s.excess_return_pct != null ? fmtPct(s.excess_return_pct) : '—', upDownClass(s.excess_return_pct)),
      stat('Profit factor', fmtNum(s.profit_factor)),
      stat('Exposure', s.exposure_pct != null ? fmtPct(s.exposure_pct, { signed: false }) : '—'),
      stat('Closed trades', fmtInt(s.num_closed_trades)),
    ),
    h('div', { class: 'c-card mb-16' },
      h('div', { class: 'c-card-header', text: 'Equity curve' }),
      h('div', { class: 'c-card-body' }, h('div', { class: 'ch-pane equity', id: 'bt-chart' }))),
  );
  const c = colors();
  const chart = createChart(document.getElementById('bt-chart'));
  const area = chart.addAreaSeries({
    lineColor: c.accent, topColor: c.accent + '33', bottomColor: 'transparent', lineWidth: 2,
  });
  area.setData((d.equity_curve || []).map(p => ({
    time: toEpochSec(p.date), value: p.equity,
  })).filter(p => p.time != null));
  chart.timeScale().fitContent();
}

/* ════ boot ════ */
async function init() {
  document.getElementById('lab-tabs').addEventListener('click', (e) => {
    const b = e.target.closest('button');
    if (b) switchTab(b.dataset.tab);
  });
  document.getElementById('run-drawer-close').addEventListener('click', () =>
    document.getElementById('run-drawer').classList.remove('open'));
  document.getElementById('rf-compare').addEventListener('click', openCompare);
  for (const id of ['rf-label', 'rf-verdict', 'rf-holdout']) {
    document.getElementById(id).addEventListener('change', () => { runsState.offset = 0; loadRuns(); });
  }
  let t = null;
  document.getElementById('rf-strategy').addEventListener('input', () => {
    clearTimeout(t);
    t = setTimeout(() => { runsState.offset = 0; loadRuns(); }, 300);
  });

  try {
    const labels = await apiGet('/api/lab/labels');
    const sel = document.getElementById('rf-label');
    for (const p of labels.prefixes) sel.append(h('option', { value: p, text: p }));
  } catch { /* dropdown stays generic */ }

  initBacktest();
  switchTab(activeTab());
}

document.addEventListener('DOMContentLoaded', init);

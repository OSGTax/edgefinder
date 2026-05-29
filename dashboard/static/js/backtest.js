/* Backtest page — run a strategy over historical daily bars and chart the result. */
(function () {
  'use strict';

  const form = document.getElementById('bt-form');
  const results = document.getElementById('bt-results');
  let chart = null, series = null, resizeObs = null;

  async function loadStrategies() {
    try {
      const list = await api('/api/strategies');
      const sel = document.getElementById('bt-strategy');
      sel.innerHTML = (list || []).map(s => `<option value="${s.name}">${s.name}</option>`).join('');
    } catch (e) {
      console.error('Failed to load strategies:', e);
    }
  }

  function fmtPct(v) { return (v >= 0 ? '+' : '') + Number(v).toFixed(2) + '%'; }
  function fmtUsd(v) {
    return '$' + Number(v).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }

  // Show/hide the symbols vs top-N inputs based on the universe mode.
  const modeSel = document.getElementById('bt-mode');
  function syncMode() {
    const m = modeSel.value;
    document.getElementById('bt-symbols-wrap').style.display = (m === 'symbols') ? 'flex' : 'none';
    document.getElementById('bt-topn-wrap').style.display = (m === 'top') ? 'flex' : 'none';
  }
  modeSel.addEventListener('change', syncMode);
  syncMode();

  let pollTimer = null;

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (pollTimer) { clearTimeout(pollTimer); pollTimer = null; }

    const mode = modeSel.value;
    const body = {
      strategy: document.getElementById('bt-strategy').value,
      mode,
      symbols: document.getElementById('bt-symbols').value.split(',').map(s => s.trim()).filter(Boolean),
      top_n: parseInt(document.getElementById('bt-topn').value, 10) || 100,
      starting_cash: parseFloat(document.getElementById('bt-cash').value) || 10000,
    };
    const start = document.getElementById('bt-start').value;
    const end = document.getElementById('bt-end').value;
    if (start) body.start = start;
    if (end) body.end = end;

    renderProgress({ status: 'queued', progress: {} });
    try {
      const r = await fetch('/api/backtest/jobs', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!r.ok) {
        const err = await r.json().catch(() => ({}));
        results.innerHTML = `<div class="empty-state"><div class="icon">&#9888;</div>${err.detail || ('HTTP ' + r.status)}</div>`;
        return;
      }
      pollJob((await r.json()).job_id);
    } catch (e) {
      console.error(e);
      results.innerHTML = '<div class="empty-state">Backtest request failed.</div>';
    }
  });

  async function pollJob(jobId) {
    try {
      const job = await api('/api/backtest/jobs/' + jobId);
      if (job.status === 'done') {
        renderResult(job.result);
        return;
      }
      if (job.status === 'error') {
        results.innerHTML = `<div class="empty-state"><div class="icon">&#9888;</div>Backtest failed: ${job.error || 'unknown error'}</div>`;
        return;
      }
      renderProgress(job);
      pollTimer = setTimeout(() => pollJob(jobId), 1200);
    } catch (e) {
      console.error(e);
      results.innerHTML = '<div class="empty-state">Lost contact with the backtest job.</div>';
    }
  }

  function renderProgress(job) {
    const p = job.progress || {};
    const phaseLabels = { loading: 'Loading bars', prepare: 'Preparing indicators', simulate: 'Simulating days' };
    let label = phaseLabels[p.phase] || (job.status === 'queued' ? 'Queued…' : 'Starting…');
    let pct = 0, detail = '';
    if (p.total) {
      pct = Math.min(100, Math.round((p.done / p.total) * 100));
      detail = `${p.done.toLocaleString()} / ${p.total.toLocaleString()}`;
    }
    const sym = job.num_symbols ? `${job.num_symbols.toLocaleString()} symbols · ` : '';
    results.innerHTML = `
      <div class="card"><div class="card-body">
        <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
          <span class="stat-label">${label}</span>
          <span class="text-secondary" style="font-size:12px;">${sym}${detail}</span>
        </div>
        <div style="background:#1a2332;border-radius:6px;height:10px;overflow:hidden;">
          <div style="height:100%;width:${pct}%;background:#00d4a1;transition:width .3s;"></div>
        </div>
        <div class="text-secondary" style="font-size:12px;margin-top:8px;">
          Replaying the live engine over daily bars — full-universe runs take a few minutes.
        </div>
      </div></div>`;
  }

  function renderResult(d) {
    if (pollTimer) { clearTimeout(pollTimer); pollTimer = null; }
    const s = d.stats || {};
    const retCls = (s.return_pct >= 0) ? 'text-positive' : 'text-negative';
    const num = (v, suf = '') => (v == null ? '&mdash;' : v + suf);

    const nSym = d.num_symbols || (d.symbols || []).length;
    const uni = d.universe_mode === 'full' ? 'full universe'
      : d.universe_mode === 'top' ? `top ${nSym} liquid` : `${nSym} symbol${nSym === 1 ? '' : 's'}`;
    const benchPeriod = s.benchmark_period ? ` · benchmark ${s.benchmark_period}` : '';

    // Benchmark sub-line on the Return card
    let benchSub = `${fmtUsd(d.starting_cash)} &rarr; ${fmtUsd(d.final_equity)}`;
    if (s.benchmark_return_pct != null) {
      const exCls = (s.excess_return_pct >= 0) ? 'text-positive' : 'text-negative';
      benchSub = `vs ${s.benchmark_symbol} ${fmtPct(s.benchmark_return_pct)} · `
        + `<span class="${exCls}">${fmtPct(s.excess_return_pct)} excess</span>`;
    }

    const cards = [
      { label: 'Return', value: `<span class="${retCls}">${fmtPct(s.return_pct)}</span>`, sub: benchSub },
      { label: 'CAGR', value: num(s.cagr_pct, '%') },
      { label: 'Sharpe', value: num(s.sharpe) },
      { label: 'Max drawdown', value: `<span class="text-negative">-${num(s.max_drawdown_pct, '%')}</span>`, sub: `${s.days} trading days` },
      { label: 'Win rate', value: num(s.win_rate, '%'), sub: `${s.num_closed_trades} closed · ${s.num_open_positions} open` },
      { label: 'Profit factor', value: num(s.profit_factor) },
      { label: 'Avg win / loss', value: `${s.avg_win == null ? '—' : fmtUsd(s.avg_win)} / ${s.avg_loss == null ? '—' : fmtUsd(s.avg_loss)}` },
      { label: 'Exposure', value: num(s.exposure_pct, '%'), sub: 'days holding ≥1 position' },
    ];

    const shown = (d.trades || []).length;
    const total = d.trades_total || shown;
    const tradesHdr = total > shown
      ? `Closed Trades — top ${shown} of ${total.toLocaleString()} by P&amp;L`
      : `Closed Trades (${shown})`;

    results.innerHTML = `
      <div class="text-secondary" style="font-size:12px;margin-bottom:12px;">
        ${d.strategy} · ${uni}${benchPeriod}
      </div>
      <div class="stats-grid grid-4 mb-20">
        ${cards.map(c => `<div class="stat-card">
          <div class="stat-label">${c.label}</div>
          <div class="stat-value">${c.value}</div>
          ${c.sub ? `<div class="stat-sub text-secondary">${c.sub}</div>` : ''}
        </div>`).join('')}
      </div>
      <div class="card mb-20">
        <div class="card-header">Equity Curve</div>
        <div class="card-body"><div id="bt-chart" style="width:100%;height:300px;"></div></div>
      </div>
      <div class="card mb-20">
        <div class="card-header">${tradesHdr}</div>
        <div class="card-body" id="bt-trades"></div>
      </div>`;

    buildChart(d.equity_curve || []);
    buildTrades(d.trades || []);
  }

  function buildChart(curve) {
    const cont = document.getElementById('bt-chart');
    if (!cont) return;
    if (resizeObs) { resizeObs.disconnect(); resizeObs = null; }
    if (chart) { try { chart.remove(); } catch (e) {} chart = null; }

    chart = LightweightCharts.createChart(cont, {
      width: cont.clientWidth, height: 300,
      layout: { background: { color: '#111a25' }, textColor: '#6b8aab' },
      grid: { vertLines: { color: '#1a2332' }, horzLines: { color: '#1a2332' } },
      timeScale: { borderColor: '#1a2332' },
      rightPriceScale: { borderColor: '#1a2332' },
    });
    series = chart.addAreaSeries({
      lineColor: '#00d4a1', topColor: 'rgba(0,212,161,0.3)',
      bottomColor: 'rgba(0,212,161,0.02)', lineWidth: 2,
    });
    series.setData(curve.map(p => ({ time: p.date, value: p.equity })));
    chart.timeScale().fitContent();
    resizeObs = new ResizeObserver(() => { if (chart) chart.applyOptions({ width: cont.clientWidth }); });
    resizeObs.observe(cont);
  }

  function buildTrades(trades) {
    const tb = document.getElementById('bt-trades');
    if (!tb) return;
    if (!trades.length) {
      tb.innerHTML = '<div class="empty-state">No closed trades in this run.</div>';
      return;
    }
    const rows = trades.map(t => {
      const cls = (t.pnl_dollars >= 0) ? 'text-positive' : 'text-negative';
      const pnl = (t.pnl_dollars == null) ? '&mdash;' : fmtUsd(t.pnl_dollars);
      return `<tr>
        <td>${t.symbol}</td>
        <td>${(t.entry_time || '').slice(0, 10)} @ ${t.entry_price}</td>
        <td>${(t.exit_time || '').slice(0, 10)} @ ${t.exit_price == null ? '&mdash;' : t.exit_price}</td>
        <td>${t.shares}</td>
        <td class="${cls}">${pnl}</td>
        <td class="text-secondary">${(t.exit_reason || '').slice(0, 48)}</td>
      </tr>`;
    }).join('');
    tb.innerHTML = `<table class="data-table">
      <thead><tr><th>Symbol</th><th>Entry</th><th>Exit</th><th>Shares</th><th>P&amp;L</th><th>Reason</th></tr></thead>
      <tbody>${rows}</tbody></table>`;
  }

  loadStrategies();
})();

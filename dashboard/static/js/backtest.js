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

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const body = {
      strategy: document.getElementById('bt-strategy').value,
      symbols: document.getElementById('bt-symbols').value.split(',').map(s => s.trim()).filter(Boolean),
      starting_cash: parseFloat(document.getElementById('bt-cash').value) || 10000,
    };
    const start = document.getElementById('bt-start').value;
    const end = document.getElementById('bt-end').value;
    if (start) body.start = start;
    if (end) body.end = end;

    results.innerHTML = '<div class="empty-state">Running backtest… (replaying the live engine over daily bars)</div>';
    try {
      const r = await fetch('/api/backtest', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!r.ok) {
        const err = await r.json().catch(() => ({}));
        results.innerHTML = `<div class="empty-state"><div class="icon">&#9888;</div>${err.detail || ('HTTP ' + r.status)}</div>`;
        return;
      }
      renderResult(await r.json());
    } catch (e) {
      console.error(e);
      results.innerHTML = '<div class="empty-state">Backtest request failed.</div>';
    }
  });

  function renderResult(d) {
    const s = d.stats || {};
    const retCls = (s.return_pct >= 0) ? 'text-positive' : 'text-negative';
    const winRate = (s.win_rate == null) ? '&mdash;' : s.win_rate + '%';

    results.innerHTML = `
      <div class="stats-grid grid-4 mb-20">
        <div class="stat-card">
          <div class="stat-label">Return</div>
          <div class="stat-value ${retCls}">${fmtPct(s.return_pct)}</div>
          <div class="stat-sub text-secondary">${fmtUsd(d.starting_cash)} &rarr; ${fmtUsd(d.final_equity)}</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Closed trades</div>
          <div class="stat-value">${s.num_closed_trades}</div>
          <div class="stat-sub text-secondary">${s.num_open_positions} open at end</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Win rate</div>
          <div class="stat-value">${winRate}</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Max drawdown</div>
          <div class="stat-value text-negative">-${s.max_drawdown_pct}%</div>
          <div class="stat-sub text-secondary">${s.days} trading days</div>
        </div>
      </div>
      <div class="card mb-20">
        <div class="card-header">Equity Curve</div>
        <div class="card-body"><div id="bt-chart" style="width:100%;height:300px;"></div></div>
      </div>
      <div class="card mb-20">
        <div class="card-header">Closed Trades (${(d.trades || []).length})</div>
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

/* ═══════════════════════════════════════════════════════════════
   EdgeFinder — Strategies Page
   ═══════════════════════════════════════════════════════════════ */

// ── State ────────────────────────────────────────────────────

let _accounts = [];       // raw accounts array from API
let _statsCache = {};     // strategy_name -> stats object
let _tradesCache = {};    // strategy_name -> trades array
let _selectedStrategy = null;
let _detailChart = null;
let _detailSeries = null;
let _detailDays = 30;

// Per-strategy risk config (mirrors config/settings.py)
const STRATEGY_RISK = {
  coward:     { risk_pct: 5,  target_pct: 15, stop_pct: 20 },
  gambler:    { risk_pct: 10, target_pct: 25, stop_pct: 20 },
  degenerate: { risk_pct: 20, target_pct: 50, stop_pct: 20 },
};

// ── Streak Computation ────────────────────────────────────────

function computeStreak(trades) {
  const closed = trades
    .filter(t => t.status === 'CLOSED' && t.pnl_dollars != null)
    .sort((a, b) => new Date(b.exit_time) - new Date(a.exit_time));
  if (closed.length === 0) return { type: 'none', count: 0 };
  const firstIsWin = closed[0].pnl_dollars > 0;
  let count = 0;
  for (const t of closed) {
    if ((t.pnl_dollars > 0) === firstIsWin) count++;
    else break;
  }
  if (count < 2) return { type: 'none', count: 0 };
  return { type: firstIsWin ? 'win' : 'loss', count };
}

// ── Strategy Cards ───────────────────────────────────────────

async function loadStrategyCards() {
  const grid = document.getElementById('strategy-cards');

  try {
    // Fetch accounts + stats for all known strategies in parallel
    const [accounts] = await Promise.all([
      api('/api/strategies/accounts'),
    ]);

    _accounts = Array.isArray(accounts) ? accounts : [];

    if (_accounts.length === 0) {
      grid.innerHTML = `
        <div class="empty-state" style="grid-column:1/-1;">
          <div class="icon">&#9670;</div>
          No strategy accounts found. Run the scanner to initialise strategies.
        </div>`;
      return;
    }

    // Fetch stats and trades per strategy concurrently
    await Promise.all(_accounts.map(async (acct) => {
      const name = acct.strategy_name;
      try {
        const stats = await api('/api/trades/stats?strategy=' + encodeURIComponent(name));
        _statsCache[name] = stats;
      } catch (_) {
        _statsCache[name] = null;
      }
      try {
        const trades = await api('/api/trades?strategy=' + encodeURIComponent(name) + '&limit=50');
        _tradesCache[name] = Array.isArray(trades) ? trades : [];
      } catch (_) {
        _tradesCache[name] = [];
      }
    }));

    grid.innerHTML = _accounts.map(acct => renderStrategyCard(acct)).join('');

    // Re-highlight selected card if any
    if (_selectedStrategy) {
      highlightCard(_selectedStrategy);
    }

  } catch (e) {
    console.error('Failed to load strategy accounts:', e);
    grid.innerHTML = `
      <div class="empty-state" style="grid-column:1/-1;">
        <div class="icon">&#9888;</div>
        Failed to load strategy data.
      </div>`;
  }
}

function renderStrategyCard(acct) {
  const name = acct.strategy_name;
  const color = stratColor(name);
  const risk = STRATEGY_RISK[name] || { risk_pct: '—', target_pct: '—', stop_pct: '—' };
  const stats = _statsCache[name];

  const equity = acct.total_equity || acct.cash || 0;
  const cash = acct.cash || 0;
  const startingCapital = acct.starting_capital || 5000;
  const drawdownPct = acct.drawdown_pct || 0;
  const drawdownDisplay = (drawdownPct * 100).toFixed(1);
  const gClass = gaugeClass(drawdownPct);
  // Cap gauge fill at 100%
  const gaugeFill = Math.min(100, drawdownPct * 100);

  const realizedPnl = acct.realized_pnl || 0;
  const unrealizedPnl = acct.unrealized_pnl || 0;
  const totalPnl = realizedPnl + unrealizedPnl;
  const isPaused = acct.is_paused;

  // Stats
  const winRate = stats && stats.total_trades > 0
    ? (stats.win_rate * 100).toFixed(1) + '%'
    : '—';
  const avgR = stats && stats.total_trades > 0
    ? fmtNum(stats.avg_r_multiple, 2) + 'R'
    : '—';
  const tradeCount = stats ? stats.total_trades : 0;

  const pausedBadge = isPaused
    ? `<span class="pill pill-warning" style="margin-left:8px;">PAUSED</span>`
    : '';

  const streak = computeStreak(_tradesCache[name] || []);
  const streakHtml = streak.type === 'win'
    ? `<div style="font-size:12px;margin-top:6px;" class="text-positive">🔥 ${streak.count}W streak</div>`
    : streak.type === 'loss'
    ? `<div style="font-size:12px;margin-top:6px;" class="text-negative">❄️ ${streak.count}L streak</div>`
    : '';

  return `
    <div class="card strategy-card" data-strategy="${name}"
         style="cursor:pointer;transition:border-color 0.2s;"
         onclick="selectStrategy('${name}')">
      <div class="card-header" style="border-bottom-color:${color}20;">
        <span class="strat-dot strat-dot-${name}"></span>
        <span style="color:${color};text-transform:capitalize;">${name}</span>
        ${pausedBadge}
      </div>
      <div class="card-body">

        <!-- Equity + Cash -->
        <div style="display:flex;justify-content:space-between;align-items:flex-end;margin-bottom:14px;">
          <div>
            <div class="stat-label">Total Equity</div>
            <div style="font-size:22px;font-weight:800;color:var(--text-primary);line-height:1.1;">
              ${fmtDollar(equity)}
            </div>
            <div class="stat-sub ${pnlClass(totalPnl)}" style="margin-top:2px;">
              ${totalPnl >= 0 ? '+' : ''}${fmtDollar(totalPnl)} total P&L
            </div>
          </div>
          <div style="text-align:right;">
            <div class="stat-label">Cash</div>
            <div style="font-size:16px;font-weight:700;color:var(--text-secondary);">
              ${fmtDollar(cash)}
            </div>
          </div>
        </div>

        <!-- Drawdown gauge -->
        <div style="margin-bottom:14px;">
          <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
            <span class="stat-label">Drawdown</span>
            <span style="font-size:11px;font-weight:600;color:var(--${gClass === 'good' ? 'positive' : gClass === 'warn' ? 'warning' : 'negative'});">
              ${drawdownDisplay}%
            </span>
          </div>
          <div class="gauge">
            <div class="gauge-fill ${gClass}" style="width:${gaugeFill}%;"></div>
          </div>
        </div>

        <!-- Stats row -->
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:14px;">
          <div style="text-align:center;">
            <div class="stat-label">Win Rate</div>
            <div style="font-size:15px;font-weight:700;color:var(--text-primary);">${winRate}</div>
          </div>
          <div style="text-align:center;">
            <div class="stat-label">Avg R</div>
            <div style="font-size:15px;font-weight:700;color:var(--text-primary);">${avgR}</div>
          </div>
          <div style="text-align:center;">
            <div class="stat-label">Trades</div>
            <div style="font-size:15px;font-weight:700;color:var(--text-primary);">${tradeCount}</div>
          </div>
        </div>
        ${streakHtml}

        <!-- Risk parameters -->
        <div style="background:var(--bg);border-radius:var(--radius-sm);padding:8px 10px;margin-bottom:14px;">
          <div style="display:flex;justify-content:space-between;font-size:11px;color:var(--text-muted);gap:8px;">
            <span>Risk <strong style="color:var(--text-secondary);">${risk.risk_pct}%</strong></span>
            <span>Target <strong style="color:var(--positive);">${risk.target_pct}%</strong></span>
            <span>Stop <strong style="color:var(--negative);">${risk.stop_pct}%</strong></span>
          </div>
        </div>

        <!-- View Details button -->
        <button
          style="width:100%;background:var(--surface-elevated);border:1px solid var(--border);
                 color:var(--text-secondary);padding:7px;border-radius:var(--radius-sm);
                 cursor:pointer;font-size:12px;font-weight:600;transition:all 0.15s;"
          onmouseover="this.style.borderColor='${color}';this.style.color='${color}';"
          onmouseout="this.style.borderColor='var(--border)';this.style.color='var(--text-secondary)';"
          onclick="event.stopPropagation();selectStrategy('${name}')">
          View Details &#9656;
        </button>

      </div>
    </div>`;
}

// ── Strategy Selection ────────────────────────────────────────

function selectStrategy(name) {
  _selectedStrategy = name;
  highlightCard(name);
  showDetail(name);
}

function highlightCard(name) {
  document.querySelectorAll('.strategy-card').forEach(card => {
    const isSelected = card.dataset.strategy === name;
    card.style.borderColor = isSelected ? stratColor(name) : '';
  });
}

function closeDetail() {
  _selectedStrategy = null;
  document.getElementById('strategy-detail').style.display = 'none';
  document.querySelectorAll('.strategy-card').forEach(card => {
    card.style.borderColor = '';
  });
}

// ── Detail Panel ─────────────────────────────────────────────

async function showDetail(name) {
  const panel = document.getElementById('strategy-detail');
  const color = stratColor(name);

  // Update header
  const dot = document.getElementById('detail-dot');
  dot.className = `strat-dot strat-dot-${name}`;

  document.getElementById('detail-name').textContent = name;

  const acct = _accounts.find(a => a.strategy_name === name);
  const pausedBadge = document.getElementById('detail-paused-badge');
  pausedBadge.innerHTML = (acct && acct.is_paused)
    ? `<span class="pill pill-warning">PAUSED</span>`
    : '';

  panel.style.display = 'block';
  panel.scrollIntoView({ behavior: 'smooth', block: 'start' });

  // Render account body
  renderDetailAccount(name, acct);

  // Load equity curve and trades in parallel
  await Promise.all([
    loadDetailEquityCurve(name, _detailDays),
    loadDetailTrades(name),
  ]);

  // Wire up time-range tabs
  initDetailTabs(name);
}

function renderDetailAccount(name, acct) {
  const body = document.getElementById('detail-account-body');
  if (!acct) {
    body.innerHTML = '<div class="empty-state">No account data.</div>';
    return;
  }

  const color = stratColor(name);
  const equity = acct.total_equity || acct.cash || 0;
  const cash = acct.cash || 0;
  const startingCapital = acct.starting_capital || 5000;
  const positionsValue = acct.open_positions_value || 0;
  const realizedPnl = acct.realized_pnl || 0;
  const unrealizedPnl = acct.unrealized_pnl || 0;
  const peakEquity = acct.peak_equity || startingCapital;
  const drawdownPct = acct.drawdown_pct || 0;

  // Risk budget: deployed vs available
  // Deployed = positions value (already market-valued in API response)
  const deployedPct = equity > 0 ? Math.min(100, (positionsValue / equity) * 100) : 0;
  const cashPct = 100 - deployedPct;

  const posCount = acct.position_count || 0;

  body.innerHTML = `
    <div class="data-row">
      <span class="label">Starting Capital</span>
      <span class="value">${fmtDollar(startingCapital)}</span>
    </div>
    <div class="data-row">
      <span class="label">Total Equity</span>
      <span class="value">${fmtDollar(equity)}</span>
    </div>
    <div class="data-row">
      <span class="label">Cash Available</span>
      <span class="value">${fmtDollar(cash)}</span>
    </div>
    <div class="data-row">
      <span class="label">Open Positions Value</span>
      <span class="value">${fmtDollar(positionsValue)}</span>
    </div>
    <div class="data-row">
      <span class="label">Peak Equity</span>
      <span class="value">${fmtDollar(peakEquity)}</span>
    </div>
    <div class="data-row">
      <span class="label">Realized P&amp;L</span>
      <span class="value ${pnlClass(realizedPnl)}">${fmtPnl(realizedPnl)}</span>
    </div>
    <div class="data-row">
      <span class="label">Unrealized P&amp;L</span>
      <span class="value ${pnlClass(unrealizedPnl)}">${fmtPnl(unrealizedPnl)}</span>
    </div>
    <div class="data-row">
      <span class="label">Open Positions</span>
      <span class="value">${posCount} / 5</span>
    </div>

    <!-- Risk Budget Bar -->
    <div style="margin-top:14px;">
      <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
        <span class="stat-label">Capital Deployment</span>
        <span style="font-size:11px;color:var(--text-secondary);">
          <span style="color:${color};">${deployedPct.toFixed(1)}% deployed</span>
          &nbsp;/&nbsp;
          <span style="color:var(--text-muted);">${cashPct.toFixed(1)}% cash</span>
        </span>
      </div>
      <div style="height:10px;border-radius:5px;background:var(--border);overflow:hidden;">
        <div style="height:100%;width:${deployedPct}%;background:${color};opacity:0.8;border-radius:5px;transition:width 0.3s;"></div>
      </div>
    </div>`;
}

// ── Detail Equity Curve ──────────────────────────────────────

async function loadDetailEquityCurve(strategyName, days) {
  const container = document.getElementById('detail-equity-chart');
  const color = stratColor(strategyName);

  try {
    const data = await api('/api/strategies/equity-curve?days=' + days);
    const series = (data && data[strategyName]) ? data[strategyName] : [];

    // Build chart data — one point per date
    let equityData = series
      .filter(pt => pt.date && pt.total_equity != null)
      .map(pt => ({ time: pt.date, value: pt.total_equity }));

    // Deduplicate by date (keep last)
    const byDate = {};
    for (const pt of equityData) byDate[pt.time] = pt.value;
    equityData = Object.keys(byDate).sort().map(d => ({ time: d, value: byDate[d] }));

    if (equityData.length === 0) {
      // Flat starting-capital line
      const today = new Date().toISOString().slice(0, 10);
      const startingCapital = (_accounts.find(a => a.strategy_name === strategyName) || {}).starting_capital || 5000;
      equityData = [{ time: today, value: startingCapital }];
    }

    if (!_detailChart) {
      _detailChart = LightweightCharts.createChart(container, {
        width: container.clientWidth,
        height: 250,
        layout: { background: { color: '#111a25' }, textColor: '#6b8aab' },
        grid: {
          vertLines: { color: '#1a2332' },
          horzLines: { color: '#1a2332' },
        },
        timeScale: { borderColor: '#1a2332' },
        rightPriceScale: { borderColor: '#1a2332' },
        crosshair: {
          horzLine: { color: '#2a3a4a', labelBackgroundColor: '#162030' },
          vertLine: { color: '#2a3a4a', labelBackgroundColor: '#162030' },
        },
      });

      _detailSeries = _detailChart.addAreaSeries({
        lineColor: color,
        topColor: color + '4D',
        bottomColor: color + '05',
        lineWidth: 2,
      });

      new ResizeObserver(() => {
        if (_detailChart) _detailChart.applyOptions({ width: container.clientWidth });
      }).observe(container);
    } else {
      // Update series colour for the newly selected strategy
      _detailSeries.applyOptions({
        lineColor: color,
        topColor: color + '4D',
        bottomColor: color + '05',
      });
    }

    _detailSeries.setData(equityData);
    _detailChart.timeScale().fitContent();

  } catch (e) {
    console.error('Failed to load detail equity curve:', e);
    container.innerHTML = '<div class="empty-state">Chart unavailable.</div>';
  }
}

function initDetailTabs(strategyName) {
  const tabs = document.getElementById('detail-eq-tabs');
  if (!tabs) return;

  // Remove old listeners by cloning (clean approach)
  const fresh = tabs.cloneNode(true);
  tabs.parentNode.replaceChild(fresh, tabs);

  fresh.addEventListener('click', async (e) => {
    const tab = e.target.closest('.filter-tab');
    if (!tab) return;
    fresh.querySelectorAll('.filter-tab').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    _detailDays = parseInt(tab.dataset.days, 10);
    await loadDetailEquityCurve(strategyName, _detailDays);
  });
}

// ── Detail Trades Table ──────────────────────────────────────

async function loadDetailTrades(strategyName) {
  const tbody = document.getElementById('detail-trades-body');
  const countEl = document.getElementById('detail-trade-count');

  tbody.innerHTML = '<tr><td colspan="9" class="empty-state" style="padding:20px;">Loading&hellip;</td></tr>';

  try {
    const trades = await api('/api/trades?strategy=' + encodeURIComponent(strategyName) + '&limit=50');

    if (!trades || trades.length === 0) {
      tbody.innerHTML = '<tr><td colspan="9" class="empty-state" style="padding:20px;">No trades yet.</td></tr>';
      if (countEl) countEl.textContent = '';
      return;
    }

    if (countEl) countEl.textContent = trades.length + ' trade' + (trades.length !== 1 ? 's' : '');

    tbody.innerHTML = trades.map(t => {
      const isOpen = t.status === 'OPEN';
      const pnl = t.pnl_dollars;
      const pnlCell = pnl != null
        ? `<span class="${pnlClass(pnl)}">${fmtPnl(pnl)}</span>`
        : '<span class="text-muted">—</span>';

      const rCell = t.r_multiple != null
        ? `<span class="${pnlClass(t.r_multiple)}">${fmtNum(t.r_multiple, 2)}R</span>`
        : '<span class="text-muted">—</span>';

      const statusPill = isOpen
        ? `<span class="pill pill-accent">OPEN</span>`
        : t.status === 'CLOSED'
          ? (pnl != null && pnl > 0
            ? `<span class="pill pill-positive">WIN</span>`
            : `<span class="pill pill-negative">LOSS</span>`)
          : `<span class="pill pill-muted">${t.status}</span>`;

      const dirPill = t.direction === 'LONG'
        ? `<span class="pill pill-positive">L</span>`
        : `<span class="pill pill-negative">S</span>`;

      const reason = t.exit_reason || '—';
      const reasonTrunc = reason.length > 20 ? reason.slice(0, 20) + '…' : reason;

      return `<tr>
        <td style="font-weight:700;color:var(--text-primary);">${t.symbol}</td>
        <td>${dirPill}</td>
        <td>${t.entry_price != null ? fmtDollar(t.entry_price) : '—'}</td>
        <td>${t.exit_price != null ? fmtDollar(t.exit_price) : (isOpen ? '<span class="text-muted">open</span>' : '—')}</td>
        <td>${pnlCell}</td>
        <td>${rCell}</td>
        <td>${statusPill}</td>
        <td style="color:var(--text-secondary);">${fmtTime(t.entry_time)}</td>
        <td style="color:var(--text-muted);" title="${reason}">${reasonTrunc}</td>
      </tr>`;
    }).join('');

  } catch (e) {
    console.error('Failed to load trades for strategy:', strategyName, e);
    tbody.innerHTML = '<tr><td colspan="9" class="empty-state" style="padding:20px;">Failed to load trades.</td></tr>';
  }
}

// ── Init ────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  loadStrategyCards();
});

/* ═══════════════════════════════════════════════════════════════
   EdgeFinder — Dashboard Page
   ═══════════════════════════════════════════════════════════════ */

const STARTING_CAPITAL = 15000; // 3 strategies x $5,000
let equityChart = null;
let equitySeries = null;

// ── Hero Stats ──────────────────────────────────────────────

async function loadStats() {
  try {
    const [accounts, stats] = await Promise.all([
      api('/api/strategies/accounts'),
      api('/api/trades/stats'),
    ]);

    // Total equity = sum of each strategy's total_equity
    let totalEquity = 0;
    let totalUnrealized = 0;
    let totalPositions = 0;

    if (Array.isArray(accounts)) {
      for (const a of accounts) {
        totalEquity += (a.total_equity || a.cash || 0);
        totalUnrealized += (a.unrealized_pnl || 0);
        totalPositions += (a.position_count || a.open_positions || 0);
      }
    } else if (accounts && typeof accounts === 'object') {
      for (const [, a] of Object.entries(accounts)) {
        const acct = Array.isArray(a) ? a[0] : a;
        if (!acct) continue;
        totalEquity += (acct.total_equity || acct.cash || 0);
        totalUnrealized += (acct.unrealized_pnl || 0);
        totalPositions += (acct.position_count || acct.open_positions || 0);
      }
    }

    // If no accounts returned, show starting capital
    if (totalEquity === 0) totalEquity = STARTING_CAPITAL;

    const equityChg = (totalEquity - STARTING_CAPITAL) / STARTING_CAPITAL;

    document.getElementById('stat-equity').textContent = fmtDollar(totalEquity);
    const chgEl = document.getElementById('stat-equity-chg');
    chgEl.textContent = fmtPct(equityChg) + ' from start';
    chgEl.className = 'stat-sub ' + pnlClass(equityChg);

    document.getElementById('stat-pnl').textContent = fmtPnl(totalUnrealized);
    document.getElementById('stat-pnl').className = 'stat-value ' + pnlClass(totalUnrealized);
    const pnlSub = document.getElementById('stat-pnl-sub');
    pnlSub.textContent = 'unrealized';
    pnlSub.className = 'stat-sub text-secondary';

    document.getElementById('stat-positions').textContent = totalPositions;
    document.getElementById('stat-positions-sub').textContent =
      totalPositions === 1 ? '1 position open' : totalPositions + ' positions open';

    // Win rate from trade stats
    let wins = 0, total = 0;
    if (stats) {
      // stats could be an object or have strategy-level breakdown
      if (stats.total_closed != null) {
        wins = stats.wins || 0;
        total = stats.total_closed || 0;
      } else if (stats.total_trades != null) {
        wins = stats.winning_trades || stats.wins || 0;
        total = stats.total_trades || 0;
      } else {
        // might be strategy-keyed
        for (const [, s] of Object.entries(stats)) {
          wins += (s.wins || s.winning_trades || 0);
          total += (s.total_closed || s.total_trades || 0);
        }
      }
    }

    const winRate = total > 0 ? (wins / total) : 0;
    document.getElementById('stat-winrate').textContent =
      total > 0 ? (winRate * 100).toFixed(1) + '%' : '--';
    document.getElementById('stat-winrate-sub').textContent =
      total > 0 ? wins + 'W / ' + (total - wins) + 'L of ' + total + ' closed' : 'no closed trades';

  } catch (e) {
    console.error('Failed to load stats:', e);
  }
}

// ── Equity Curve ────────────────────────────────────────────

async function loadEquityCurve(days = 90) {
  try {
    const data = await api('/api/strategies/equity-curve?days=' + days);
    if (!data) return;

    // Aggregate all strategies per date
    const dateMap = {};
    for (const [, series] of Object.entries(data)) {
      if (!Array.isArray(series)) continue;
      for (const pt of series) {
        const d = pt.date;
        if (!dateMap[d]) dateMap[d] = 0;
        dateMap[d] += (pt.total_equity || pt.equity || 0);
      }
    }

    const equityData = Object.keys(dateMap)
      .sort()
      .map(d => ({ time: d, value: dateMap[d] }));

    if (equityData.length === 0) {
      // Show flat line at starting capital
      const today = new Date().toISOString().slice(0, 10);
      equityData.push({ time: today, value: STARTING_CAPITAL });
    }

    const container = document.getElementById('equity-chart');

    if (!equityChart) {
      equityChart = LightweightCharts.createChart(container, {
        width: container.clientWidth,
        height: 300,
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

      equitySeries = equityChart.addAreaSeries({
        lineColor: '#00d4a1',
        topColor: 'rgba(0,212,161,0.3)',
        bottomColor: 'rgba(0,212,161,0.02)',
        lineWidth: 2,
      });

      // Responsive resize
      new ResizeObserver(() => {
        equityChart.applyOptions({ width: container.clientWidth });
      }).observe(container);
    }

    equitySeries.setData(equityData);
    equityChart.timeScale().fitContent();

  } catch (e) {
    console.error('Failed to load equity curve:', e);
  }
}

function initEquityTabs() {
  const tabs = document.getElementById('eq-range-tabs');
  if (!tabs) return;
  tabs.addEventListener('click', (e) => {
    const tab = e.target.closest('.filter-tab');
    if (!tab) return;
    tabs.querySelectorAll('.filter-tab').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    const days = parseInt(tab.dataset.days, 10);
    loadEquityCurve(days);
  });
}

// ── Open Positions ──────────────────────────────────────────

async function loadPositions() {
  const grid = document.getElementById('positions-grid');
  const countHeader = document.getElementById('positions-count-header');

  try {
    const [posData, accounts] = await Promise.all([
      api('/api/strategies/positions').catch(() => null),
      api('/api/strategies/accounts').catch(() => null),
    ]);

    // Flatten positions from all strategies
    const positions = [];
    if (posData && typeof posData === 'object') {
      for (const [strat, list] of Object.entries(posData)) {
        if (!Array.isArray(list)) continue;
        for (const p of list) {
          positions.push({ ...p, strategy: strat });
        }
      }
    }

    if (positions.length === 0) {
      grid.innerHTML = `
        <div class="empty-state">
          <div class="icon">&#9678;</div>
          No open positions
        </div>`;
      if (countHeader) countHeader.textContent = '';
      return;
    }

    if (countHeader) {
      countHeader.innerHTML = `<span class="text-secondary" style="font-size:11px;text-transform:none;letter-spacing:0;">${positions.length} open</span>`;
    }

    grid.innerHTML = '<div class="stats-grid" style="grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:12px;">' +
      positions.map(pos => {
        const entry = pos.entry_price || 0;
        const current = pos.current_price || pos.last_price || entry;
        const shares = pos.shares || pos.quantity || 0;
        const pnl = (current - entry) * shares * (pos.direction === 'short' ? -1 : 1);
        const pnlPct = entry > 0 ? ((current - entry) / entry) * (pos.direction === 'short' ? -1 : 1) : 0;
        const isPositive = pnl >= 0;
        const tint = isPositive ? 'var(--dim-positive)' : 'var(--dim-negative)';
        const stratName = pos.strategy || 'unknown';

        // Progress bar: map current price between a notional stop and target
        const stopDist = entry * 0.05;
        const stop = pos.direction === 'short' ? entry + stopDist : entry - stopDist;
        const target = pos.direction === 'short' ? entry - stopDist * 2 : entry + stopDist * 2;
        const range = target - stop;
        const pctPos = range !== 0 ? Math.max(0, Math.min(100, ((current - stop) / range) * 100)) : 50;
        const barColor = isPositive ? 'var(--positive)' : 'var(--negative)';

        return `
          <a href="/research?ticker=${pos.symbol}" style="text-decoration:none;color:inherit;">
            <div class="stat-card" style="background:${tint};cursor:pointer;padding:14px;">
              <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">
                <div style="display:flex;align-items:center;gap:8px;">
                  <span style="font-size:18px;font-weight:800;color:var(--text-primary);">${pos.symbol}</span>
                  <span style="font-size:10px;color:var(--text-muted);text-transform:uppercase;">${pos.direction === 'short' ? 'SHORT' : 'LONG'}</span>
                </div>
                <div style="display:flex;align-items:center;gap:4px;">
                  ${stratDot(stratName)}
                  <span style="font-size:10px;color:var(--text-secondary);text-transform:capitalize;">${stratName}</span>
                </div>
              </div>
              <div style="display:flex;justify-content:space-between;font-size:12px;color:var(--text-secondary);margin-bottom:6px;">
                <span>${shares} shares @ ${fmtPrice(entry)}</span>
                <span>Now ${fmtPrice(current)}</span>
              </div>
              <div class="pos-bar" style="margin-bottom:8px;">
                <div class="pos-bar-fill" style="width:${pctPos}%;background:${barColor};opacity:0.4;"></div>
                <div class="pos-bar-marker" style="left:${pctPos}%;"></div>
              </div>
              <div style="display:flex;justify-content:flex-end;gap:8px;font-size:13px;font-weight:700;">
                <span class="${pnlClass(pnl)}">${fmtPnl(pnl)}</span>
                <span class="${pnlClass(pnl)}">(${fmtPct(pnlPct)})</span>
              </div>
            </div>
          </a>`;
      }).join('') +
      '</div>';

  } catch (e) {
    console.error('Failed to load positions:', e);
    grid.innerHTML = `
      <div class="empty-state">
        <div class="icon">&#9678;</div>
        No open positions
      </div>`;
  }
}

// ── Market Overview ─────────────────────────────────────────

async function loadMarketOverview() {
  const row = document.getElementById('market-row');
  if (!row) return;

  try {
    const data = await api('/api/benchmarks/comparison?days=5');
    if (!data) return;

    const indices = ['SPY', 'QQQ', 'IWM', 'DIA', 'VIX'];

    row.innerHTML = indices.map(sym => {
      const series = data[sym];
      if (!series || !series.length) {
        return `<div class="stat-card" style="text-align:center;">
          <div class="stat-label">${sym}</div>
          <div class="stat-value" style="font-size:18px;">&mdash;</div>
        </div>`;
      }

      const latest = series[series.length - 1];
      const price = latest.close || latest.value || 0;
      const prev = series.length > 1
        ? (series[series.length - 2].close || series[series.length - 2].value || price)
        : price;
      const chg = prev ? ((price - prev) / prev) : 0;
      const arrow = chg >= 0 ? '\u25B2' : '\u25BC';

      return `<div class="stat-card" style="text-align:center;">
        <div class="stat-label">${sym}</div>
        <div class="stat-value" style="font-size:18px;">${fmtPrice(price)}</div>
        <div class="stat-sub ${pnlClass(chg)}">
          ${arrow} ${(Math.abs(chg) * 100).toFixed(2)}%
        </div>
      </div>`;
    }).join('');

  } catch (e) {
    console.error('Failed to load market overview:', e);
    row.innerHTML = '<div class="empty-state" style="grid-column:1/-1;">Market data unavailable</div>';
  }
}

// ── Init ────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  loadStats();
  loadEquityCurve(90);
  loadPositions();
  loadMarketOverview();
  initEquityTabs();
});

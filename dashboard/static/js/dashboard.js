/* ═══════════════════════════════════════════════════════════════
   EdgeFinder — Dashboard Page
   ═══════════════════════════════════════════════════════════════ */

const STARTING_CAPITAL = 30000; // 3 strategies x $10,000
const STRATEGY_TARGETS = { coward: 0.15, gambler: 0.25, degenerate: 0.50 };
let equityChart = null;
let equitySeries = null;
let comparisonChart = null;
let refreshInterval = null;
let refreshTimerInterval = null;
let lastUpdate = Date.now();
let autoRefreshEnabled = localStorage.getItem('ef-auto-refresh') !== 'off';

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

    // Aggregate all strategies per timestamp (UTC epoch seconds) so the
    // total-equity curve shows intraday shape, not one point per day.
    const timeMap = {};
    for (const [, series] of Object.entries(data)) {
      if (!Array.isArray(series)) continue;
      for (const pt of series) {
        const t = pt.time;
        if (t == null) continue;
        timeMap[t] = (timeMap[t] || 0) + (pt.total_equity || pt.equity || 0);
      }
    }

    const equityData = Object.keys(timeMap)
      .map(Number).sort((a, b) => a - b)
      .map(t => ({ time: t, value: timeMap[t] }));

    if (equityData.length === 0) {
      // Show flat line at starting capital
      equityData.push({ time: Math.floor(Date.now() / 1000), value: STARTING_CAPITAL });
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
        timeScale: { borderColor: '#1a2332', timeVisible: true, secondsVisible: false },
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

        // Alert detection
        const alertStop = entry * 0.80;
        const targetPct = STRATEGY_TARGETS[stratName] || 0.25;
        const alertTarget = entry * (1 + targetPct);
        const nearStop = current > 0 && current <= alertStop * 1.05;
        const nearTarget = current > 0 && current >= alertTarget * 0.95;
        const pulseClass = nearStop ? 'pulse-stop' : nearTarget ? 'pulse-target' : '';
        const alertBadge = nearStop
          ? '<span class="alert-badge stop">\u26A0\uFE0F Near Stop</span>'
          : nearTarget
          ? '<span class="alert-badge target">\uD83C\uDFAF Near Target</span>'
          : '';

        return `
          <a href="/research?ticker=${pos.symbol}" style="text-decoration:none;color:inherit;">
            <div class="stat-card ${pulseClass}" style="background:${tint};cursor:pointer;padding:14px;">
              <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">
                <div style="display:flex;align-items:center;gap:8px;">
                  <span style="font-size:18px;font-weight:800;color:var(--text-primary);">${pos.symbol}</span>
                  <span style="font-size:10px;color:var(--text-muted);text-transform:uppercase;">${pos.direction === 'short' ? 'SHORT' : 'LONG'}</span>
                  ${alertBadge}
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
    // Indices from benchmarks; regime/VIX/sectors from the captured snapshots.
    const [data, regime] = await Promise.all([
      api('/api/benchmarks/comparison?days=5'),
      api('/api/market/regime?limit=1').catch(() => null),
    ]);
    const latest = regime && regime.latest;

    // Regime badge in the card header
    const badge = document.getElementById('market-regime-badge');
    if (badge) {
      if (latest && latest.regime) {
        const r = String(latest.regime).toLowerCase();
        const cls = r.includes('bull') ? 'text-positive'
                  : r.includes('bear') ? 'text-negative' : 'text-secondary';
        badge.innerHTML = `<span class="${cls}" style="text-transform:capitalize;font-weight:600;">${latest.regime}</span>`;
      } else {
        badge.innerHTML = '';
      }
    }

    if (!data || !data.indices) {
      row.innerHTML = '<div class="empty-state" style="grid-column:1/-1;">Market data unavailable</div>';
      return;
    }

    const symbols = ['SPY', 'QQQ', 'IWM', 'DIA'];
    const dates = data.dates || [];

    let cells = symbols.map(sym => {
      const series = data.indices[sym];
      if (!series || !series.length) {
        return `<div class="stat-card" style="text-align:center;">
          <div class="stat-label">${sym}</div>
          <div class="stat-value" style="font-size:18px;">&mdash;</div>
        </div>`;
      }

      // series is cumulative % change from start — latest value is total change
      const latestPct = series[series.length - 1] || 0;
      // Daily change = difference between last two data points
      const prevPct = series.length > 1 ? series[series.length - 2] : 0;
      const dailyChg = latestPct - prevPct;
      const arrow = dailyChg >= 0 ? '\u25B2' : '\u25BC';
      const latestDate = dates.length ? dates[dates.length - 1] : '';

      return `<div class="stat-card" style="text-align:center;">
        <div class="stat-label">${sym}</div>
        <div class="stat-value" style="font-size:18px;">
          <span class="${dailyChg >= 0 ? 'text-positive' : 'text-negative'}">${arrow} ${Math.abs(dailyChg).toFixed(2)}%</span>
        </div>
        <div class="stat-sub text-secondary">${latestPct >= 0 ? '+' : ''}${latestPct.toFixed(2)}% (5d)</div>
      </div>`;
    }).join('');

    // 5th cell: VIX from the latest captured market snapshot
    const vix = latest && latest.vix != null ? Number(latest.vix) : null;
    cells += `<div class="stat-card" style="text-align:center;">
      <div class="stat-label">VIX</div>
      <div class="stat-value" style="font-size:18px;">${vix != null ? vix.toFixed(2) : '&mdash;'}</div>
      <div class="stat-sub text-secondary">volatility</div>
    </div>`;
    row.innerHTML = cells;

    // Sector performance strip (best → worst)
    const strip = document.getElementById('sector-strip');
    if (strip) {
      const sectors = latest && latest.sector_performance;
      if (sectors && Object.keys(sectors).length) {
        strip.innerHTML = Object.entries(sectors)
          .sort((a, b) => (Number(b[1]) || 0) - (Number(a[1]) || 0))
          .map(([name, pct]) => {
            const p = Number(pct) || 0;
            const cls = p >= 0 ? 'text-positive' : 'text-negative';
            return `<span style="padding:3px 8px;border:1px solid #1a2332;border-radius:4px;font-size:11px;">${name} <span class="${cls}">${p >= 0 ? '+' : ''}${p.toFixed(2)}%</span></span>`;
          }).join('');
      } else {
        strip.innerHTML = '';
      }
    }

  } catch (e) {
    console.error('Failed to load market overview:', e);
    row.innerHTML = '<div class="empty-state" style="grid-column:1/-1;">Market data unavailable</div>';
  }
}

// ── Strategy Comparison Chart ────────────────────────────────

async function loadComparisonChart(days = 90) {
  try {
    const [equityData, benchData] = await Promise.all([
      api('/api/strategies/equity-curve?days=' + days),
      api('/api/benchmarks/comparison?days=' + days),
    ]);

    // Normalize each strategy to cumulative % change
    const series = {};
    if (equityData) {
      for (const [strat, points] of Object.entries(equityData)) {
        if (!Array.isArray(points) || points.length === 0) continue;
        const first = points[0].total_equity || points[0].equity || 0;
        if (first <= 0) continue;
        series[strat] = points
          .filter(p => p.time != null)
          .map(p => ({
            time: p.time,
            value: ((( p.total_equity || p.equity || first) - first) / first) * 100,
          }));
      }
    }

    // Add SPY from benchmarks — convert daily date strings to UTC epoch
    // seconds so the chart shares one time type with the intraday strategy
    // series (lightweight-charts forbids mixing date strings and timestamps).
    if (benchData && benchData.indices && benchData.indices.SPY && benchData.dates) {
      series['SPY'] = benchData.dates.map((d, i) => ({
        time: Math.floor(Date.parse(d + 'T00:00:00Z') / 1000),
        value: benchData.indices.SPY[i] || 0,
      }));
    }

    const container = document.getElementById('comparison-chart');
    if (!container) return;

    if (comparisonChart) {
      comparisonChart.remove();
      comparisonChart = null;
    }

    if (Object.keys(series).length === 0) {
      container.innerHTML = '<div class="empty-state">No comparison data yet</div>';
      return;
    }

    comparisonChart = LightweightCharts.createChart(container, {
      width: container.clientWidth,
      height: 280,
      layout: { background: { color: getComputedStyle(document.body).getPropertyValue('--surface').trim() || '#111a25' }, textColor: '#6b8aab' },
      grid: { vertLines: { color: '#1a2332' }, horzLines: { color: '#1a2332' } },
      timeScale: { borderColor: '#1a2332', timeVisible: true, secondsVisible: false },
      rightPriceScale: { borderColor: '#1a2332' },
    });

    const colors = { coward: '#00d4a1', gambler: '#6ea8fe', degenerate: '#f0b429', SPY: '#6b8aab' };

    for (const [name, data] of Object.entries(series)) {
      const s = comparisonChart.addLineSeries({
        color: colors[name] || '#6b8aab',
        lineWidth: name === 'SPY' ? 1 : 2,
        lineStyle: name === 'SPY' ? 2 : 0,
      });
      s.setData(data);
    }

    comparisonChart.timeScale().fitContent();
    new ResizeObserver(() => {
      comparisonChart.applyOptions({ width: container.clientWidth });
    }).observe(container);

    // Legend
    const legendEl = document.getElementById('comparison-legend');
    if (legendEl) {
      legendEl.innerHTML = Object.keys(series).map(name => {
        const color = colors[name] || '#6b8aab';
        const style = name === 'SPY' ? 'border-bottom: 2px dashed ' + color : 'border-bottom: 2px solid ' + color;
        return `<span style="display:flex;align-items:center;gap:4px;">
          <span style="width:16px;${style};"></span>
          <span style="color:var(--text-secondary);text-transform:capitalize;">${name}</span>
        </span>`;
      }).join('');
    }
  } catch (e) {
    console.error('Failed to load comparison chart:', e);
  }
}

function initComparisonTabs() {
  const tabs = document.getElementById('cmp-range-tabs');
  if (!tabs) return;
  tabs.addEventListener('click', (e) => {
    const tab = e.target.closest('.filter-tab');
    if (!tab) return;
    tabs.querySelectorAll('.filter-tab').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    loadComparisonChart(parseInt(tab.dataset.days, 10));
  });
}

// ── Auto-Refresh ────────────────────────────────────────────

function startAutoRefresh() {
  if (refreshInterval) return;
  refreshInterval = setInterval(() => {
    loadStats();
    loadPositions();
    loadMarketOverview();
    lastUpdate = Date.now();
  }, 60000);
  refreshTimerInterval = setInterval(updateRefreshStatus, 1000);
}

function stopAutoRefresh() {
  clearInterval(refreshInterval);
  clearInterval(refreshTimerInterval);
  refreshInterval = null;
  refreshTimerInterval = null;
}

function toggleAutoRefresh() {
  autoRefreshEnabled = !autoRefreshEnabled;
  localStorage.setItem('ef-auto-refresh', autoRefreshEnabled ? 'on' : 'off');
  const btn = document.getElementById('refresh-toggle');
  if (btn) btn.classList.toggle('active', autoRefreshEnabled);
  if (autoRefreshEnabled) {
    startAutoRefresh();
  } else {
    stopAutoRefresh();
  }
  updateRefreshStatus();
}

function updateRefreshStatus() {
  const el = document.getElementById('refresh-status');
  if (!el) return;
  const ago = Math.floor((Date.now() - lastUpdate) / 1000);
  const agoText = ago < 5 ? 'just now' : ago + 's ago';
  el.textContent = autoRefreshEnabled
    ? 'Auto-refresh: on \u00B7 Updated ' + agoText
    : 'Auto-refresh: paused';
}

// ── Init ────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  loadStats();
  loadEquityCurve(90);
  loadComparisonChart(90);
  loadPositions();
  loadMarketOverview();
  initEquityTabs();
  initComparisonTabs();
  lastUpdate = Date.now();
  if (autoRefreshEnabled) startAutoRefresh();
  updateRefreshStatus();
});

/* ═══════════════════════════════════════════════════════════════
   EdgeFinder — Trades Page
   ═══════════════════════════════════════════════════════════════ */

// ── State ────────────────────────────────────────────────────

let _allTrades = [];        // full fetched trade list
let _activeStatus = 'all';  // current status tab filter
let _sortCol = 'entry_time';
let _sortDir = -1;          // -1 = desc, 1 = asc
let _expandedTradeId = null; // currently expanded timeline row

// ── Stats ─────────────────────────────────────────────────────

async function loadStats(strategy) {
  const statsEl = document.getElementById('trade-stats');
  statsEl.innerHTML = '<div class="stat-card" style="grid-column:1/-1;"><div class="stat-label">Loading...</div></div>';

  try {
    const url = strategy ? '/api/trades/stats?strategy=' + encodeURIComponent(strategy) : '/api/trades/stats';
    const s = await api(url);

    const winRate = s.win_rate != null ? (s.win_rate * 100).toFixed(1) + '%' : '—';
    const avgR    = s.avg_r_multiple != null ? fmtNum(s.avg_r_multiple, 2) + 'R' : '—';
    const best    = s.best_trade != null ? fmtPnl(s.best_trade) : '—';
    const worst   = s.worst_trade != null ? fmtPnl(s.worst_trade) : '—';

    // Avg hold: not always available; derive from trade data if field exists
    const avgHold = s.avg_hold_hours != null
      ? fmtHold(s.avg_hold_hours)
      : (s.avg_hold != null ? s.avg_hold : '—');

    const cards = [
      { label: 'Total Trades', value: s.total_trades != null ? s.total_trades : '—' },
      { label: 'Win Rate',     value: winRate, cls: s.win_rate >= 0.5 ? 'text-positive' : (s.win_rate != null ? 'text-negative' : '') },
      { label: 'Avg R',        value: avgR,    cls: s.avg_r_multiple != null ? pnlClass(s.avg_r_multiple) : '' },
      { label: 'Best Trade',   value: best,    cls: 'text-positive' },
      { label: 'Worst Trade',  value: worst,   cls: s.worst_trade != null && s.worst_trade < 0 ? 'text-negative' : '' },
      { label: 'Avg Hold',     value: avgHold },
    ];

    statsEl.innerHTML = cards.map(c => `
      <div class="stat-card">
        <div class="stat-label">${c.label}</div>
        <div class="stat-value ${c.cls || ''}">${c.value}</div>
      </div>`).join('');

  } catch (e) {
    console.error('Failed to load trade stats:', e);
    statsEl.innerHTML = '<div class="stat-card" style="grid-column:1/-1;"><div class="stat-label text-negative">Failed to load stats.</div></div>';
  }
}

function fmtHold(hours) {
  if (hours == null) return '—';
  if (hours < 1)   return Math.round(hours * 60) + 'm';
  if (hours < 24)  return hours.toFixed(1) + 'h';
  return (hours / 24).toFixed(1) + 'd';
}

// ── Trades Fetch ──────────────────────────────────────────────

async function loadTrades(strategy) {
  const tbody   = document.getElementById('trades-body');
  const countEl = document.getElementById('trade-count');

  tbody.innerHTML = '<tr><td colspan="13" class="empty-state" style="padding:24px;">Loading&hellip;</td></tr>';

  try {
    const url = strategy
      ? '/api/trades?strategy=' + encodeURIComponent(strategy)
      : '/api/trades';
    _allTrades = await api(url);
    if (!Array.isArray(_allTrades)) _allTrades = [];
    renderTable(countEl, tbody);
    renderCalendar(_allTrades);
  } catch (e) {
    console.error('Failed to load trades:', e);
    tbody.innerHTML = '<tr><td colspan="13" class="empty-state" style="padding:24px;">Failed to load trades.</td></tr>';
  }
}

// ── Filtering ─────────────────────────────────────────────────

function filteredTrades() {
  return _allTrades.filter(t => {
    if (_activeStatus === 'all')    return true;
    if (_activeStatus === 'OPEN')   return t.status === 'OPEN';
    if (_activeStatus === 'wins')   return t.status === 'CLOSED' && (t.pnl_dollars || 0) > 0;
    if (_activeStatus === 'losses') return t.status === 'CLOSED' && (t.pnl_dollars || 0) <= 0;
    return true;
  });
}

// ── Sorting ───────────────────────────────────────────────────

function sortedTrades(trades) {
  return [...trades].sort((a, b) => {
    let av = a[_sortCol];
    let bv = b[_sortCol];
    // nulls last
    if (av == null && bv == null) return 0;
    if (av == null) return 1;
    if (bv == null) return -1;
    // string compare
    if (typeof av === 'string' && typeof bv === 'string') {
      return _sortDir * av.localeCompare(bv);
    }
    return _sortDir * (av - bv);
  });
}

function updateSortIndicators() {
  document.querySelectorAll('.data-table th[data-col]').forEach(th => {
    const col = th.dataset.col;
    const arrow = col === _sortCol ? (_sortDir === -1 ? ' ▼' : ' ▲') : '';
    // Strip old indicator and re-add
    th.textContent = th.textContent.replace(/ [▼▲]$/, '') + arrow;
  });
}

// ── Render Table ──────────────────────────────────────────────

function renderTable(countEl, tbody) {
  tbody = tbody || document.getElementById('trades-body');
  countEl = countEl || document.getElementById('trade-count');
  _expandedTradeId = null;

  const trades = sortedTrades(filteredTrades());

  if (countEl) {
    countEl.textContent = trades.length
      ? ' — ' + trades.length + ' trade' + (trades.length !== 1 ? 's' : '')
      : '';
  }

  if (trades.length === 0) {
    tbody.innerHTML = `<tr><td colspan="13" class="empty-state" style="padding:32px;">
      <div class="icon">&#9670;</div>No trades found.
    </td></tr>`;
    return;
  }

  tbody.innerHTML = trades.map(t => renderRow(t)).join('');
  updateSortIndicators();
}

function renderRow(t) {
  const isOpen = t.status === 'OPEN';
  const pnl    = t.pnl_dollars;
  const pnlPct = t.pnl_percent;

  // Symbol — bold, links to research
  const symCell = `<a href="/research?ticker=${encodeURIComponent(t.symbol)}"
    style="font-weight:700;color:var(--text-primary);" title="${t.symbol}">${t.symbol}</a>`;

  // Strategy dot + name
  const stratCell = `<span style="display:inline-flex;align-items:center;gap:5px;">
    ${stratDot(t.strategy_name)}
    <span style="color:var(--text-secondary);text-transform:capitalize;">${t.strategy_name || '—'}</span>
  </span>`;

  // Direction pill
  const dirPill = t.direction === 'LONG'
    ? `<span class="pill pill-positive">LONG</span>`
    : `<span class="pill pill-negative">SHORT</span>`;

  // Entry / Exit price
  const entryCell = t.entry_price != null ? fmtPrice(t.entry_price) : '—';
  const exitCell  = t.exit_price != null
    ? fmtPrice(t.exit_price)
    : (isOpen ? `<span class="text-muted">open</span>` : '—');

  // Shares
  const sharesCell = t.shares != null ? fmtNum(t.shares, 0) : '—';

  // P&L $
  const pnlCell = pnl != null
    ? `<span class="${pnlClass(pnl)}">${fmtPnl(pnl)}</span>`
    : `<span class="text-muted">—</span>`;

  // P&L %
  const pnlPctCell = pnlPct != null
    ? `<span class="${pnlClass(pnlPct)}">${(pnlPct >= 0 ? '+' : '')}${Number(pnlPct).toFixed(2)}%</span>`
    : `<span class="text-muted">—</span>`;

  // R-multiple
  const rCell = t.r_multiple != null
    ? `<span class="${pnlClass(t.r_multiple)}">${fmtNum(t.r_multiple, 2)}R</span>`
    : `<span class="text-muted">—</span>`;

  // Status pill
  let statusPill;
  if (isOpen) {
    statusPill = `<span class="pill pill-accent">OPEN</span>`;
  } else if (t.status === 'CLOSED') {
    statusPill = (pnl != null && pnl > 0)
      ? `<span class="pill pill-positive">WIN</span>`
      : `<span class="pill pill-negative">LOSS</span>`;
  } else {
    statusPill = `<span class="pill pill-muted">${t.status || '—'}</span>`;
  }

  // Exit reason (truncated)
  const reason = t.exit_reason || '—';
  const reasonTrunc = reason.length > 22 ? reason.slice(0, 22) + '…' : reason;
  const reasonCell = `<span class="truncate text-muted" style="max-width:120px;display:inline-block;" title="${reason}">${reasonTrunc}</span>`;

  // Times
  const entryTime = fmtTime(t.entry_time);
  const exitTime  = t.exit_time ? fmtTime(t.exit_time) : `<span class="text-muted">—</span>`;

  return `<tr data-trade-id="${t.trade_id}" onclick="toggleTimeline(${t.trade_id})" style="cursor:pointer;">
    <td>${symCell}</td>
    <td>${stratCell}</td>
    <td>${dirPill}</td>
    <td>${entryCell}</td>
    <td>${exitCell}</td>
    <td style="color:var(--text-secondary);">${sharesCell}</td>
    <td>${pnlCell}</td>
    <td>${pnlPctCell}</td>
    <td>${rCell}</td>
    <td>${statusPill}</td>
    <td>${reasonCell}</td>
    <td style="color:var(--text-secondary);">${entryTime}</td>
    <td style="color:var(--text-muted);">${exitTime}</td>
  </tr>`;
}

// ── P&L Calendar Heatmap ─────────────────────────────────────

function renderCalendar(trades) {
  const container = document.getElementById('pnl-calendar');
  const rangeEl = document.getElementById('calendar-range');
  if (!container) return;

  // Filter to closed trades with exit_time
  const closed = (trades || []).filter(t => t.status === 'CLOSED' && t.exit_time);

  // Group by exit date, sum pnl_dollars
  const pnlByDate = {};
  closed.forEach(t => {
    const d = t.exit_time.slice(0, 10); // YYYY-MM-DD
    pnlByDate[d] = (pnlByDate[d] || 0) + (t.pnl_dollars || 0);
  });

  // Build 91-day range ending today
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const days = [];
  for (let i = 90; i >= 0; i--) {
    const d = new Date(today);
    d.setDate(d.getDate() - i);
    days.push(d);
  }

  // Find start day-of-week (0=Sun..6=Sat) — we need Mon=0 layout
  // Reorganize into 7 rows (Mon-Sun) x 13 columns
  const dayLabels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];

  // Build a grid: rows[dow][weekIdx]
  const grid = Array.from({ length: 7 }, () => Array(13).fill(null));
  days.forEach(d => {
    const diffDays = Math.round((today - d) / 86400000);
    const col = 12 - Math.floor(diffDays / 7);
    let dow = d.getDay() - 1; // Mon=0..Sun=6
    if (dow < 0) dow = 6;
    if (col >= 0 && col < 13) {
      const key = d.toISOString().slice(0, 10);
      grid[dow][col] = { date: d, key: key, pnl: pnlByDate[key] || null };
    }
  });

  // Find max abs pnl for color normalization
  const allPnl = Object.values(pnlByDate).map(Math.abs);
  const maxAbsPnl = allPnl.length ? Math.max(...allPnl) : 1;

  // Set range label
  if (rangeEl) {
    const startStr = days[0].toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    const endStr = days[days.length - 1].toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    rangeEl.textContent = '(' + startStr + ' — ' + endStr + ')';
  }

  // Render
  let html = '';
  for (let row = 0; row < 7; row++) {
    // Day label
    html += '<div class="calendar-label">' + dayLabels[row] + '</div>';
    for (let col = 0; col < 13; col++) {
      const cell = grid[row][col];
      if (!cell) {
        html += '<div class="calendar-cell" style="background:transparent;"></div>';
        continue;
      }
      let bg;
      const dateStr = cell.date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
      if (cell.pnl === null) {
        bg = 'var(--border)';
        html += '<div class="calendar-cell" style="background:' + bg + ';">' +
          '<span class="calendar-tooltip">' + dateStr + ': no trades</span></div>';
      } else {
        const opacity = (Math.abs(cell.pnl) / maxAbsPnl) * 0.8 + 0.1;
        if (cell.pnl >= 0) {
          bg = 'rgba(0,212,161,' + opacity.toFixed(2) + ')';
        } else {
          bg = 'rgba(239,68,68,' + opacity.toFixed(2) + ')';
        }
        const sign = cell.pnl >= 0 ? '+' : '';
        html += '<div class="calendar-cell" style="background:' + bg + ';">' +
          '<span class="calendar-tooltip">' + dateStr + ': ' + sign + '$' + Math.abs(cell.pnl).toFixed(2) + '</span></div>';
      }
    }
  }
  container.innerHTML = html;
}

// ── Trade Reasoning Timeline ─────────────────────────────────

function renderIndicators(ind) {
  if (!ind || typeof ind !== 'object') return '<div style="color:var(--text-muted);font-size:11px;">No indicator data</div>';
  const rows = [];
  if (ind.rsi != null) rows.push(['RSI', ind.rsi.toFixed(1)]);
  if (ind.macd_histogram != null) rows.push(['MACD', ind.macd_histogram.toFixed(3)]);
  if (ind.close != null) rows.push(['Price', '$' + ind.close.toFixed(2)]);
  if (ind.ema_21 != null) rows.push(['EMA 21', '$' + ind.ema_21.toFixed(2)]);
  if (rows.length === 0) return '<div style="color:var(--text-muted);font-size:11px;">No indicator data</div>';
  return '<div class="timeline-indicators">' +
    rows.map(function(r) { return '<span class="label">' + r[0] + '</span><span class="value">' + r[1] + '</span>'; }).join('') +
    '</div>';
}

function formatHoldDuration(hours) {
  if (hours == null) return '\u2014';
  if (hours < 1) return Math.round(hours * 60) + 'm';
  if (hours < 24) return hours.toFixed(1) + 'h';
  const days = Math.floor(hours / 24);
  const rem = Math.round(hours % 24);
  return days + 'd ' + rem + 'h';
}

function renderTimeline(trade) {
  const isWin = (trade.pnl_dollars || 0) > 0;
  const dotColor = isWin ? 'var(--positive)' : 'var(--negative)';
  const lineColor = isWin ? 'rgba(0,212,161,0.3)' : 'rgba(239,68,68,0.3)';

  const entryReasoning = trade.entry_reasoning || 'No reasoning captured';
  const exitReasoning = trade.exit_reasoning || 'No reasoning captured';

  return '<div class="trade-timeline">' +
    '<div class="timeline-card entry">' +
      '<h4>Entry</h4>' +
      '<div style="font-size:12px;color:var(--text-primary);margin-bottom:8px;">' + escapeHtml(entryReasoning) + '</div>' +
      renderIndicators(trade.indicators_at_entry) +
    '</div>' +
    '<div class="timeline-connector">' +
      '<div class="timeline-dot" style="background:' + dotColor + ';"></div>' +
      '<div class="timeline-line" style="background:' + lineColor + ';"></div>' +
      '<div class="timeline-duration">' + formatHoldDuration(trade.hold_duration_hours) + '</div>' +
      '<div class="timeline-line" style="background:' + lineColor + ';"></div>' +
      '<div class="timeline-dot" style="background:' + dotColor + ';"></div>' +
    '</div>' +
    '<div class="timeline-card exit">' +
      '<h4>Exit</h4>' +
      '<div style="font-size:12px;color:var(--text-primary);margin-bottom:8px;">' + escapeHtml(exitReasoning) + '</div>' +
      renderIndicators(trade.indicators_at_exit) +
    '</div>' +
  '</div>';
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

function toggleTimeline(tradeId) {
  const tbody = document.getElementById('trades-body');
  const existing = document.getElementById('timeline-row-' + tradeId);

  // Collapse any open timeline
  const openRow = tbody.querySelector('tr[id^="timeline-row-"]');
  if (openRow) {
    openRow.remove();
    // If we clicked the same row, just collapse
    if (_expandedTradeId === tradeId) {
      _expandedTradeId = null;
      return;
    }
  }

  // Find the trade
  const trade = _allTrades.find(t => t.trade_id === tradeId);
  if (!trade) return;

  _expandedTradeId = tradeId;

  // Find the clicked row and insert after it
  const rows = tbody.querySelectorAll('tr');
  for (const row of rows) {
    if (row.dataset.tradeId === String(tradeId)) {
      const detailRow = document.createElement('tr');
      detailRow.id = 'timeline-row-' + tradeId;
      detailRow.innerHTML = '<td colspan="13" style="padding:0;border-bottom:1px solid var(--border);">' + renderTimeline(trade) + '</td>';
      row.after(detailRow);
      break;
    }
  }
}

// ── Strategy Dropdown ─────────────────────────────────────────

async function loadStrategyFilter() {
  const sel = document.getElementById('filter-strategy');
  try {
    const strategies = await api('/api/strategies');
    if (!Array.isArray(strategies)) return;
    strategies.forEach(s => {
      const opt = document.createElement('option');
      opt.value = s.name || s;
      opt.textContent = (s.name || s);
      sel.appendChild(opt);
    });
  } catch (e) {
    // silently fail — dropdown stays at "All Strategies"
  }
}

// ── Event Wiring ──────────────────────────────────────────────

function initFilters() {
  const sel   = document.getElementById('filter-strategy');
  const tabs  = document.getElementById('status-tabs');

  // Strategy dropdown
  sel.addEventListener('change', async () => {
    const strategy = sel.value;
    await Promise.all([
      loadStats(strategy),
      loadTrades(strategy),
    ]);
  });

  // Status tabs
  tabs.addEventListener('click', e => {
    const tab = e.target.closest('.filter-tab');
    if (!tab) return;
    tabs.querySelectorAll('.filter-tab').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    _activeStatus = tab.dataset.status;
    renderTable();
  });

  // Column sort headers
  document.querySelectorAll('.data-table th[data-col]').forEach(th => {
    th.addEventListener('click', () => {
      const col = th.dataset.col;
      if (_sortCol === col) {
        _sortDir *= -1;
      } else {
        _sortCol = col;
        _sortDir = col === 'entry_time' ? -1 : 1;
      }
      renderTable();
    });
  });
}

// ── Init ─────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', async () => {
  initFilters();
  await Promise.all([
    loadStrategyFilter(),
    loadStats(''),
    loadTrades(''),
  ]);
});

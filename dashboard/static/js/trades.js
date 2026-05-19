/* ═══════════════════════════════════════════════════════════════
   EdgeFinder — Trades Page
   ═══════════════════════════════════════════════════════════════ */

// ── State ────────────────────────────────────────────────────

let _allTrades = [];        // full fetched trade list
let _activeStatus = 'all';  // current status tab filter
let _sortCol = 'entry_time';
let _sortDir = -1;          // -1 = desc, 1 = asc

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

  return `<tr>
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

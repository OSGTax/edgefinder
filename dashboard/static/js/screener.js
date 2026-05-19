/* ═══════════════════════════════════════════════════════════════
   EdgeFinder — Screener Page
   ═══════════════════════════════════════════════════════════════ */

// ── State ─────────────────────────────────────────────────────

let _allStocks = [];          // raw data from /api/research/active
let _filteredStocks = [];     // after filters applied
let _sortCol = 'symbol';
let _sortDir = 1;             // 1 = asc, -1 = desc

// ── Sector Rotation ───────────────────────────────────────────

async function loadSectors() {
  const grid = document.getElementById('sector-grid');
  try {
    const sectors = await api('/api/benchmarks/sectors');
    if (!sectors || !sectors.length) {
      grid.innerHTML = '<div class="empty-state" style="grid-column:1/-1;">No sector data available.</div>';
      return;
    }
    grid.innerHTML = sectors.map(s => renderSectorItem(s)).join('');
  } catch (e) {
    console.error('Failed to load sectors:', e);
    grid.innerHTML = '<div class="empty-state" style="grid-column:1/-1;">Sector data unavailable.</div>';
  }
}

function quadrantPillClass(quadrant) {
  switch ((quadrant || '').toLowerCase()) {
    case 'leading':   return 'pill-positive';
    case 'improving': return 'pill-warning';
    case 'weakening': return 'pill-muted';
    case 'lagging':   return 'pill-negative';
    default:          return 'pill-muted';
  }
}

function renderSectorItem(s) {
  const pillClass = quadrantPillClass(s.quadrant);
  const label = s.quadrant
    ? s.quadrant.charAt(0).toUpperCase() + s.quadrant.slice(1).toLowerCase()
    : '—';
  const ret = s.return_1m != null
    ? `<span class="${s.return_1m >= 0 ? 'text-positive' : 'text-negative'}" style="font-size:11px;margin-left:auto;padding-right:4px;">${s.return_1m >= 0 ? '+' : ''}${Number(s.return_1m).toFixed(1)}%</span>`
    : '';
  return `
    <div class="sector-item">
      <span class="etf">${s.symbol}</span>
      <span class="name" title="${s.name || ''}">${s.name || '—'}</span>
      ${ret}
      <span class="pill ${pillClass}">${label}</span>
    </div>`;
}

// ── Stock Screener ─────────────────────────────────────────────

async function loadStocks() {
  try {
    const stocks = await api('/api/research/active');
    _allStocks = Array.isArray(stocks) ? stocks : [];
    populateSectorFilter();
    applyFilters();
  } catch (e) {
    console.error('Failed to load stocks:', e);
    document.getElementById('screener-body').innerHTML =
      '<tr><td colspan="12" class="empty-state" style="padding:40px;">Failed to load stock data.</td></tr>';
  }
}

async function loadStrategies() {
  const sel = document.getElementById('filter-strategy');
  try {
    const strategies = await api('/api/strategies');
    const names = Array.isArray(strategies)
      ? strategies
      : (strategies && strategies.strategies ? strategies.strategies : []);
    names.forEach(name => {
      const opt = document.createElement('option');
      opt.value = name;
      opt.textContent = name.charAt(0).toUpperCase() + name.slice(1);
      sel.appendChild(opt);
    });
  } catch (e) {
    // silently fail — filter just won't be populated
  }
}

function populateSectorFilter() {
  const sel = document.getElementById('filter-sector');
  // Preserve current selection
  const current = sel.value;
  // Remove all but first option
  while (sel.options.length > 1) sel.remove(1);

  const sectors = [...new Set(
    _allStocks
      .map(s => s.sector)
      .filter(Boolean)
      .sort()
  )];
  sectors.forEach(sector => {
    const opt = document.createElement('option');
    opt.value = sector;
    opt.textContent = sector;
    sel.appendChild(opt);
  });

  if (current) sel.value = current;
}

// ── Filtering ─────────────────────────────────────────────────

function applyFilters() {
  const stratFilter = document.getElementById('filter-strategy').value.toLowerCase();
  const sectorFilter = document.getElementById('filter-sector').value.toLowerCase();
  const searchFilter = document.getElementById('filter-search').value.toLowerCase().trim();

  _filteredStocks = _allStocks.filter(s => {
    // Strategy filter: stock must have the strategy in qualifying_strategies
    if (stratFilter) {
      const qs = s.qualifying_strategies || [];
      if (!qs.some(q => q.toLowerCase() === stratFilter)) return false;
    }

    // Sector filter
    if (sectorFilter && (s.sector || '').toLowerCase() !== sectorFilter) return false;

    // Search filter: symbol or company name
    if (searchFilter) {
      const sym = (s.symbol || '').toLowerCase();
      const co = (s.company_name || '').toLowerCase();
      if (!sym.includes(searchFilter) && !co.includes(searchFilter)) return false;
    }

    return true;
  });

  sortStocks();
  renderTable();
}

// ── Sorting ───────────────────────────────────────────────────

function sortStocks() {
  const col = _sortCol;
  _filteredStocks.sort((a, b) => {
    let av = a[col];
    let bv = b[col];
    // Nulls to end regardless of direction
    if (av == null && bv == null) return 0;
    if (av == null) return 1;
    if (bv == null) return -1;
    if (typeof av === 'string') return av.localeCompare(bv) * _sortDir;
    return (av - bv) * _sortDir;
  });
}

function initSortHeaders() {
  document.querySelectorAll('#screener-table thead th[data-col]').forEach(th => {
    th.addEventListener('click', () => {
      const col = th.dataset.col;
      if (_sortCol === col) {
        _sortDir *= -1;
      } else {
        _sortCol = col;
        _sortDir = 1;
      }
      // Update header indicators
      document.querySelectorAll('#screener-table thead th').forEach(h => {
        h.style.color = '';
      });
      th.style.color = 'var(--accent)';

      sortStocks();
      renderTable();
    });
  });
}

// ── Table Rendering ───────────────────────────────────────────

function fmtMarketCap(n) {
  if (n == null) return '—';
  const abs = Math.abs(n);
  if (abs >= 1e12) return '$' + (n / 1e12).toFixed(1) + 'T';
  if (abs >= 1e9)  return '$' + (n / 1e9).toFixed(1) + 'B';
  if (abs >= 1e6)  return '$' + (n / 1e6).toFixed(1) + 'M';
  return '$' + n.toLocaleString('en-US', { maximumFractionDigits: 0 });
}

function fmtGrowth(n) {
  if (n == null) return '<span class="text-muted">—</span>';
  const sign = n >= 0 ? '+' : '';
  const cls = n >= 0 ? 'text-positive' : 'text-negative';
  return `<span class="${cls}">${sign}${(n * 100).toFixed(1)}%</span>`;
}

function fmtRsi(n) {
  if (n == null) return '<span class="text-muted">—</span>';
  const val = Number(n).toFixed(1);
  if (n < 30) return `<span class="text-positive">${val}</span>`;
  if (n > 70) return `<span class="text-negative">${val}</span>`;
  return `<span>${val}</span>`;
}

function fmtMacd(n) {
  if (n == null) return '<span class="text-muted">—</span>';
  const val = Number(n).toFixed(3);
  const cls = n >= 0 ? 'text-positive' : 'text-negative';
  return `<span class="${cls}">${n >= 0 ? '+' : ''}${val}</span>`;
}

function fmtPe(n) {
  if (n == null) return '<span class="text-muted">—</span>';
  const val = Number(n).toFixed(1);
  return `<span>${val}</span>`;
}

function fmtSi(n) {
  if (n == null) return '<span class="text-muted">—</span>';
  const val = (n * 100).toFixed(1);
  return `<span>${val}%</span>`;
}

function renderStratDots(strategies) {
  if (!strategies || !strategies.length) return '<span class="text-muted">—</span>';
  return strategies
    .map(name => `<span title="${name}">${stratDot(name)}</span>`)
    .join(' ');
}

function renderTable() {
  const tbody = document.getElementById('screener-body');
  const countEl = document.getElementById('stock-count');

  const n = _filteredStocks.length;
  const total = _allStocks.length;
  if (countEl) {
    countEl.textContent = n === total
      ? `— ${n} stocks`
      : `— ${n} of ${total} stocks`;
  }

  if (n === 0) {
    tbody.innerHTML = '<tr><td colspan="12" class="empty-state" style="padding:40px;">No stocks match the current filters.</td></tr>';
    return;
  }

  tbody.innerHTML = _filteredStocks.map(s => {
    const co = s.company_name || '—';
    const coTrunc = co.length > 24 ? co.slice(0, 24) + '…' : co;
    const sector = s.sector || '—';

    return `<tr>
      <td style="font-weight:700;color:var(--text-primary);">
        <a href="/research?ticker=${encodeURIComponent(s.symbol)}" style="color:inherit;text-decoration:none;"
           onmouseover="this.style.color='var(--accent)'"
           onmouseout="this.style.color='inherit'">
          ${s.symbol}
        </a>
      </td>
      <td class="truncate" style="max-width:160px;color:var(--text-secondary);" title="${co}">${coTrunc}</td>
      <td style="color:var(--text-muted);font-size:11px;">${sector}</td>
      <td class="text-right">${fmtPrice(s.last_price)}</td>
      <td class="text-right">${fmtMarketCap(s.market_cap)}</td>
      <td class="text-right">${fmtGrowth(s.earnings_growth)}</td>
      <td class="text-right">${fmtGrowth(s.revenue_growth)}</td>
      <td class="text-right">${fmtPe(s.price_to_earnings)}</td>
      <td class="text-right">${fmtRsi(s.rsi_14)}</td>
      <td class="text-right">${fmtMacd(s.macd_histogram)}</td>
      <td class="text-right">${fmtSi(s.short_interest)}</td>
      <td style="white-space:nowrap;">${renderStratDots(s.qualifying_strategies)}</td>
    </tr>`;
  }).join('');
}

// ── Filter Event Listeners ─────────────────────────────────────

function initFilters() {
  document.getElementById('filter-strategy').addEventListener('change', applyFilters);
  document.getElementById('filter-sector').addEventListener('change', applyFilters);
  document.getElementById('filter-search').addEventListener('input', applyFilters);
}

// ── Init ──────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  initSortHeaders();
  initFilters();
  // Load all data in parallel
  Promise.all([
    loadSectors(),
    loadStocks(),
    loadStrategies(),
  ]);
});

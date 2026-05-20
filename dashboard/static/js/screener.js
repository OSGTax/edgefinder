/* ═══════════════════════════════════════════════════════════════
   EdgeFinder — Screener Page
   ═══════════════════════════════════════════════════════════════ */

// ── State ─────────────────────────────────────────────────────

let _allStocks = [];          // raw data from /api/research/active
let _filteredStocks = [];     // after filters applied
let _sortCol = 'symbol';
let _sortDir = 1;             // 1 = asc, -1 = desc

// ── Sector Rotation (Treemap) ─────────────────────────────────

let treemapChart = null;

async function loadSectors() {
  try {
    const sectorData = await api('/api/benchmarks/sectors').then(r => r.sectors || r).catch(() => []);

    const canvas = document.getElementById('sector-treemap');
    if (!canvas) return;

    // Count stocks per sector name from screener data
    const sectorCounts = {};
    const stockList = _allStocks || [];
    for (const s of stockList) {
      const sec = s.sector || 'Unknown';
      sectorCounts[sec] = (sectorCounts[sec] || 0) + 1;
    }

    // Build treemap data: merge sector rotation quadrants with stock counts
    const quadrantMap = {};
    if (Array.isArray(sectorData)) {
      for (const s of sectorData) {
        quadrantMap[s.name || s.symbol] = s.quadrant || 'unknown';
        if (s.symbol) quadrantMap[s.symbol] = s.quadrant || 'unknown';
      }
    }

    const SECTOR_NAME_MAP = {
      'Technology': 'XLK', 'Financial Services': 'XLF', 'Energy': 'XLE',
      'Healthcare': 'XLV', 'Industrials': 'XLI', 'Consumer Defensive': 'XLP',
      'Consumer Cyclical': 'XLY', 'Utilities': 'XLU', 'Real Estate': 'XLRE',
      'Communication Services': 'XLC', 'Basic Materials': 'XLB',
    };

    const treemapData = Object.entries(sectorCounts).map(([name, count]) => {
      const etf = SECTOR_NAME_MAP[name];
      const quadrant = quadrantMap[name] || quadrantMap[etf] || 'unknown';
      return { sector: name, count, quadrant };
    }).filter(d => d.count > 0).sort((a, b) => b.count - a.count);

    const quadrantColors = {
      leading: '#00d4a1', improving: '#f0b429',
      weakening: '#4a6a8a', lagging: '#ef4444',
      unknown: '#2a3a4a',
    };

    if (treemapChart) {
      treemapChart.destroy();
      treemapChart = null;
    }

    if (treemapData.length === 0) {
      canvas.parentElement.innerHTML = '<div class="empty-state">Sector data unavailable</div>';
      return;
    }

    treemapChart = new Chart(canvas, {
      type: 'treemap',
      data: {
        datasets: [{
          tree: treemapData,
          key: 'count',
          labels: {
            display: true,
            color: '#e8f0f8',
            font: { size: 11, weight: 'bold' },
            formatter: (ctx) => {
              const d = ctx.raw._data;
              return d ? d.sector + ' (' + d.count + ')' : '';
            },
          },
          backgroundColor: (ctx) => {
            const d = ctx.raw && ctx.raw._data;
            return d ? (quadrantColors[d.quadrant] || '#2a3a4a') : '#2a3a4a';
          },
          borderColor: 'rgba(0,0,0,0.2)',
          borderWidth: 2,
          spacing: 2,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              title: (items) => {
                const d = items[0] && items[0].raw && items[0].raw._data;
                return d ? d.sector : '';
              },
              label: (item) => {
                const d = item.raw && item.raw._data;
                return d ? d.count + ' stocks \u00b7 ' + d.quadrant : '';
              },
            }
          }
        },
        onClick: (evt, elements) => {
          if (elements.length > 0) {
            const d = elements[0].element.$context.raw._data;
            if (d && d.sector) {
              const filterEl = document.getElementById('filter-sector');
              if (filterEl) {
                filterEl.value = d.sector;
                if (typeof applyFilters === 'function') applyFilters();
                else if (typeof renderTable === 'function') renderTable();
              }
            }
          }
        }
      }
    });

  } catch (e) {
    console.error('Failed to load sectors:', e);
  }
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

function renderRsiGauge(rsi) {
  if (rsi == null) return '<span class="text-muted">—</span>';
  const pct = Math.min(100, Math.max(0, rsi));
  const cls = rsi < 30 ? 'oversold' : rsi > 70 ? 'overbought' : 'neutral';
  return `<span class="rsi-gauge">
    <span class="rsi-bar"><span class="rsi-fill ${cls}" style="width:${pct}%"></span></span>
    <span style="font-size:11px;${rsi < 30 ? 'color:var(--positive)' : rsi > 70 ? 'color:var(--negative)' : ''}">${rsi.toFixed(0)}</span>
  </span>`;
}

function renderMacd(val) {
  if (val == null) return '<span class="text-muted">—</span>';
  const cls = val > 0 ? 'text-positive' : 'text-negative';
  const arrow = val > 0 ? '▲' : '▼';
  return `<span class="${cls}">${arrow} ${val.toFixed(2)}</span>`;
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
      <td class="text-right">${renderRsiGauge(s.rsi_14)}</td>
      <td class="text-right">${renderMacd(s.macd_histogram)}</td>
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
  // Load stocks first (treemap needs sector counts from stock data),
  // then load sectors; strategies can load in parallel.
  Promise.all([
    loadStocks().then(() => loadSectors()),
    loadStrategies(),
  ]);
});

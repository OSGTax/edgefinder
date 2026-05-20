/* ═══════════════════════════════════════════════════════════════
   EdgeFinder — Common Utilities
   ═══════════════════════════════════════════════════════════════ */

// ── API Fetch ────────────────────────────────────────────────

async function api(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`API ${path}: ${res.status}`);
  return res.json();
}

// ── Number Formatting ────────────────────────────────────────

function fmtDollar(n) {
  if (n == null) return '\u2014';
  const abs = Math.abs(n);
  if (abs >= 1e12) return (n < 0 ? '-' : '') + '$' + (abs / 1e12).toFixed(1) + 'T';
  if (abs >= 1e9) return (n < 0 ? '-' : '') + '$' + (abs / 1e9).toFixed(1) + 'B';
  if (abs >= 1e6) return (n < 0 ? '-' : '') + '$' + (abs / 1e6).toFixed(1) + 'M';
  return '$' + n.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function fmtPrice(n) {
  if (n == null) return '\u2014';
  return '$' + Number(n).toFixed(2);
}

function fmtPct(n) {
  if (n == null) return '\u2014';
  const sign = n >= 0 ? '+' : '';
  return sign + (n * 100).toFixed(1) + '%';
}

function fmtPctRaw(n) {
  if (n == null) return '\u2014';
  const sign = n >= 0 ? '+' : '';
  return sign + Number(n).toFixed(1) + '%';
}

function fmtNum(n, decimals = 1) {
  if (n == null) return '\u2014';
  return Number(n).toFixed(decimals);
}

function fmtInt(n) {
  if (n == null) return '\u2014';
  return Number(n).toLocaleString('en-US', { maximumFractionDigits: 0 });
}

// ── P&L Formatting ───────────────────────────────────────────

function fmtPnl(n) {
  if (n == null) return '\u2014';
  const sign = n >= 0 ? '+' : '';
  return sign + fmtDollar(n).replace('$', '$');
}

function pnlClass(n) {
  if (n == null) return '';
  return n >= 0 ? 'text-positive' : 'text-negative';
}

function pnlSign(n) {
  if (n == null) return '';
  return n >= 0 ? '\u25B2' : '\u25BC';
}

// ── Time Formatting ──────────────────────────────────────────

function fmtTime(iso) {
  if (!iso) return '\u2014';
  const d = new Date(iso);
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }) + ' ' +
         d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
}

function fmtDate(iso) {
  if (!iso) return '\u2014';
  const d = new Date(iso);
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
}

function timeAgo(iso) {
  if (!iso) return '';
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 60) return mins + 'm ago';
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return hrs + 'h ago';
  const days = Math.floor(hrs / 24);
  return days + 'd ago';
}

// ── Strategy Helpers ─────────────────────────────────────────

const STRATEGY_COLORS = {
  coward: '#00d4a1',
  gambler: '#6ea8fe',
  degenerate: '#f0b429',
};

function stratColor(name) {
  return STRATEGY_COLORS[name] || '#6b8aab';
}

function stratDot(name) {
  return `<span class="strat-dot strat-dot-${name}"></span>`;
}

// ── Gauge ────────────────────────────────────────────────────

function gaugeClass(pct) {
  if (pct < 0.10) return 'good';
  if (pct < 0.15) return 'warn';
  return 'danger';
}

// ── Nav State ────────────────────────────────────────────────

function initNav() {
  const path = window.location.pathname;
  document.querySelectorAll('.topnav-tab').forEach(tab => {
    if (tab.getAttribute('href') === path) {
      tab.classList.add('active');
    }
  });
}

// ── Market Indices (top bar) ─────────────────────────────────

async function loadTopBarIndices() {
  try {
    // API returns { dates: [...], indices: { SPY: [cumPct, ...], ... } }
    const data = await api('/api/benchmarks/comparison?days=5');
    const el = document.getElementById('topnav-indices');
    if (!el || !data || !data.indices) return;

    const symbols = ['SPY', 'QQQ', 'VIX'];
    el.innerHTML = symbols.map(sym => {
      const series = data.indices[sym];
      if (!series || !series.length) return '';
      const latestPct = series[series.length - 1] || 0;
      const prevPct = series.length > 1 ? series[series.length - 2] : 0;
      const dailyChg = latestPct - prevPct;
      const cls = dailyChg >= 0 ? 'text-positive' : 'text-negative';
      const arrow = dailyChg >= 0 ? '\u25B2' : '\u25BC';
      return `<div class="topnav-index">
        <span class="sym">${sym}</span>
        <span class="${cls}">${arrow}${Math.abs(dailyChg).toFixed(2)}%</span>
      </div>`;
    }).join('');
  } catch (e) {
    // silently fail — indices are decorative
  }
}

async function loadTopBarHealth() {
  try {
    const data = await api('/api/health');
    const el = document.getElementById('topnav-status');
    if (!el) return;
    const ok = data.status === 'ok';
    el.innerHTML = `<span class="dot" style="background:${ok ? 'var(--positive)' : 'var(--negative)'}"></span> v${data.version}`;
  } catch (e) {}
}

// ── Init (runs on every page) ────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  initNav();
  loadTopBarIndices();
  loadTopBarHealth();
});

/* =================================================================
   EdgeFinder — Research Page
   Full ticker deep-dive: search, fundamentals, technicals, trades
   ================================================================= */

(function () {
  'use strict';

  const searchInput  = document.getElementById('search-input');
  const searchResults = document.getElementById('search-results');
  const profileEl    = document.getElementById('ticker-profile');
  const emptyEl      = document.getElementById('research-empty');

  let debounceTimer = null;
  let currentSymbol = null;

  // ── Search Autocomplete ───────────────────────────────────

  searchInput.addEventListener('input', () => {
    clearTimeout(debounceTimer);
    const q = searchInput.value.trim();
    if (q.length < 2) { closeDropdown(); return; }
    debounceTimer = setTimeout(() => fetchSearch(q), 250);
  });

  searchInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      const q = searchInput.value.trim().toUpperCase();
      if (q) { closeDropdown(); loadTicker(q); }
    }
    if (e.key === 'Escape') closeDropdown();
  });

  document.addEventListener('click', (e) => {
    if (!searchInput.contains(e.target) && !searchResults.contains(e.target)) {
      closeDropdown();
    }
  });

  async function fetchSearch(q) {
    try {
      const data = await api(`/api/research/search?q=${encodeURIComponent(q)}&limit=10`);
      if (!data || !data.length) { closeDropdown(); return; }
      searchResults.innerHTML = data.map(r => `
        <div class="search-result" data-symbol="${r.symbol}">
          <span class="sym">${r.symbol}</span>
          <span class="name">${r.company_name || ''}</span>
          ${r.sector ? `<span class="text-muted" style="margin-left:auto;font-size:11px">${r.sector}</span>` : ''}
          ${r.is_active ? '<span class="pill pill-accent" style="margin-left:8px">Active</span>' : ''}
        </div>
      `).join('');
      searchResults.classList.add('open');
      searchResults.querySelectorAll('.search-result').forEach(el => {
        el.addEventListener('click', () => {
          closeDropdown();
          loadTicker(el.dataset.symbol);
        });
      });
    } catch (e) {
      closeDropdown();
    }
  }

  function closeDropdown() {
    searchResults.classList.remove('open');
    searchResults.innerHTML = '';
  }

  // ── Load Ticker Profile ───────────────────────────────────

  async function loadTicker(symbol) {
    symbol = symbol.toUpperCase();
    currentSymbol = symbol;
    searchInput.value = symbol;

    // Update URL without reload
    const url = new URL(window.location);
    url.searchParams.set('ticker', symbol);
    window.history.replaceState({}, '', url);

    emptyEl.style.display = 'none';
    profileEl.style.display = 'block';
    profileEl.innerHTML = '<div class="empty-state">Loading...</div>';

    try {
      const [report, trades] = await Promise.all([
        api(`/api/research/ticker/${symbol}`),
        api(`/api/trades?symbol=${symbol}&limit=50`).catch(() => []),
      ]);
      renderProfile(report, trades);
    } catch (e) {
      profileEl.innerHTML = `<div class="empty-state"><div class="icon">&#9888;</div>Could not load data for ${symbol}</div>`;
    }
  }

  // ── Render Full Profile ───────────────────────────────────

  function renderProfile(r, trades) {
    const f = r.fundamentals || {};
    const ind = r.indicators || {};

    profileEl.innerHTML = `
      ${renderHeader(r)}
      ${renderPriceDisplay(r, ind)}
      <div class="grid-2 mb-16">
        ${renderFundamentals(f)}
        ${renderTechnicals(ind)}
      </div>
      ${renderShortInterest(f)}
      ${renderDividends(r.dividends)}
      ${renderNews(r.recent_news, f)}
      ${renderTradeHistory(trades)}
      ${renderRelated(r.related_tickers)}
    `;
  }

  // ── A. Header Card ────────────────────────────────────────

  function renderHeader(r) {
    const strats = (r.qualifying_strategies || []).map(s => stratDot(s)).join(' ');
    const activeTag = r.is_active ? '<span class="pill pill-accent">Active</span>' : '';
    return `
      <div class="card mb-16">
        <div class="card-body" style="display:flex;align-items:center;gap:16px;flex-wrap:wrap">
          <div style="flex:1;min-width:200px">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px">
              <span style="font-size:22px;font-weight:800;color:var(--accent)">${r.symbol}</span>
              ${activeTag}
              ${strats ? `<span style="display:flex;gap:4px;align-items:center">${strats}</span>` : ''}
            </div>
            <div style="font-size:14px;color:var(--text-secondary)">${r.company_name || 'Unknown'}</div>
            <div style="font-size:12px;color:var(--text-muted);margin-top:2px">
              ${r.sector || ''} ${r.industry ? '&middot; ' + r.industry : ''}
            </div>
          </div>
          <div style="text-align:right">
            <div style="font-size:28px;font-weight:800;color:var(--text-primary)">${fmtPrice(r.price)}</div>
            <div style="font-size:12px;color:var(--text-muted)">Market Cap: ${fmtDollar(r.market_cap)}</div>
          </div>
        </div>
      </div>
    `;
  }

  // ── B. Price + Indicator Display ──────────────────────────

  function renderPriceDisplay(r, ind) {
    if (!ind || Object.keys(ind).length === 0) return '';

    const emaRows = [
      ['EMA 9 (Fast Day)', ind.ema_fast_day],
      ['EMA 21 (Slow Day)', ind.ema_slow_day],
      ['EMA 9 (Fast Swing)', ind.ema_fast_swing],
      ['EMA 21 (Slow Swing)', ind.ema_slow_swing],
    ].filter(([, v]) => v != null);

    const price = r.price || ind.close;
    const bbPercent = (ind.bb_upper && ind.bb_lower && price)
      ? ((price - ind.bb_lower) / (ind.bb_upper - ind.bb_lower) * 100).toFixed(1)
      : null;

    return `
      <div class="card mb-16">
        <div class="card-header">Price Context</div>
        <div class="card-body">
          <div style="display:flex;gap:24px;flex-wrap:wrap">
            ${emaRows.length ? `
            <div style="flex:1;min-width:200px">
              <div style="font-size:10px;font-weight:600;color:var(--text-muted);text-transform:uppercase;margin-bottom:8px">Moving Averages</div>
              ${emaRows.map(([label, val]) => {
                const above = price && val ? price > val : null;
                const cls = above === true ? 'text-positive' : above === false ? 'text-negative' : '';
                return `<div class="data-row"><span class="label">${label}</span><span class="value ${cls}">${fmtPrice(val)}</span></div>`;
              }).join('')}
            </div>` : ''}
            <div style="flex:1;min-width:200px">
              <div style="font-size:10px;font-weight:600;color:var(--text-muted);text-transform:uppercase;margin-bottom:8px">Bollinger Bands</div>
              <div class="data-row"><span class="label">Upper</span><span class="value">${fmtPrice(ind.bb_upper)}</span></div>
              <div class="data-row"><span class="label">Middle</span><span class="value">${fmtPrice(ind.bb_middle)}</span></div>
              <div class="data-row"><span class="label">Lower</span><span class="value">${fmtPrice(ind.bb_lower)}</span></div>
              ${bbPercent !== null ? `<div class="data-row"><span class="label">%B Position</span><span class="value">${bbPercent}%</span></div>` : ''}
              ${ind.bb_width != null ? `<div class="data-row"><span class="label">BB Width</span><span class="value">${fmtNum(ind.bb_width, 3)}</span></div>` : ''}
            </div>
            <div style="flex:1;min-width:160px">
              <div style="font-size:10px;font-weight:600;color:var(--text-muted);text-transform:uppercase;margin-bottom:8px">Volatility</div>
              ${ind.atr != null ? `<div class="data-row"><span class="label">ATR</span><span class="value">${fmtPrice(ind.atr)}</span></div>` : ''}
              ${ind.recent_high != null ? `<div class="data-row"><span class="label">Recent High</span><span class="value">${fmtPrice(ind.recent_high)}</span></div>` : ''}
              ${ind.recent_low != null ? `<div class="data-row"><span class="label">Recent Low</span><span class="value">${fmtPrice(ind.recent_low)}</span></div>` : ''}
              ${ind.volume_ratio != null ? `<div class="data-row"><span class="label">Vol Ratio</span><span class="value">${fmtNum(ind.volume_ratio, 2)}x</span></div>` : ''}
            </div>
          </div>
        </div>
      </div>
    `;
  }

  // ── C-Left. Fundamentals Card ─────────────────────────────

  function renderFundamentals(f) {
    if (!f || Object.keys(f).length === 0) {
      return '<div class="card"><div class="card-header">Fundamentals</div><div class="card-body"><div class="empty-state">No fundamental data available</div></div></div>';
    }

    function section(title, rows) {
      const valid = rows.filter(([, v]) => v != null);
      if (!valid.length) return '';
      return `
        <div style="margin-bottom:12px">
          <div style="font-size:10px;font-weight:600;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px">${title}</div>
          ${valid.map(([label, val, fmt]) => `<div class="data-row"><span class="label">${label}</span><span class="value">${fmt ? fmt(val) : val}</span></div>`).join('')}
        </div>
      `;
    }

    return `
      <div class="card">
        <div class="card-header">Fundamentals</div>
        <div class="card-body">
          ${section('Valuation', [
            ['P/E Ratio', f.price_to_earnings, fmtNum],
            ['P/B Ratio', f.price_to_book, fmtNum],
            ['PEG Ratio', f.peg_ratio, fmtNum],
            ['EV/EBITDA', f.ev_to_ebitda, fmtNum],
            ['P/Tangible Book', f.price_to_tangible_book, fmtNum],
          ])}
          ${section('Growth', [
            ['Earnings Growth', f.earnings_growth, fmtPct],
            ['Revenue Growth', f.revenue_growth, fmtPct],
          ])}
          ${section('Returns', [
            ['Return on Equity', f.return_on_equity, fmtPct],
            ['Return on Assets', f.return_on_assets, fmtPct],
          ])}
          ${section('Financial Health', [
            ['Debt / Equity', f.debt_to_equity, fmtNum],
            ['Current Ratio', f.current_ratio, fmtNum],
            ['Quick Ratio', f.quick_ratio, fmtNum],
            ['FCF Yield', f.fcf_yield, fmtPct],
            ['Free Cash Flow', f.free_cash_flow, fmtDollar],
          ])}
          ${section('Ownership', [
            ['Institutional %', f.institutional_pct, fmtPct],
          ])}
          ${section('Sentiment', [
            ['News Sentiment', f.news_sentiment, (v) => {
              const cls = v > 0.2 ? 'text-positive' : v < -0.2 ? 'text-negative' : 'text-warning';
              return `<span class="${cls}">${fmtNum(v, 2)}</span>`;
            }],
          ])}
        </div>
      </div>
    `;
  }

  // ── C-Right. Technicals Card ──────────────────────────────

  function renderTechnicals(ind) {
    if (!ind || Object.keys(ind).length === 0) {
      return '<div class="card"><div class="card-header">Technicals</div><div class="card-body"><div class="empty-state">No technical data available</div></div></div>';
    }

    // RSI gauge
    const rsi = ind.rsi;
    let rsiGauge = '';
    if (rsi != null) {
      const fillCls = rsi < 30 ? 'good' : rsi > 70 ? 'danger' : 'warn';
      const rsiLabel = rsi < 30 ? 'Oversold' : rsi > 70 ? 'Overbought' : 'Neutral';
      const rsiLabelCls = rsi < 30 ? 'text-positive' : rsi > 70 ? 'text-negative' : 'text-secondary';
      rsiGauge = `
        <div style="margin-bottom:16px">
          <div style="display:flex;justify-content:space-between;margin-bottom:4px">
            <span style="font-size:10px;font-weight:600;color:var(--text-muted);text-transform:uppercase">RSI (14)</span>
            <span style="font-size:13px;font-weight:700" class="${rsiLabelCls}">${fmtNum(rsi, 1)} &mdash; ${rsiLabel}</span>
          </div>
          <div class="gauge"><div class="gauge-fill ${fillCls}" style="width:${Math.min(rsi, 100)}%"></div></div>
          <div style="display:flex;justify-content:space-between;font-size:10px;color:var(--text-muted);margin-top:2px">
            <span>0</span><span>30</span><span>50</span><span>70</span><span>100</span>
          </div>
        </div>
      `;
    }

    // MACD section
    let macdSection = '';
    if (ind.macd_line != null || ind.macd_signal != null) {
      const histVal = ind.macd_histogram;
      const histCls = histVal != null ? (histVal >= 0 ? 'text-positive' : 'text-negative') : '';
      macdSection = `
        <div style="margin-bottom:12px">
          <div style="font-size:10px;font-weight:600;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px">MACD</div>
          <div class="data-row"><span class="label">MACD Line</span><span class="value">${fmtNum(ind.macd_line, 3)}</span></div>
          <div class="data-row"><span class="label">Signal Line</span><span class="value">${fmtNum(ind.macd_signal, 3)}</span></div>
          <div class="data-row"><span class="label">Histogram</span><span class="value ${histCls}">${fmtNum(histVal, 3)}</span></div>
        </div>
      `;
    }

    // Momentum section
    const momentumRows = [
      ['Stochastic RSI K', ind.stoch_rsi_k, fmtNum],
      ['Stochastic RSI D', ind.stoch_rsi_d, fmtNum],
      ['Williams %R', ind.williams_r, fmtNum],
    ].filter(([, v]) => v != null);

    // Trend section
    const trendRows = [
      ['ADX', ind.adx, fmtNum],
      ['+DI', ind.plus_di, fmtNum],
      ['-DI', ind.minus_di, fmtNum],
    ].filter(([, v]) => v != null);

    return `
      <div class="card">
        <div class="card-header">Technicals</div>
        <div class="card-body">
          ${rsiGauge}
          ${macdSection}
          ${momentumRows.length ? `
            <div style="margin-bottom:12px">
              <div style="font-size:10px;font-weight:600;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px">Momentum</div>
              ${momentumRows.map(([label, val, fmt]) => `<div class="data-row"><span class="label">${label}</span><span class="value">${fmt(val)}</span></div>`).join('')}
            </div>
          ` : ''}
          ${trendRows.length ? `
            <div style="margin-bottom:12px">
              <div style="font-size:10px;font-weight:600;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px">Trend</div>
              ${trendRows.map(([label, val, fmt]) => `<div class="data-row"><span class="label">${label}</span><span class="value">${fmt(val)}</span></div>`).join('')}
            </div>
          ` : ''}
        </div>
      </div>
    `;
  }

  // ── D. Short Interest Card ────────────────────────────────

  function renderShortInterest(f) {
    if (f.short_interest == null && f.days_to_cover == null && f.short_shares == null) return '';
    return `
      <div class="card mb-16">
        <div class="card-header">Short Interest</div>
        <div class="card-body">
          <div style="display:flex;gap:32px;flex-wrap:wrap">
            ${f.short_interest != null ? `
              <div>
                <div class="stat-label">Short Interest</div>
                <div class="stat-value" style="font-size:20px">${fmtPct(f.short_interest)}</div>
              </div>` : ''}
            ${f.days_to_cover != null ? `
              <div>
                <div class="stat-label">Days to Cover</div>
                <div class="stat-value" style="font-size:20px">${fmtNum(f.days_to_cover, 1)}</div>
              </div>` : ''}
            ${f.short_shares != null ? `
              <div>
                <div class="stat-label">Short Shares</div>
                <div class="stat-value" style="font-size:20px">${fmtInt(f.short_shares)}</div>
              </div>` : ''}
          </div>
        </div>
      </div>
    `;
  }

  // ── E. Dividends Table ────────────────────────────────────

  function renderDividends(divs) {
    if (!divs || !divs.length) return '';
    return `
      <div class="card mb-16">
        <div class="card-header">Dividends</div>
        <div class="card-body" style="padding:0">
          <table class="data-table">
            <thead><tr>
              <th>Ex-Date</th><th>Pay Date</th><th class="text-right">Amount</th><th>Frequency</th>
            </tr></thead>
            <tbody>
              ${divs.map(d => `<tr>
                <td>${fmtDate(d.ex_date)}</td>
                <td>${fmtDate(d.pay_date)}</td>
                <td class="text-right">${fmtPrice(d.amount)}</td>
                <td>${d.frequency || '\u2014'}</td>
              </tr>`).join('')}
            </tbody>
          </table>
        </div>
      </div>
    `;
  }

  // ── F. News Feed ──────────────────────────────────────────

  function renderNews(news, f) {
    // Also check for news_headlines in fundamentals
    const headlines = (f && f.news_headlines) || [];
    const allNews = (news && news.length) ? news : headlines.map(h => ({
      title: h.title || h,
      published: h.published || h.published_utc,
      url: h.url || h.article_url,
      publisher: h.publisher || h.publisher_name,
    }));

    if (!allNews.length) return '';

    return `
      <div class="card mb-16">
        <div class="card-header">Recent News</div>
        <div class="card-body">
          ${allNews.map(n => `
            <div class="news-item">
              <a class="headline" href="${n.url || '#'}" target="_blank" rel="noopener">${escapeHtml(n.title || 'Untitled')}</a>
              <div class="meta">
                ${n.publisher ? `<span>${escapeHtml(n.publisher)}</span>` : ''}
                ${n.published ? `<span>${timeAgo(n.published)}</span>` : ''}
              </div>
            </div>
          `).join('')}
        </div>
      </div>
    `;
  }

  // ── G. Trade History ──────────────────────────────────────

  function renderTradeHistory(trades) {
    if (!trades || !trades.length) return '';
    return `
      <div class="card mb-16">
        <div class="card-header">Trade History <span class="right" style="color:var(--text-secondary);font-weight:400;text-transform:none">${trades.length} trade${trades.length !== 1 ? 's' : ''}</span></div>
        <div class="card-body" style="padding:0;overflow-x:auto">
          <table class="data-table">
            <thead><tr>
              <th>Strategy</th><th>Direction</th><th>Shares</th>
              <th class="text-right">Entry</th><th class="text-right">Exit</th>
              <th class="text-right">P&amp;L</th><th>Status</th><th>Opened</th>
            </tr></thead>
            <tbody>
              ${trades.map(t => {
                const pnl = t.pnl_dollars;
                return `<tr>
                  <td>${stratDot(t.strategy_name)} ${t.strategy_name}</td>
                  <td><span class="pill ${t.direction === 'LONG' ? 'pill-positive' : 'pill-negative'}">${t.direction}</span></td>
                  <td>${fmtInt(t.shares)}</td>
                  <td class="text-right">${fmtPrice(t.entry_price)}</td>
                  <td class="text-right">${fmtPrice(t.exit_price)}</td>
                  <td class="text-right ${pnlClass(pnl)}">${fmtPnl(pnl)}</td>
                  <td><span class="pill ${t.status === 'OPEN' ? 'pill-accent' : 'pill-muted'}">${t.status}</span></td>
                  <td>${fmtTime(t.entry_time)}</td>
                </tr>`;
              }).join('')}
            </tbody>
          </table>
        </div>
      </div>
    `;
  }

  // ── H. Related Tickers ────────────────────────────────────

  function renderRelated(tickers) {
    if (!tickers || !tickers.length) return '';
    return `
      <div class="card mb-16">
        <div class="card-header">Related Tickers</div>
        <div class="card-body">
          <div class="chip-row">
            ${tickers.map(t => `<span class="chip" data-ticker="${t}">${t}</span>`).join('')}
          </div>
        </div>
      </div>
    `;
  }

  // ── Utilities ─────────────────────────────────────────────

  function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  // ── Event Delegation (related ticker clicks) ──────────────

  profileEl.addEventListener('click', (e) => {
    const chip = e.target.closest('.chip[data-ticker]');
    if (chip) {
      loadTicker(chip.dataset.ticker);
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  });

  // ── URL Parameter Support ─────────────────────────────────

  const params = new URLSearchParams(window.location.search);
  const tickerParam = params.get('ticker');
  if (tickerParam) {
    loadTicker(tickerParam);
  }

})();

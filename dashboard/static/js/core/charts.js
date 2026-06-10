/* Chart factory + interactivity for the trading-terminal feel.
   Wraps the vendored lightweight-charts v4.1 global:
   - themed options from tokens, restyled live on theme toggle
   - pan/zoom/kinetic scroll fully enabled (mouse + touch)
   - pane sync: shared visible range + crosshair across price/RSI/MACD
   - range switcher chips, fullscreen, double-click fit
   - one shared ResizeObserver for autosizing */

import { chartColors, onThemeChange } from './theme.js';

const LWC = window.LightweightCharts;

const registry = new Set(); // every chart, for theme restyle
const ro = new ResizeObserver(entries => {
  for (const e of entries) {
    const chart = e.target.__efChart;
    if (chart) chart.applyOptions({ width: Math.floor(e.contentRect.width) });
  }
});

onThemeChange(() => {
  for (const { chart } of registry) chart.applyOptions(baseOptions());
});

function baseOptions() {
  const c = chartColors();
  return {
    layout: {
      background: { type: 'solid', color: 'transparent' },
      textColor: c.text,
      fontSize: 11,
    },
    grid: {
      vertLines: { color: c.grid, style: LWC.LineStyle.Dotted },
      horzLines: { color: c.grid, style: LWC.LineStyle.Dotted },
    },
    crosshair: {
      mode: LWC.CrosshairMode.Normal,
      vertLine: { color: c.text, width: 1, style: LWC.LineStyle.Dashed, labelBackgroundColor: c.border },
      horzLine: { color: c.text, width: 1, style: LWC.LineStyle.Dashed, labelBackgroundColor: c.border },
    },
    rightPriceScale: { borderColor: c.border },
    timeScale: {
      borderColor: c.border,
      rightOffset: 5,
      barSpacing: 6,
      timeVisible: false,
    },
    handleScroll: { mouseWheel: true, pressedMouseMove: true, horzTouchDrag: true, vertTouchDrag: false },
    handleScale: { mouseWheel: true, pinch: true, axisPressedMouseMove: true, axisDoubleClickReset: true },
  };
}

export function createChart(el, { height } = {}) {
  const chart = LWC.createChart(el, {
    ...baseOptions(),
    width: el.clientWidth || 300,
    height: height || el.clientHeight || 300,
  });
  el.__efChart = chart;
  ro.observe(el);
  const entry = { chart, el };
  registry.add(entry);
  el.addEventListener('dblclick', () => chart.timeScale().fitContent());
  chart.__efDestroy = () => {
    ro.unobserve(el);
    registry.delete(entry);
    chart.remove();
  };
  return chart;
}

export function colors() {
  return chartColors();
}

/* Lock N panes together: shared visible logical range + crosshair.
   The Bloomberg/TradingView multi-pane behavior. */
export function syncPanes(panes) {
  // panes: [{chart, series}] — series is each pane's primary series
  let syncing = false;
  for (const p of panes) {
    p.chart.timeScale().subscribeVisibleLogicalRangeChange(range => {
      if (syncing || !range) return;
      syncing = true;
      for (const q of panes) {
        if (q !== p) q.chart.timeScale().setVisibleLogicalRange(range);
      }
      syncing = false;
    });
    p.chart.subscribeCrosshairMove(param => {
      if (syncing) return;
      syncing = true;
      for (const q of panes) {
        if (q === p) continue;
        if (param.time != null) {
          const val = param.seriesData && param.seriesData.get(p.series);
          const price = val && (val.close ?? val.value);
          q.chart.setCrosshairPosition(price ?? 0, param.time, q.series);
        } else {
          q.chart.clearCrosshairPosition();
        }
      }
      syncing = false;
    });
  }
}

/* Range chips. ranges: [{label:'3M', value:'3m'}, ...]. onSelect(value)
   is responsible for fetching/setting data when needed. */
export function rangeSwitcher(el, ranges, initial, onSelect) {
  el.classList.add('c-chips', 'scroll');
  const buttons = ranges.map(r => {
    const b = document.createElement('button');
    b.className = 'c-chip' + (r.value === initial ? ' active' : '');
    b.textContent = r.label;
    b.addEventListener('click', () => {
      for (const x of buttons) x.classList.remove('active');
      b.classList.add('active');
      onSelect(r.value);
    });
    return b;
  });
  el.replaceChildren(...buttons);
}

export function fullscreenButton(btnEl, shellEl) {
  btnEl.addEventListener('click', () => {
    if (document.fullscreenElement) document.exitFullscreen();
    else shellEl.requestFullscreen?.();
  });
}

/* Marker builders (set on the candle/line series). */
export function tradeMarkers(trades, c = chartColors()) {
  const out = [];
  for (const t of trades) {
    if (t.entry_epoch != null) {
      out.push({
        time: t.entry_epoch,
        position: 'belowBar',
        shape: 'arrowUp',
        color: c.accent,
        text: `B ${t.strategy_name ? t.strategy_name[0].toUpperCase() : ''}`,
        id: `entry:${t.trade_id}`,
      });
    }
    if (t.exit_epoch != null) {
      out.push({
        time: t.exit_epoch,
        position: 'aboveBar',
        shape: 'arrowDown',
        color: (t.pnl_dollars ?? 0) >= 0 ? c.up : c.down,
        text: 'S',
        id: `exit:${t.trade_id}`,
      });
    }
  }
  return out.sort((a, b) => a.time - b.time);
}

export function eventMarkers(events, kinds, c = chartColors()) {
  const out = [];
  if (kinds.has('dividends')) {
    for (const d of events.dividends || []) {
      out.push({ time: d.time, position: 'belowBar', shape: 'circle', color: c.info, text: 'D', id: `div:${d.time}` });
    }
  }
  if (kinds.has('splits')) {
    for (const s of events.splits || []) {
      out.push({ time: s.time, position: 'belowBar', shape: 'square', color: c.warn, text: `S ${s.ratio || ''}`, id: `split:${s.time}` });
    }
  }
  if (kinds.has('news')) {
    for (const n of events.news || []) {
      out.push({ time: n.time, position: 'aboveBar', shape: 'circle', color: c.text, text: '', id: `news:${n.time}` });
    }
  }
  return out.sort((a, b) => a.time - b.time);
}

export function mergeMarkers(...lists) {
  return lists.flat().sort((a, b) => a.time - b.time);
}

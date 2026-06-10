/* Calendar P&L heatmap — weeks computed from the ACTUAL date range
   (fixes the old hardcoded 91-day/13-week grid). Pure DOM, classes only;
   intensity via data-lvl attributes mapped in components.css. */

import { h } from '../core/dom.js';
import { fmtPnl } from '../core/fmt.js';

function isoDay(d) {
  return d.toISOString().slice(0, 10);
}

export function calendarHeatmap(dailyPnl, { weeks } = {}) {
  // dailyPnl: { 'YYYY-MM-DD': number }
  const dates = Object.keys(dailyPnl).sort();
  const end = new Date();
  end.setUTCHours(0, 0, 0, 0);
  const start = dates.length
    ? new Date(dates[0] + 'T00:00:00Z')
    : new Date(end.getTime() - 90 * 86400e3);
  // snap start back to Monday
  const startDow = (start.getUTCDay() + 6) % 7;
  start.setUTCDate(start.getUTCDate() - startDow);

  const nWeeks = weeks
    || Math.max(1, Math.ceil(((end - start) / 86400e3 + 1) / 7));

  const magnitudes = Object.values(dailyPnl).map(Math.abs).filter(v => v > 0).sort((a, b) => a - b);
  const p80 = magnitudes.length ? magnitudes[Math.floor(magnitudes.length * 0.8)] : 1;

  const lvl = (v) => {
    if (!v) return null;
    const m = Math.abs(v) / (p80 || 1);
    const band = m >= 1 ? 3 : m >= 0.4 ? 2 : 1;
    return (v > 0 ? 'p' : 'n') + band;
  };

  const grid = h('div', { class: 'c-heat' });
  const cursor = new Date(start);
  for (let w = 0; w < nWeeks; w++) {
    const col = h('div', { class: 'col' });
    for (let d = 0; d < 7; d++) {
      const key = isoDay(cursor);
      const v = dailyPnl[key];
      const cell = h('div', {
        class: 'cell',
        title: v != null ? `${key}: ${fmtPnl(v)}` : key,
      });
      const level = lvl(v);
      if (level) cell.dataset.lvl = level;
      if (cursor > end) cell.classList.add('hidden');
      col.append(cell);
      cursor.setUTCDate(cursor.getUTCDate() + 1);
    }
    grid.append(col);
  }
  const wrap = h('div', { class: 'scroll-x' }, grid);
  return wrap;
}

/* /trades — a plain human history of every fill and the profit it realized.
   Deliberately simple: no filters, no sorting, no polling. Load once, render
   a table, stop. The server does all the math (see /api/desk/trade-history);
   this file must never re-derive P&L — in particular the option x100
   multiplier is ALREADY inside `realized`.

   Era model (REBUILD-V4): era-2 rows are fills executed by the Alpaca paper
   broker (the current book of record); era-1 rows are the frozen
   pre-migration ledger, rendered in their own collapsible section with the
   old corp-action conventions intact. */

import { apiGet } from '../core/net.js';
import { h, clear, skeleton, renderEmpty, renderError } from '../core/dom.js';
import { fmtDollar, fmtPnl, upDownClass } from '../core/fmt.js';

const DASH = '—';

/* Buy/Sell for real fills; stops and the corp-action kinds name themselves. */
function actionCell(r) {
  if (r.kind === 'trade' || r.kind === 'expiry' || r.kind === 'stop') {
    const cls = r.side === 'BUY' ? 'up' : 'down';
    const label = r.kind === 'expiry' ? `${r.side} (expiry)`
      : r.kind === 'stop' ? `${r.side} (stop)` : r.side;
    return h('span', { class: 'c-pill ' + cls, text: label });
  }
  return h('span', { class: 'c-pill neutral', text: r.kind });
}

/* Split rows carry price 0 / dollars 0 — showing "$0.00" would read as a
   trade that moved no money rather than a share-count adjustment. */
function amountCell(r) {
  if (r.kind === 'split') return DASH;
  return fmtDollar(r.dollars);
}

/* null = closed nothing, so there is no profit to state. A literal 0.00 would
   render bright green via upDownClass(0) and claim a breakeven win. A genuine
   breakeven close is shown neutral for the same reason. */
function profitCell(r) {
  if (r.realized == null) return h('td', { class: 'num t-dim', text: DASH });
  const cls = Math.abs(r.realized) < 0.005 ? 't-dim' : upDownClass(r.realized);
  return h('td', { class: 'num ' + cls, text: fmtPnl(r.realized) });
}

function row(r) {
  return h('tr', {},
    h('td', { text: r.date || DASH }),
    h('td', {}, h('a', { href: '/symbol/' + r.underlying, text: r.label })),
    h('td', {}, actionCell(r)),
    h('td', { class: 'num', text: amountCell(r) }),
    profitCell(r));
}

function table(rows) {
  const head = h('thead', {}, h('tr', {},
    h('th', { text: 'Date' }),
    h('th', { text: 'Symbol' }),
    h('th', { text: 'Action' }),
    h('th', { class: 'num', text: 'Amount' }),
    h('th', { class: 'num', text: 'Profit' })));
  // .c-table-wrap is required: body{overflow-x:hidden} clips a wide table on
  // mobile instead of scrolling it.
  return h('div', { class: 'c-table-wrap' },
    h('table', { class: 'c-table' }, head, h('tbody', {}, ...rows.map(row))));
}

async function load() {
  const el = document.getElementById('trades-table');
  if (!el) return;
  skeleton(el);
  try {
    const data = await apiGet('/api/desk/trade-history?limit=500');
    const rows = data.rows || [];
    if (!rows.length) { renderEmpty(el, 'No trades yet.'); return; }
    clear(el);

    const era2 = rows.filter(r => r.era !== 1);
    const era1 = rows.filter(r => r.era === 1);

    if (era2.length) {
      el.append(table(era2));
    } else {
      renderEmpty(el, era1.length
        ? 'No trades on the current book yet — the frozen Era-1 history is below.'
        : 'No trades yet.');
    }

    if (era1.length) {
      // The frozen pre-migration ledger, collapsed by default — history,
      // not the working book. When it is the ONLY history (era 2 has no
      // fills yet), open it: an empty page hiding 100+ real fills behind
      // a closed disclosure reads as "history lost".
      const era1Attrs = { class: 'trades-era1' };
      if (!era2.length) era1Attrs.open = '';
      const details = h('details', era1Attrs,
        h('summary', {},
          h('span', { class: 'c-pill neutral', text: 'ERA 1' }),
          ' The pre-migration book (frozen at cutover) — ' + era1.length
            + ' rows, realized ' + fmtPnl(data.era1_realized || 0)),
        table(era1));
      el.append(details);
    }

    const total = document.getElementById('trades-total');
    if (total) {
      // The totals cover the WHOLE ledger, so say so when the table does not.
      const shown = data.total > rows.length
        ? ` — showing the newest ${rows.length} of ${data.total} fills`
        : '';
      const eras = era1.length
        ? ` (era 2 ${fmtPnl(data.era2_realized || 0)} · era 1 ${fmtPnl(data.era1_realized || 0)})`
        : '';
      total.textContent =
        `Realized profit ${fmtPnl(data.realized_pnl)} across `
        + `${data.closing_fills} closing fills${eras}${shown}`;
    }
  } catch (err) {
    renderError(el, err, load);
  }
}

load();

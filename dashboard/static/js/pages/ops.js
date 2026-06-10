/* Ops & Health — heartbeat grid, scheduler next-runs, two-tier storage
   panel (DB hot set vs R2 permanent asset), observations, and the merged
   agent activity timeline. Auto-refreshes via the bounded poller. */

import { apiGet } from '../core/net.js';
import { fmtInt, fmtDate, timeAgo } from '../core/fmt.js';
import { h, clear, renderEmpty, panel } from '../core/dom.js';
import { poller } from '../core/poll.js';

async function loadHealth() {
  const hbEl = document.getElementById('ops-heartbeats');
  const obsEl = document.getElementById('ops-observations');
  const schedEl = document.getElementById('ops-scheduler');
  await panel(hbEl, () => apiGet('/api/ops/health'), (host, ops) => {
    // heartbeats
    const hbs = ops.heartbeats || [];
    if (!hbs.length) renderEmpty(host, 'No heartbeats recorded');
    else {
      const wrap = h('div', { class: 'flex-col gap-4' });
      for (const hb of hbs) {
        wrap.append(h('div', { class: 'c-tl' },
          h('div', { class: 'when', text: hb.age_minutes != null ? `${Math.round(hb.age_minutes)}m ago` : '—' }),
          h('div', { class: 'what flex items-center gap-8' },
            h('span', { class: `c-pill ${hb.ok ? 'up' : 'down'}`, text: hb.ok ? 'OK' : 'ERR' }),
            h('span', { class: 'num', text: hb.component }))));
      }
      clear(host).append(wrap);
    }
    // observations
    const obs = ops.observations || [];
    if (!obs.length) renderEmpty(obsEl, 'No open observations — all clear');
    else {
      const wrap = h('div', { class: 'flex-col gap-4' });
      for (const o of obs) {
        wrap.append(h('div', { class: 'c-tl' },
          h('div', { class: 'when', text: timeAgo(o.timestamp) }),
          h('div', { class: 'what' },
            h('span', { class: `c-pill ${o.severity === 'CRITICAL' ? 'down' : 'warn'}`, text: o.severity }),
            h('span', { text: ` ${o.category}: ${o.message}` }))));
      }
      clear(obsEl).append(wrap);
    }
    // scheduler
    const sched = ops.scheduler || {};
    const jobs = sched.jobs || sched.next_runs || [];
    if (!jobs.length) renderEmpty(schedEl, sched.running === false ? 'Scheduler stopped' : 'No scheduled jobs visible');
    else {
      const wrap = h('div', { class: 'flex-col gap-4' });
      for (const j of jobs) {
        wrap.append(h('div', { class: 'c-tl' },
          h('div', { class: 'when', text: (j.next_run || '').slice(5, 16) || '—' }),
          h('div', { class: 'what num', text: j.id || j.name || String(j) })));
      }
      clear(schedEl).append(wrap);
    }
  });
}

async function loadStorage() {
  const el = document.getElementById('ops-storage');
  await panel(el, () => apiGet('/api/ops/storage'), (host, st) => {
    const kv = h('dl', { class: 'c-kv' });
    kv.append(h('dt', { text: 'DB (hot set)' }),
      h('dd', { text: `${fmtInt(st.db.symbols)} symbols · ${fmtInt(st.db.rows)} rows · ${st.db.min_date || '?'} → ${st.db.max_date || '?'}` }));
    if (st.r2) {
      kv.append(h('dt', { text: 'R2 (permanent)' }),
        h('dd', { text: `${fmtInt(st.r2.symbols)} symbols · ${fmtInt(st.r2.rows)} rows · grow-only, through ${st.r2.max_date || '?'}` }));
    } else {
      kv.append(h('dt', { text: 'R2' }), h('dd', { text: 'not configured in this environment' }));
    }
    clear(host).append(kv,
      h('div', { class: 't-dim mt-8', text:
        'DB keeps protected ETFs full-history + trailing-365d top-1000; the full asset lives in R2 (nightly merge-sync, fingerprint-guarded prune).' }));
  });
}

async function loadActivity() {
  const el = document.getElementById('ops-activity');
  const inc = document.getElementById('ops-resolved').checked;
  await panel(el, () => apiGet(`/api/ops/activity?limit=80&include_resolved=${inc}`), (host, data) => {
    const items = data.items || [];
    if (!items.length) { renderEmpty(host, 'No agent activity'); return; }
    const wrap = h('div', { class: 'flex-col gap-4' });
    for (const it of items) {
      const body = it.kind === 'observation'
        ? h('div', { class: 'what' },
            h('span', { class: `c-pill ${it.severity === 'CRITICAL' ? 'down' : it.resolved_at ? 'neutral' : 'warn'}`,
                        text: it.resolved_at ? `${it.severity} ✓` : it.severity }),
            h('span', { text: ` ${it.agent} · ${it.category}: ${it.message}` }))
        : h('div', { class: 'what' },
            h('span', { class: `c-pill ${it.status === 'merged' ? 'up' : 'info'}`, text: it.action_type }),
            h('span', { text: ` ${it.agent}: ${it.summary}` }),
            it.pr_url ? h('a', { href: it.pr_url, target: '_blank', rel: 'noopener', text: ' PR ↗' }) : null);
      wrap.append(h('div', { class: 'c-tl' },
        h('div', { class: 'when', text: fmtDate(it.timestamp) }), body));
    }
    clear(host).append(wrap);
  });
}

async function refresh() {
  await Promise.allSettled([loadHealth(), loadStorage(), loadActivity()]);
}

async function init() {
  try {
    const hcheck = await apiGet('/api/health');
    document.getElementById('ops-version').textContent = `v${hcheck.version} · plan: ${hcheck.data_plan || '—'}`;
  } catch { /* header optional */ }
  document.getElementById('ops-resolved').addEventListener('change', loadActivity);
  await refresh();
  poller(refresh, { intervalMs: 60000, maxFailures: 10 });
}

document.addEventListener('DOMContentLoaded', init);

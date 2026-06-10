/* Shell behavior — the whole navigation layer (common.js retired):
   top-nav + tab-bar active state, indices strip, health dot, theme
   toggle, the mobile "More" bottom sheet. */

import { apiGet } from './net.js';
import { toggleTheme, currentTheme } from './theme.js';

function pathMatches(href) {
  const path = window.location.pathname;
  if (href === '/') return path === '/';
  return path === href || path.startsWith(href + '/');
}

function initTabbar() {
  const items = document.querySelectorAll('.tabbar-item[href]');
  let any = false;
  for (const it of items) {
    if (pathMatches(it.getAttribute('href'))) {
      it.classList.add('active');
      any = true;
    }
  }
  // "More" lights up when the active page lives inside the sheet
  if (!any) {
    for (const link of document.querySelectorAll('.sheet-link[href]')) {
      if (pathMatches(link.getAttribute('href'))) {
        link.classList.add('active');
        document.getElementById('tabbar-more')?.classList.add('active');
      }
    }
  }
}

function initSheet() {
  const moreBtn = document.getElementById('tabbar-more');
  const backdrop = document.querySelector('.sheet-backdrop');
  if (!moreBtn || !backdrop) return;
  const toggle = (open) => document.body.classList.toggle('sheet-open', open);
  moreBtn.addEventListener('click', () => toggle(!document.body.classList.contains('sheet-open')));
  backdrop.addEventListener('click', () => toggle(false));
  document.addEventListener('keydown', e => { if (e.key === 'Escape') toggle(false); });
}

function initTopnav() {
  const path = window.location.pathname;
  for (const tab of document.querySelectorAll('.topnav-tab')) {
    if (pathMatches(tab.getAttribute('href'))) tab.classList.add('active');
  }
}

function initThemeButtons() {
  const setIcon = () => {
    const btn = document.getElementById('theme-toggle-btn');
    if (btn) btn.textContent = currentTheme() === 'light' ? '☀️' : '🌙';
  };
  setIcon();
  for (const el of document.querySelectorAll('[data-action="toggle-theme"]')) {
    el.addEventListener('click', () => { toggleTheme(); setIcon(); });
  }
}

async function loadIndices() {
  try {
    const data = await apiGet('/api/benchmarks/comparison?days=5');
    const el = document.getElementById('topnav-indices');
    if (!el || !data.indices) return;
    el.replaceChildren();
    for (const sym of ['SPY', 'QQQ', 'VIX']) {
      const series = data.indices[sym];
      if (!series || series.length < 2) continue;
      const chg = series[series.length - 1] - series[series.length - 2];
      const wrap = document.createElement('div');
      wrap.className = 'topnav-index';
      const s1 = document.createElement('span');
      s1.className = 'sym';
      s1.textContent = sym;
      const s2 = document.createElement('span');
      s2.className = 'num ' + (chg >= 0 ? 't-up' : 't-down');
      s2.textContent = `${chg >= 0 ? '▲' : '▼'}${Math.abs(chg).toFixed(2)}%`;
      wrap.append(s1, s2);
      el.append(wrap);
    }
  } catch { /* decorative */ }
}

async function loadHealthDot() {
  try {
    const data = await apiGet('/api/health');
    const el = document.getElementById('topnav-status');
    if (!el) return;
    el.replaceChildren();
    const dot = document.createElement('span');
    dot.className = 'dot ' + (data.status === 'ok' ? 'ok' : 'bad');
    el.append(dot, document.createTextNode(` v${data.version}`));
  } catch { /* decorative */ }
}

document.addEventListener('DOMContentLoaded', () => {
  initTopnav();
  initTabbar();
  initSheet();
  initThemeButtons();
  loadIndices();
  loadHealthDot();
});

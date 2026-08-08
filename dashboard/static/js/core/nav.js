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

/* The index strip that used to live here (#topnav-indices) fetched a
   workbench-era route that no longer exists; the dead slot was removed from
   base.html in the V4 dashboard redesign. The desk hero's live index chips
   (SSE tape) are the replacement. */

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
  loadHealthDot();
});

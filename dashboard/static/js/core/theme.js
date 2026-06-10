/* Theme state + token access. The attribute lives on <html> as
   data-theme; during the transition common.js also mirrors the legacy
   body.light-mode class so old pages keep their light styling. Charts
   read live token values here and restyle on toggle. */

const KEY = 'ef-theme';
const listeners = new Set();

export function currentTheme() {
  return localStorage.getItem(KEY) === 'light' ? 'light' : 'dark';
}

export function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  document.body.classList.toggle('light-mode', theme === 'light'); // legacy bridge
  localStorage.setItem(KEY, theme);
  for (const cb of listeners) cb(theme);
}

export function toggleTheme() {
  applyTheme(currentTheme() === 'light' ? 'dark' : 'light');
}

export function onThemeChange(cb) {
  listeners.add(cb);
  return () => listeners.delete(cb);
}

function tok(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

/* Token values for chart options — resolved at call time so a theme
   toggle followed by applyOptions() restyles charts live. */
export function chartColors() {
  return {
    bg: tok('--t-bg'),
    surface: tok('--t-surface'),
    border: tok('--t-border'),
    grid: tok('--t-border'),
    text: tok('--t-text-2'),
    up: tok('--t-up'),
    down: tok('--t-down'),
    warn: tok('--t-warn'),
    info: tok('--t-info'),
    accent: tok('--t-accent'),
    benchmark: tok('--t-benchmark'),
    series: [0, 1, 2, 3, 4, 5, 6, 7].map(i => tok(`--t-s${i}`)),
  };
}

export function seriesColor(slot) {
  return tok(`--t-s${((slot % 8) + 8) % 8}`);
}

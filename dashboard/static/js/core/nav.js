/* Shell behavior for the redesigned navigation: bottom tab bar active
   state + the mobile "More" bottom sheet. Top-nav active state, the
   indices strip and the health dot stay in common.js until the cleanup
   phase deletes it (then this module absorbs them). */

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

document.addEventListener('DOMContentLoaded', () => {
  initTabbar();
  initSheet();
});

/* DOM helpers — element builder, standard loading / error / empty states.
   Every async panel renders through these three states; classes only
   (no inline styles — theming stays intact). */

export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null) continue;
    if (k === 'class') el.className = v;
    else if (k === 'dataset') Object.assign(el.dataset, v);
    else if (k.startsWith('on') && typeof v === 'function') {
      el.addEventListener(k.slice(2), v);
    } else if (k === 'text') el.textContent = v;
    else el.setAttribute(k, v);
  }
  for (const c of children.flat()) {
    if (c == null) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

export function clear(el) {
  while (el.firstChild) el.removeChild(el.firstChild);
  return el;
}

export function skeleton(el, kind = 'block') {
  clear(el).append(h('div', { class: `c-skel ${kind}` }));
}

export function renderEmpty(el, message = 'No data') {
  clear(el).append(h('div', { class: 'c-empty', text: message }));
}

export function renderError(el, err, retryFn) {
  const detail = err && err.status
    ? `${err.path || ''} → ${err.status}`
    : (err && err.message) || 'network error';
  const card = h('div', { class: 'c-error' },
    h('div', { class: 'icon', text: '⚠' }),
    h('div', { text: 'Failed to load' }),
    h('div', { class: 'detail', text: detail }),
  );
  if (retryFn) {
    card.append(h('button', {
      class: 'c-btn ghost',
      text: 'Retry',
      onclick: () => retryFn(),
    }));
  }
  clear(el).append(card);
}

/* Standard async panel runner: skeleton -> loader() -> render(data),
   error card with retry on failure. Returns the loader promise. */
export function panel(el, loader, render, { skeletonKind = 'block', empty } = {}) {
  const go = async () => {
    skeleton(el, skeletonKind);
    try {
      const data = await loader();
      if (empty && empty(data)) renderEmpty(el, typeof empty(data) === 'string' ? empty(data) : 'No data');
      else render(el, data);
    } catch (err) {
      console.error(err);
      renderError(el, err, go);
    }
  };
  return go();
}

export function debounce(fn, ms = 250) {
  let timer = null;
  const wrapped = (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
  wrapped.cancel = () => clearTimeout(timer);
  return wrapped;
}

/* Squarified treemap rendered as positioned divs — replaces the only
   Chart.js usage (screener sector rotation). Nodes: {name, value,
   quadrant, count}; onClick(name) wires sector filtering. Positions are
   set via CSSOM (style properties on elements created here), keeping
   templates inline-style-free. */

export function treemap(el, nodes, { height = 220, onClick } = {}) {
  el.classList.add('c-treemap');
  el.replaceChildren();
  const width = el.clientWidth || 600;
  el.style.height = `${height}px`;

  const items = nodes.filter(n => n.value > 0)
    .sort((a, b) => b.value - a.value);
  const total = items.reduce((s, n) => s + n.value, 0);
  if (!total) return;

  // squarify: lay rows along the shorter side
  let x = 0, y = 0, w = width, hgt = height;
  let row = [], rowValue = 0;
  const scale = (width * height) / total;

  const worst = (row, rv, side) => {
    const s2 = (rv * scale) ** 2;
    let max = 0;
    for (const n of row) {
      const a = n.value * scale;
      const r = Math.max((side * side * a) / s2, s2 / (side * side * a));
      if (r > max) max = r;
    }
    return max;
  };

  const layoutRow = (row, rv) => {
    const horizontal = w >= hgt;
    const side = horizontal ? hgt : w;
    const thickness = (rv * scale) / side;
    let offset = 0;
    for (const n of row) {
      const len = (n.value * scale) / thickness;
      const node = document.createElement('div');
      node.className = `node q-${(n.quadrant || 'unknown').toLowerCase()}`;
      node.style.left = `${horizontal ? x : x + offset}px`;
      node.style.top = `${horizontal ? y + offset : y}px`;
      node.style.width = `${Math.max(0, (horizontal ? thickness : len) - 1)}px`;
      node.style.height = `${Math.max(0, (horizontal ? len : thickness) - 1)}px`;
      node.title = `${n.name} — ${n.count ?? n.value}`;
      const nm = document.createElement('div');
      nm.className = 'nm';
      nm.textContent = n.name;
      node.append(nm);
      if (n.count != null) {
        const ct = document.createElement('div');
        ct.className = 'ct';
        ct.textContent = String(n.count);
        node.append(ct);
      }
      if (onClick) node.addEventListener('click', () => onClick(n.name));
      el.append(node);
      offset += len;
    }
    if (horizontal) { x += thickness; w -= thickness; }
    else { y += thickness; hgt -= thickness; }
  };

  for (const n of items) {
    const side = Math.min(w, hgt);
    if (row.length && worst([...row, n], rowValue + n.value, side) >
        worst(row, rowValue, side)) {
      layoutRow(row, rowValue);
      row = [n]; rowValue = n.value;
    } else {
      row.push(n); rowValue += n.value;
    }
  }
  if (row.length) layoutRow(row, rowValue);
}

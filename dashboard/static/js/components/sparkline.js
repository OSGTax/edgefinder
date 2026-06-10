/* Tiny canvas sparkline for table cells / position cards. Display size
   is fixed by the .c-sparkline CSS class (90x24; .lg = 140x36); the
   bitmap is dpr-scaled for crisp retina rendering. */

const SIZES = { base: [90, 24], lg: [140, 36] };

export function sparkline(values, { size = 'base', color, baseline } = {}) {
  const [width, height] = SIZES[size] || SIZES.base;
  const canvas = document.createElement('canvas');
  canvas.className = 'c-sparkline' + (size === 'lg' ? ' lg' : '');
  const dpr = window.devicePixelRatio || 1;
  canvas.width = width * dpr;
  canvas.height = height * dpr;

  const pts = (values || []).filter(v => v != null);
  if (pts.length < 2) return canvas;

  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  const min = Math.min(...pts, baseline ?? Infinity);
  const max = Math.max(...pts, baseline ?? -Infinity);
  const span = max - min || 1;
  const x = i => (i / (pts.length - 1)) * (width - 2) + 1;
  const y = v => height - 2 - ((v - min) / span) * (height - 4);

  const css = (name) =>
    getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const stroke = color || css(pts[pts.length - 1] >= pts[0] ? '--t-up' : '--t-down');

  if (baseline != null) {
    ctx.strokeStyle = css('--t-border-strong');
    ctx.setLineDash([2, 3]);
    ctx.beginPath();
    ctx.moveTo(0, y(baseline));
    ctx.lineTo(width, y(baseline));
    ctx.stroke();
    ctx.setLineDash([]);
  }

  ctx.strokeStyle = stroke;
  ctx.lineWidth = 1.4;
  ctx.lineJoin = 'round';
  ctx.beginPath();
  pts.forEach((v, i) => (i === 0 ? ctx.moveTo(x(i), y(v)) : ctx.lineTo(x(i), y(v))));
  ctx.stroke();
  return canvas;
}

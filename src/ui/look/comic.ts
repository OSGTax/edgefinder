import { h } from '../dom';
import { letteringParts, type LetterOpts } from './letters';
import { hashStr, n2, rng, sym } from './rand';

// Comic-book pop-ups for the big moments: a starburst (or a jagged blast, or
// a puffy balloon) with a thick ink outline, an offset shadow, halftone dots
// and speed lines, and the word in fat slanted comic lettering. The shape is
// seeded from the word, so "SPLOOSH!" always gets its own burst. The element
// scales in with an overshoot, shakes a little and is gone in about a second.
//
// Which word, when, and the no-repeat rule belong to the game screen; this
// file draws (and offers a default pop animation, `animate: false` to skip it).

export type BurstShape = 'star' | 'jagged' | 'cloud';

export interface PopOpts {
  /** burst outline: star (default), jagged blast, or puffy cloud balloon */
  shape?: BurstShape;
  /** same as `shape` */
  burst?: BurstShape;
  /** shorthand for [color, color2, textColor] */
  colors?: [string, string?, string?];
  /** play the built-in pop animation (default true); false gives a still element to animate yourself */
  animate?: boolean;
  /** outer fill (default sunshine yellow) */
  color?: string;
  /** inner burst fill (default a paler yellow); 'none' for a single layer */
  color2?: string;
  /** letter fill (default tomato red) */
  textColor?: string;
  /** outline and shadow colour (default the warm ink) */
  ink?: string;
  /** width in CSS px (default 280) */
  width?: number;
  /** degrees of tilt (default: a few, seeded) */
  tilt?: number;
  /** speed lines around the burst (default true) */
  lines?: boolean;
  /** halftone dots (default true) */
  dots?: boolean;
  /** remove the element after this many ms (default: when the animation ends; 0 keeps it) */
  ms?: number;
  /** a small caption under the burst (e.g. who did it) */
  sub?: string;
  seed?: string;
  /** extra lettering options */
  letters?: LetterOpts;
}

let uid = 0;

interface Pt { x: number; y: number }

function burstPath(shape: BurstShape, rx: number, ry: number, r: () => number): Pt[] | string {
  if (shape === 'cloud') {
    const n = 11;
    const pts: Pt[] = [];
    for (let i = 0; i < n; i++) {
      const a = (i / n) * Math.PI * 2 + sym(r) * 0.08;
      pts.push({ x: Math.cos(a) * rx * 0.86, y: Math.sin(a) * ry * 0.84 });
    }
    // a puff between each pair of points
    let d = `M${n2(pts[0].x)} ${n2(pts[0].y)}`;
    for (let i = 0; i < n; i++) {
      const p = pts[i], q = pts[(i + 1) % n];
      const mx = (p.x + q.x) / 2, my = (p.y + q.y) / 2;
      const len = Math.hypot(mx, my) || 1;
      const bulge = 0.28 + r() * 0.12;
      const cx = mx + (mx / len) * Math.hypot(q.x - p.x, q.y - p.y) * bulge * 1.6;
      const cy = my + (my / len) * Math.hypot(q.x - p.x, q.y - p.y) * bulge * 1.6;
      d += ` Q${n2(cx)} ${n2(cy)} ${n2(q.x)} ${n2(q.y)}`;
    }
    return `${d} Z`;
  }
  const n = shape === 'star' ? 14 + Math.floor(r() * 4) : 10 + Math.floor(r() * 3);
  const pts: Pt[] = [];
  for (let i = 0; i < n * 2; i++) {
    const outer = i % 2 === 0;
    const a = (i / (n * 2)) * Math.PI * 2 + (shape === 'jagged' ? sym(r) * 0.12 : sym(r) * 0.04);
    const k = outer
      ? (shape === 'star' ? 1 + sym(r) * 0.06 : 0.92 + r() * 0.24)
      : (shape === 'star' ? 0.74 + sym(r) * 0.03 : 0.6 + r() * 0.1);
    pts.push({ x: Math.cos(a) * rx * k, y: Math.sin(a) * ry * k });
  }
  return pts;
}

const toD = (p: Pt[] | string, s = 1) => typeof p === 'string'
  ? p.replace(/-?\d*\.?\d+/g, (v) => n2(Number(v) * s))
  : `M${p.map((q) => `${n2(q.x * s)} ${n2(q.y * s)}`).join(' L')} Z`;

/** The burst as an SVG string (no animation). */
export function comicBurstSVG(text: string, opts: PopOpts = {}): string {
  const o: PopOpts = { ...opts, shape: opts.shape ?? opts.burst, color: opts.color ?? opts.colors?.[0], color2: opts.color2 ?? opts.colors?.[1], textColor: opts.textColor ?? opts.colors?.[2] };
  const id = `gslpop${++uid}`;
  const shape = o.shape ?? 'star';
  const r = rng(hashStr(`${o.seed ?? text}|${shape}`));
  const ink = o.ink ?? 'var(--marker, #2b1d14)';
  const c1 = o.color ?? 'var(--highlighter, #ffcf33)';
  const c2 = o.color2 ?? '#fff1b8';
  const L = letteringParts(text, { style: 'comic', color: o.textColor ?? 'var(--marker-red, #f25c3c)', ink, seed: o.seed ?? text, ...o.letters });
  // size the burst around the word: words are wide, so the burst is an ellipse
  const tw = 100, th = (L.vb[3] / L.vb[2]) * tw;
  const rx = tw * 0.5 / 0.62, ry = Math.max(th * 0.5 / 0.5, rx * 0.56);
  const shapeD = burstPath(shape, rx, ry, r);
  const pad = 26;
  const W = rx * 2 + pad * 2, H = ry * 2 + pad * 2;
  let lines = '';
  if (o.lines !== false) {
    const n = 12;
    for (let i = 0; i < n; i++) {
      const a = (i / n) * Math.PI * 2 + sym(r) * 0.18;
      const k0 = 1.12 + r() * 0.06, k1 = k0 + 0.1 + r() * 0.12;
      lines += `M${n2(Math.cos(a) * rx * k0)} ${n2(Math.sin(a) * ry * k0)} L${n2(Math.cos(a) * rx * k1)} ${n2(Math.sin(a) * ry * k1)} `;
    }
  }
  const s = Math.min((rx * 1.4) / L.vb[2], (ry * 1.2) / L.vb[3]);
  const lw = L.vb[2] * s, lh = L.vb[3] * s;
  const dots = o.dots !== false
    ? `<pattern id="${id}d" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(30)"><circle cx="2.5" cy="2.5" r="1.25" fill="${ink}" opacity=".22"/></pattern>
<radialGradient id="${id}g"><stop offset="0.45" stop-color="#000"/><stop offset="1" stop-color="#fff"/></radialGradient>
<mask id="${id}m"><path d="${toD(shapeD)}" fill="url(#${id}g)"/></mask>`
    : '';
  return `<svg class="comic-burst" xmlns="http://www.w3.org/2000/svg" viewBox="${n2(-W / 2)} ${n2(-H / 2)} ${n2(W)} ${n2(H)}" role="img" aria-label="${text.replace(/"/g, '&quot;')}">
<defs>${dots}</defs>
${lines ? `<path d="${lines}" stroke="${ink}" stroke-width="3.2" stroke-linecap="round"/>` : ''}
<path d="${toD(shapeD)}" fill="${ink}" transform="translate(4 5)"/>
<path d="${toD(shapeD)}" fill="${c1}" stroke="${ink}" stroke-width="3.6" stroke-linejoin="round"/>
${dots ? `<path d="${toD(shapeD)}" fill="url(#${id}d)" mask="url(#${id}m)"/>` : ''}
${c2 !== 'none' ? `<path d="${toD(shapeD, 0.8)}" fill="${c2}"/>` : ''}
<svg x="${n2(-lw / 2)}" y="${n2(-lh / 2)}" width="${n2(lw)}" height="${n2(lh)}" viewBox="${L.vb.map(n2).join(' ')}" overflow="visible">${L.body}</svg>
</svg>`;
}

/**
 * A comic pop-up element, animated (scale in with overshoot, a shake, gone in
 * ~1.1 s). Position it yourself (e.g. inside a centred layer) and append it.
 */
export function comicPop(text: string, o: PopOpts = {}): HTMLDivElement {
  const tilt = o.tilt ?? Math.round(sym(rng(hashStr(`${text}|tilt`))) * 8);
  const still = o.animate === false;
  const el = h('div', { class: `comic-pop${still ? ' still' : ''}`, role: 'status', style: `--tilt:${tilt}deg;width:${o.width ?? 280}px` });
  el.innerHTML = comicBurstSVG(text, o);
  if (o.sub) el.appendChild(h('div', { class: 'comic-sub' }, o.sub));
  const ms = o.ms ?? (still ? 0 : 1150);
  if (ms > 0) setTimeout(() => el.remove(), ms);
  return el;
}

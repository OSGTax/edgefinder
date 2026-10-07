import { hashStr, n2, rng, sym } from './rand';

// Hand lettering generated in code. Every glyph is a single-stroke skeleton
// (the way a kid draws a letter with a marker), drawn into a 10-unit-tall box
// (y = 0 is the cap line, y = 10 the baseline). At render time each letter
// gets its own small tilt, bounce off the baseline, size change and a wobble
// on every point, seeded from the text so a word always looks the same.
// The skeleton is then stroked in one of a few styles: comic (fat, slanted,
// ink outline and offset shadow, the display style), marker, poster, chalk or
// brush.

type Glyph = [width: number, path: string];

const G: Record<string, Glyph> = {
  A: [6, 'M0 10 L3 0 L6 10 M1.15 6.6 L4.9 6.4'],
  B: [5.6, 'M0.5 10 L0.5 0 C4.8 -0.2 5.2 1.4 5 2.5 C4.9 4.2 3 4.9 0.5 4.9 C4.4 4.8 5.8 5.8 5.7 7.4 C5.6 9.4 3.6 10 0.5 10'],
  C: [5.6, 'M5.5 1.6 C4.8 0.4 3.9 0 3 0 C1 0 0 2.4 0 5 C0 7.8 1.2 10 3.2 10 C4.4 10 5.2 9.3 5.7 8.3'],
  D: [5.8, 'M0.5 0 L0.5 10 C4.4 10.2 5.8 8 5.8 5 C5.8 1.8 4.2 -0.1 0.5 0'],
  E: [4.8, 'M4.6 0 L0.5 0 L0.5 10 L4.9 10 M0.5 4.9 L3.9 4.9'],
  F: [4.7, 'M4.7 0 L0.5 0 L0.5 10 M0.5 4.8 L3.7 4.8'],
  G: [6.1, 'M5.6 1.7 C4.9 0.5 4 0 3 0 C1 0 0 2.4 0 5 C0 7.8 1.2 10 3.2 10 C5 10 6.1 8.6 6.1 6.1 L3.5 6.1'],
  H: [5.6, 'M0.5 0 L0.5 10 M5.1 0 L5.1 10 M0.5 5.1 L5.1 4.9'],
  I: [3, 'M0 0 L3 0 M1.5 0 L1.5 10 M0 10 L3 10'],
  J: [4.6, 'M2.2 0 L4.4 0 M3.9 0 L3.9 7.2 C3.9 9.2 3 10 2 10 C0.8 10 0.1 9.2 0 8'],
  K: [5.2, 'M0.5 0 L0.5 10 M5 0 L0.5 6.2 M2.1 4.3 L5.3 10'],
  L: [4.4, 'M0.5 0 L0.5 10 L4.4 10'],
  M: [7.2, 'M0 10 L0.6 0 L3.6 6.6 L6.6 0 L7.2 10'],
  N: [5.7, 'M0.5 10 L0.5 0 L5.2 10 L5.2 0'],
  O: [6.3, 'M3.1 0 C1.1 0 0 2.3 0 5 C0 7.8 1.2 10 3.2 10 C5.1 10 6.3 7.8 6.3 5 C6.3 2.1 5 -0.1 2.6 0.3'],
  P: [5.2, 'M0.5 10 L0.5 0 C4 -0.2 5.3 1 5.2 2.7 C5.1 4.6 3.4 5.5 0.5 5.4'],
  Q: [6.4, 'M3.1 0 C1.1 0 0 2.3 0 5 C0 7.8 1.2 10 3.2 10 C5.1 10 6.3 7.8 6.3 5 C6.3 2.1 5 -0.1 2.6 0.3 M3.8 7.2 L6.5 10.4'],
  R: [5.4, 'M0.5 10 L0.5 0 C4 -0.2 5.3 1 5.2 2.6 C5.1 4.4 3.4 5.2 0.5 5.2 M2.4 5.2 L5.4 10'],
  S: [5.1, 'M4.8 1.5 C4.2 0.4 3.3 0 2.4 0 C1 0 0.2 0.9 0.2 2.3 C0.2 5 4.9 4.6 4.9 7.5 C4.9 9.1 3.7 10 2.4 10 C1.3 10 0.4 9.5 0 8.4'],
  T: [5.8, 'M0 0.2 L5.8 0 M2.9 0.1 L2.9 10'],
  U: [5.6, 'M0.4 0 L0.4 6.6 C0.4 9 1.6 10 2.8 10 C4 10 5.2 9 5.2 6.6 L5.2 0'],
  V: [6, 'M0 0 L3 10 L6 0'],
  W: [8.2, 'M0 0 L1.9 10 L4.1 3 L6.3 10 L8.2 0'],
  X: [5.6, 'M0 0 L5.6 10 M5.6 0 L0 10'],
  Y: [5.8, 'M0 0 L2.9 5.1 L5.8 0 M2.9 5.1 L2.9 10'],
  Z: [5.3, 'M0.2 0 L5.2 0 L0 10 L5.4 10'],
  '0': [5.1, 'M2.6 0 C0.9 0 0 2.4 0 5 C0 7.8 1 10 2.6 10 C4.2 10 5.1 7.8 5.1 5 C5.1 2.1 4.2 -0.1 2.1 0.3'],
  '1': [3.6, 'M0.2 2 L2 0 L2 10 M0.4 10 L3.6 10'],
  '2': [5.1, 'M0.3 2.2 C0.6 0.7 1.6 0 2.6 0 C3.9 0 4.8 1 4.8 2.6 C4.8 5.1 0 7.4 0 10 L5.1 10'],
  '3': [5, 'M0.3 1.1 C1 0.3 1.8 0 2.6 0 C5.3 0 5.2 4.6 2 4.8 C5.6 4.8 5.6 10 2.4 10 C1.4 10 0.6 9.6 0 8.8'],
  '4': [5.4, 'M3.8 10 L3.8 0 L0 7 L5.4 7'],
  '5': [5, 'M4.8 0 L1 0 L0.6 4.4 C1.2 4 2 3.8 2.6 3.8 C4.2 3.8 5 5 5 6.8 C5 8.8 3.8 10 2.4 10 C1.4 10 0.6 9.6 0 8.8'],
  '6': [5, 'M4.4 0.6 C3.8 0.2 3.2 0 2.6 0 C0.8 0 0 2.6 0 6 C0 8.6 1 10 2.6 10 C4.2 10 5 8.8 5 7.2 C5 5.4 4 4.4 2.6 4.4 C1.4 4.4 0.4 5.2 0 6'],
  '7': [5, 'M0 0.1 L5 0 L1.6 10'],
  '8': [5, 'M2.5 4.8 C0.4 4.8 0.4 0 2.5 0 C4.6 0 4.6 4.8 2.5 4.8 C0 4.8 0 10 2.5 10 C5 10 5 4.8 2.5 4.8'],
  '9': [5, 'M5 4 C4.6 5.4 3.6 6 2.5 6 C1 6 0 4.8 0 3 C0 1.2 1 0 2.5 0 C4.2 0 5 1.4 5 4 C5 7 4.4 10 1.8 10 C1 10 0.4 9.6 0.2 9.2'],
  '!': [1.8, 'M0.9 0 L0.9 6.8 M0.9 9.6 L0.9 9.75'],
  '?': [4.6, 'M0.2 2 C0.4 0.6 1.4 0 2.4 0 C3.6 0 4.6 0.8 4.6 2.2 C4.6 4.4 2.3 4.4 2.3 6.8 M2.3 9.6 L2.3 9.75'],
  '.': [1.5, 'M0.75 9.6 L0.75 9.75'],
  ',': [1.5, 'M1 9.3 L0.3 11.2'],
  "'": [1.4, 'M0.7 0 L0.7 2.7'],
  '’': [1.4, 'M0.9 0 L0.5 2.7'],
  '"': [2.8, 'M0.6 0 L0.6 2.7 M2.2 0 L2.2 2.7'],
  '“': [2.8, 'M0.6 0 L0.6 2.7 M2.2 0 L2.2 2.7'],
  '”': [2.8, 'M0.6 0 L0.6 2.7 M2.2 0 L2.2 2.7'],
  '-': [3.4, 'M0.2 5.6 L3.2 5.4'],
  '–': [4.4, 'M0.2 5.6 L4.2 5.4'],
  '—': [5.6, 'M0.2 5.6 L5.4 5.4'],
  ':': [1.5, 'M0.75 3.4 L0.75 3.55 M0.75 9.6 L0.75 9.75'],
  '/': [4, 'M4 0 L0 10'],
  '&': [6, 'M5.6 10 L1.5 3.7 C0.8 2.6 0.9 0 2.6 0 C4.5 0 4.4 2.6 2.6 3.8 L1.3 4.8 C0 5.8 0 10 2.6 10 C4 10 5 8.6 5.8 6.4'],
  '#': [6, 'M2 0 L1.2 10 M4.6 0 L3.8 10 M0.1 3.4 L6 3.4 M-0.1 6.8 L5.8 6.8'],
  '+': [4.4, 'M2.2 2.7 L2.2 7.7 M0 5.2 L4.4 5.2'],
  '½': [6, 'M0.3 1.3 L1.2 0.4 L1.2 4.4 M4.8 0 L1.4 10 M3.3 6.8 C3.5 5.8 4.2 5.6 4.7 5.6 C5.5 5.6 5.9 6.2 5.9 6.8 C5.9 8 3.3 8.8 3.3 10 L6 10'],
  '(': [2.4, 'M2.1 0 C0 2 0 8 2.1 10'],
  ')': [2.4, 'M0.3 0 C2.4 2 2.4 8 0.3 10'],
  '*': [4, 'M2 2 L2 6.4 M0.1 3 L3.9 5.4 M0.1 5.4 L3.9 3'],
  '%': [6, 'M5.6 0 L0.4 10 M1.3 0.4 C0.4 0.4 0.2 1.4 0.2 2 C0.2 2.8 0.6 3.6 1.3 3.6 C2.1 3.6 2.4 2.8 2.4 2 C2.4 1.2 2.1 0.4 1.3 0.4 M4.7 6.4 C3.9 6.4 3.6 7.2 3.6 8 C3.6 8.8 3.9 9.6 4.7 9.6 C5.5 9.6 5.8 8.8 5.8 8 C5.8 7.2 5.5 6.4 4.7 6.4'],
};

/** Marks added above a letter for accented names (Calderón, Peña...). */
const MARKS: Record<string, string> = {
  '́': 'M2.3 -1.2 L3.5 -2.8', // acute
  '̀': 'M2.3 -2.8 L3.5 -1.2', // grave
  '̃': 'M1.2 -1.6 C1.8 -2.6 2.6 -2.4 3 -2 C3.4 -1.6 4.2 -1.4 4.8 -2.4', // tilde
  '̈': 'M1.8 -1.6 L1.8 -1.45 M3.8 -1.6 L3.8 -1.45', // diaeresis
};

const SPACE = 3.2;
const GAP = 1.5;
/** letters are drawn a touch wider than their skeletons: kids letter wide */
const SX = 1.12;
const LINE = 13.2;

export type LetterStyle = 'comic' | 'marker' | 'poster' | 'chalk' | 'brush';

export interface LetterOpts {
  /** comic: fat slanted letters, thick ink outline, offset shadow (titles, pop-ups) ·
   *  marker (default): one even stroke · poster: like comic but upright ·
   *  chalk: grainy sidewalk chalk · brush: two loose passes, like house paint */
  style?: LetterStyle;
  /** stroke / paint colour (CSS colour, default the marker ink) */
  color?: string;
  /** colours cycled per word (overrides `color`) */
  colors?: string[];
  /** outline and shadow colour for poster/brush */
  ink?: string;
  /** cap height in CSS px (default 32) */
  size?: number;
  /** how shaky the hand is, 0..2 (default 1) */
  wobble?: number;
  /** stroke weight multiplier (default 1) */
  weight?: number;
  /** extra space between letters, in 10ths of the cap height (default 0) */
  spacing?: number;
  /** alignment of multi-line text */
  align?: 'left' | 'center' | 'right';
  /** extra seed so the same word can be drawn more than one way */
  seed?: string | number;
  /** a thin light streak on each stroke (comic: on by default) */
  highlight?: boolean;
  /** italic slant in degrees (comic: 11, others 0) */
  slant?: number;
  /** whole-word tilt in degrees (default 0) */
  tilt?: number;
  /** extra CSS class(es) on the <svg> */
  class?: string;
}

interface Pt { x: number; y: number }
type Seg = { c: string; p: Pt[] };

function parse(d: string): Seg[] {
  const out: Seg[] = [];
  const tok = d.match(/[MLCQ]|-?\d*\.?\d+/g) ?? [];
  let cur: Seg | null = null;
  for (let i = 0; i < tok.length; i++) {
    const t = tok[i];
    if (/[MLCQ]/.test(t)) { cur = { c: t, p: [] }; out.push(cur); continue; }
    const x = Number(t), y = Number(tok[++i]);
    if (!cur) continue;
    if ((cur.c === 'M' || cur.c === 'L') && cur.p.length === 1) { cur = { c: 'L', p: [] }; out.push(cur); }
    cur.p.push({ x, y });
  }
  return out;
}

const parsed = new Map<string, Seg[]>();
const segs = (d: string) => { let s = parsed.get(d); if (!s) { s = parse(d); parsed.set(d, s); } return s; };

interface Placed { segs: Seg[]; word: number }


function weightOf(style: LetterStyle) {
  return style === 'comic' ? 2.9 : style === 'poster' ? 2.5 : style === 'brush' ? 1.9 : style === 'chalk' ? 1.6 : 1.45;
}

/**
 * Lay out and wobble the text. Returns per-word-colour path data plus the
 * bounding box, all in glyph units.
 */
function layout(text: string, o: LetterOpts, pass: number) {
  const wob = (o.wobble ?? 1) * 0.2;
  const r = rng(hashStr(`${text}|${o.seed ?? ''}|${pass}`));
  const glyphR = rng(hashStr(`${text}|${o.seed ?? ''}|g`)); // shared by passes: brush passes differ only in wobble
  const lines = text.toUpperCase().normalize('NFD').split('\n');
  const style = o.style ?? 'marker';
  const spacing = GAP + (style === 'comic' ? 1.7 : style === 'poster' ? 1.3 : style === 'brush' ? 0.6 : 0) + (o.spacing ?? 0);
  const rows: { placed: Placed[]; w: number }[] = [];
  let word = 0;
  for (const line of lines) {
    const placed: Placed[] = [];
    let x = 0;
    const chars = [...line];
    for (let i = 0; i < chars.length; i++) {
      const ch = chars[i];
      if (ch === ' ') { x += SPACE; word++; continue; }
      if (MARKS[ch]) continue;
      const g = G[ch];
      if (!g) { x += SPACE * 0.6; continue; }
      const marks = [];
      for (let j = i + 1; j < chars.length && MARKS[chars[j]]; j++) marks.push(MARKS[chars[j]]);
      const [w, d] = g;
      const s = 1 + sym(glyphR) * 0.05;
      const rot = (sym(glyphR) * 3.2 * Math.PI) / 180;
      const dy = sym(glyphR) * 0.35;
      const cx = w / 2, cy = 5;
      const cos = Math.cos(rot), sin = Math.sin(rot);
      const tf = (p: Pt): Pt => {
        const px = p.x + sym(r) * wob, py = p.y + sym(r) * wob;
        const lx = (px - cx) * s * SX, ly = (py - cy) * s;
        return { x: x + cx * s * SX + lx * cos - ly * sin, y: cy + dy + lx * sin + ly * cos };
      };
      const all = [d, ...marks.map((m) => m.replace(/-?\d*\.?\d+ -?\d*\.?\d+/g, (pair) => {
        const [mx, my] = pair.split(' ').map(Number);
        return `${mx + (w - 6) / 2} ${my}`;
      }))];
      const out: Seg[] = [];
      for (const dd of all) for (const sg of segs(dd)) out.push({ c: sg.c, p: sg.p.map(tf) });
      placed.push({ segs: out, word });
      x += w * s * SX + spacing;
    }
    rows.push({ placed, w: Math.max(0, x - spacing) });
    word++;
  }
  const maxW = Math.max(1, ...rows.map((rw) => rw.w));
  const tilt = ((o.tilt ?? 0) * Math.PI) / 180;
  const tc = Math.cos(tilt), ts = Math.sin(tilt);
  const lineH = LINE + (style === 'comic' ? 2.4 : style === 'poster' ? 1.6 : 0);
  const slant = Math.tan((((o.slant ?? (style === 'comic' ? 11 : 0)) * Math.PI) / 180));
  const totalH = (rows.length - 1) * lineH + 10;
  const byWord = new Map<number, string[]>();
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  rows.forEach((row, li) => {
    const off = o.align === 'left' ? 0 : o.align === 'right' ? maxW - row.w : (maxW - row.w) / 2;
    const oy = li * lineH;
    for (const pl of row.placed) {
      const parts: string[] = [];
      for (const sg of pl.segs) {
        parts.push(sg.c);
        for (const p of sg.p) {
          const lx = p.x + off - maxW / 2, ly = p.y + oy - totalH / 2;
          const sx = lx - ly * slant;
          const X = sx * tc - ly * ts, Y = sx * ts + ly * tc;
          if (X < minX) minX = X; if (X > maxX) maxX = X;
          if (Y < minY) minY = Y; if (Y > maxY) maxY = Y;
          parts.push(`${n2(X)} ${n2(Y)}`);
        }
      }
      const list = byWord.get(pl.word) ?? [];
      list.push(parts.join(' '));
      byWord.set(pl.word, list);
    }
  });
  if (!isFinite(minX)) { minX = minY = 0; maxX = maxY = 1; }
  return { byWord, box: { minX, minY, maxX, maxY } };
}

/** The lettering as SVG pieces (viewBox and inner markup), for embedding in a bigger SVG. */
export function letteringParts(text: string, o: LetterOpts = {}): { vb: [number, number, number, number]; body: string } {
  const style = o.style ?? 'marker';
  const ink = o.ink ?? 'var(--marker, #23201c)';
  const colors = o.colors ?? [o.color ?? (style === 'chalk' ? 'var(--chalk, #f3f0e4)' : style === 'marker' ? ink : 'var(--highlighter, #f4d64e)')];
  const W = weightOf(style) * (o.weight ?? 1);
  const outline = style === 'comic' ? 1.75 : style === 'poster' ? 1.5 : style === 'brush' && o.ink ? 1.1 : 0;
  const shadow = style === 'comic' ? 1.1 : style === 'poster' ? 0.7 : 0;
  const L = layout(text, o, 0);
  const pad = W / 2 + outline + shadow + 0.4;
  const { minX, minY, maxX, maxY } = L.box;
  const colorOf = (w: number) => colors[w % colors.length];
  const words = [...L.byWord.entries()];
  const all = words.map(([, d]) => d.join(' ')).join(' ');
  const attrs = 'fill="none" stroke-linecap="round" stroke-linejoin="round"';
  let body = '';
  if (shadow) body += `<path d="${all}" stroke="${ink}" stroke-width="${n2(W + outline * 2)}" transform="translate(${n2(shadow * 0.75)} ${shadow})" ${attrs}/>`;
  if (outline) body += `<path d="${all}" stroke="${ink}" stroke-width="${n2(W + outline * 2)}" ${attrs}/>`;
  if (style === 'brush') {
    const L2 = layout(text, o, 1);
    for (const [w, d] of words) body += `<path d="${d.join(' ')}" stroke="${colorOf(w)}" stroke-width="${n2(W * 0.8)}" ${attrs}/>`;
    for (const [w, d] of L2.byWord) body += `<path d="${d.join(' ')}" stroke="${colorOf(w)}" stroke-width="${n2(W * 0.72)}" opacity="0.9" ${attrs}/>`;
  } else {
    const filter = style === 'chalk' ? ' filter="url(#gsl-chalk)"' : '';
    for (const [w, d] of words) body += `<path d="${d.join(' ')}" stroke="${colorOf(w)}" stroke-width="${n2(W)}"${filter} ${attrs}/>`;
    if (o.highlight ?? style === 'comic') body += `<path d="${all}" stroke="#fff" stroke-opacity=".45" stroke-width="${n2(W * 0.22)}" transform="translate(${n2(-W * 0.2)} ${n2(-W * 0.2)})" ${attrs}/>`;
  }
  return { vb: [minX - pad, minY - pad, maxX - minX + pad * 2, maxY - minY + pad * 2], body };
}

const cache = new Map<string, string>();

/** The lettering as an SVG string (cached). */
export function letteringSVG(text: string, o: LetterOpts = {}): string {
  const key = `${text}\u0000${JSON.stringify(o)}`;
  const hit = cache.get(key);
  if (hit) return hit;
  const { vb: [vx, vy, vw, vh], body } = letteringParts(text, o);
  const px = (o.size ?? 32) / 10;
  const label = text.replace(/\n/g, ' ').replace(/[&<>"]/g, (c) => `&#${c.charCodeAt(0)};`);
  const svg = `<svg class="lt lt-${o.style ?? 'marker'}${o.class ? ` ${o.class}` : ''}" xmlns="http://www.w3.org/2000/svg" viewBox="${n2(vx)} ${n2(vy)} ${n2(vw)} ${n2(vh)}" width="${n2(vw * px)}" height="${n2(vh * px)}" role="img" aria-label="${label}">${body}</svg>`;
  if (cache.size > 400) cache.delete(cache.keys().next().value!);
  cache.set(key, svg);
  return svg;
}

const tpl = typeof document !== 'undefined' ? document.createElement('template') : null;

/** Hand lettering as an inline <svg> element, ready to append. */
export function lettering(text: string, o: LetterOpts = {}): SVGSVGElement {
  ensureDefs();
  tpl!.innerHTML = letteringSVG(text, o);
  return tpl!.content.firstElementChild as SVGSVGElement;
}

/** Width/height ratio of a piece of lettering (for sizing it to a box). */
export function letteringAspect(text: string, o: LetterOpts = {}): number {
  const m = letteringSVG(text, o).match(/viewBox="[^ ]+ [^ ]+ ([^ ]+) ([^"]+)"/);
  return m ? Number(m[1]) / Number(m[2]) : 1;
}

let defsDone = false;
/** Shared SVG filters (chalk grain, rough edges), added to the page once. */
export function ensureDefs() {
  if (defsDone || typeof document === 'undefined') return;
  defsDone = true;
  const holder = document.createElement('div');
  holder.style.cssText = 'position:absolute;width:0;height:0;overflow:hidden';
  holder.setAttribute('aria-hidden', 'true');
  holder.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="0" height="0"><defs>
<filter id="gsl-chalk" x="-10%" y="-10%" width="120%" height="120%">
  <feTurbulence type="fractalNoise" baseFrequency="1.6" numOctaves="2" seed="4" result="n"/>
  <feColorMatrix in="n" type="matrix" values="0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 -1.6 1.25" result="holes"/>
  <feComposite in="SourceGraphic" in2="holes" operator="in" result="grit"/>
  <feTurbulence type="fractalNoise" baseFrequency="0.5" numOctaves="1" seed="9" result="n2"/>
  <feDisplacementMap in="grit" in2="n2" scale="0.5"/>
</filter>
<filter id="gsl-rough" x="-5%" y="-5%" width="110%" height="110%">
  <feTurbulence type="fractalNoise" baseFrequency="0.04" numOctaves="3" seed="2" result="n"/>
  <feDisplacementMap in="SourceGraphic" in2="n" scale="5"/>
</filter>
</defs></svg>`;
  document.body.appendChild(holder);
}

import { h } from '../ui/dom';

// The comic pop-ups: a jagged burst with our own word for the moment, for the
// big plays only. Words never repeat back to back, and each moment cycles
// through its own short list before reusing one.

export type Moment =
  | 'crush' | 'homer' | 'kLooking' | 'kSwinging' | 'snag' | 'splash' | 'fence'
  | 'double' | 'triple' | 'doublePlay' | 'oops' | 'hbp' | 'scores' | 'dog';

interface MomentStyle { words: string[]; fill: string; ink?: string; size: number; shape: 'burst' | 'cloud' | 'splat' }

const STYLE: Record<Moment, MomentStyle> = {
  crush: { words: ['THWACK!', 'KER-RACK!', 'SMACKED!'], fill: '#ffd23f', size: 1.15, shape: 'burst' },
  homer: { words: ['SEE YA!', 'OUTTA HERE!', 'BYE-BYE!', 'GOING, GONE!'], fill: '#ffd23f', size: 1.4, shape: 'burst' },
  kLooking: { words: ['SIT DOWN!', 'FROZEN!', 'CAUGHT LOOKING!'], fill: '#7fc8ff', size: 1, shape: 'burst' },
  kSwinging: { words: ['WHIFF!', 'WHOOSH!', 'FANNED!'], fill: '#ffffff', size: 1, shape: 'cloud' },
  snag: { words: ['SNAG!', 'GOTCHA!', 'YOINK!'], fill: '#8be28f', size: 1.05, shape: 'burst' },
  splash: { words: ['SPLOOSH!', 'KER-SPLASH!'], fill: '#7fd6ff', size: 1.2, shape: 'splat' },
  fence: { words: ['BONK!', 'CLANG!', 'THUNK!'], fill: '#ffb36b', size: 0.95, shape: 'burst' },
  double: { words: ['TWO BAGS!', 'ZIP!'], fill: '#8be28f', size: 1, shape: 'burst' },
  triple: { words: ['THREE BAGS!', 'ZOOM!'], fill: '#8be28f', size: 1.1, shape: 'burst' },
  doublePlay: { words: ['TWO FOR ONE!', 'DOUBLE PLAY!'], fill: '#ff8a87', size: 1.05, shape: 'burst' },
  oops: { words: ['OOPS!', 'WHOOPS!', 'BUTTERFINGERS!'], fill: '#ffb36b', size: 0.95, shape: 'splat' },
  hbp: { words: ['OUCH!', 'YEOWCH!'], fill: '#ff8a87', size: 1, shape: 'burst' },
  scores: { words: ['SCORES!', 'HOME FREE!'], fill: '#8be28f', size: 0.95, shape: 'burst' },
  dog: { words: ['WOOF!', 'ARF!'], fill: '#ffffff', size: 0.85, shape: 'cloud' },
};

const NS = 'http://www.w3.org/2000/svg';

/** A jagged starburst, a puffy cloud, or a splat, as an SVG path in a 200×120 box. */
function shapePath(kind: MomentStyle['shape']): string {
  const cx = 100, cy = 60, rx = 92, ry = 52;
  if (kind === 'cloud') {
    // puffy bumps round an ellipse
    const n = 11;
    let d = '';
    for (let i = 0; i <= n; i++) {
      const a = (i / n) * Math.PI * 2;
      const x = cx + Math.cos(a) * rx * 0.86, y = cy + Math.sin(a) * ry * 0.82;
      if (i === 0) { d += `M${x.toFixed(1)},${y.toFixed(1)}`; continue; }
      const am = ((i - 0.5) / n) * Math.PI * 2;
      const bump = 1.22 + Math.random() * 0.1;
      d += ` Q${(cx + Math.cos(am) * rx * bump).toFixed(1)},${(cy + Math.sin(am) * ry * bump).toFixed(1)} ${x.toFixed(1)},${y.toFixed(1)}`;
    }
    return d + 'Z';
  }
  const n = kind === 'splat' ? 9 : 14;
  const at = (i: number, k: number) => {
    const a = (i / (n * 2)) * Math.PI * 2 + 0.1;
    return `${(cx + Math.cos(a) * rx * k).toFixed(1)},${(cy + Math.sin(a) * ry * k).toFixed(1)}`;
  };
  const outer = Array.from({ length: n }, (_, i) => at(i * 2, 1 + (Math.random() - 0.5) * 0.18));
  const inner = Array.from({ length: n }, (_, i) => at(i * 2 + 1, kind === 'splat' ? 0.55 + Math.random() * 0.12 : 0.7 + Math.random() * 0.08));
  if (kind === 'splat') {
    // blobby drips: curve from tip to tip, pulled in toward the middle
    return `M${outer[0]}` + outer.map((_, i) => ` Q${inner[i]} ${outer[(i + 1) % n]}`).join('') + 'Z';
  }
  return 'M' + outer.map((o, i) => `${o} L${inner[i]}`).join(' L') + 'Z';
}

function burst(st: MomentStyle): SVGSVGElement {
  const svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('viewBox', '-8 -8 216 136');
  svg.setAttribute('class', 'comic-shape');
  const id = `ht${Math.floor(Math.random() * 1e9)}`;
  const d = shapePath(st.shape);
  svg.innerHTML = `
    <defs><pattern id="${id}" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(30)">
      <circle cx="3.5" cy="3.5" r="1.5" fill="rgba(43,29,20,0.16)"/></pattern></defs>
    <path d="${d}" transform="translate(6 7)" fill="#2b1d14"/>
    <path d="${d}" fill="${st.fill}" stroke="#2b1d14" stroke-width="5" stroke-linejoin="round"/>
    <path d="${d}" fill="url(#${id})"/>`;
  return svg;
}

export class ComicPops {
  private last = '';
  private used = new Map<Moment, number>();
  private lastT = -9;

  constructor(private host: HTMLElement) {}

  /** the next word for a moment: cycles its list, never the same word as last time */
  word(m: Moment): string {
    const ws = STYLE[m].words;
    let i = this.used.get(m) ?? Math.floor(Math.random() * ws.length);
    let w = ws[i % ws.length];
    if (w === this.last && ws.length > 1) { i++; w = ws[i % ws.length]; }
    this.used.set(m, i + 1);
    this.last = w;
    return w;
  }

  /** Show a pop-up. `now` is game time; a smaller moment won't stomp a fresh one. */
  show(m: Moment, now: number, text?: string) {
    const st = STYLE[m];
    if (now - this.lastT < 0.5 && st.size < 1.1) return;
    this.lastT = now;
    const word = text ?? this.word(m);
    const tilt = (Math.random() * 10 - 5).toFixed(1);
    const el = h('div', { class: `comic ${st.shape}`, style: `--s:${st.size};--tilt:${tilt}deg` },
      burst(st) as unknown as Node,
      h('div', { class: 'comic-word', style: word.length > 9 ? 'font-size:0.78em' : '' }, word));
    while (this.host.firstChild) this.host.removeChild(this.host.firstChild);
    this.host.appendChild(el);
    // dev: `window.__holdPops = true` keeps the last one up for screenshots in the slow headless browser
    if (import.meta.env.DEV && (window as unknown as { __holdPops?: boolean }).__holdPops) { el.getAnimations().forEach((a) => { a.currentTime = 450; a.pause(); }); return; }
    setTimeout(() => el.remove(), 1250);
  }
}

import { h } from '../dom';
import { icon, type IconName } from './icons';
import { ensureDefs, lettering, letteringParts, type LetterOpts } from './letters';
import { comicPop } from './comic';
import { hashStr, n2, rng, sym } from './rand';

// The kid-made surfaces as small DOM helpers. Each returns a plain element with
// the matching CSS class from `look.css`, so a screen can also write the
// markup by hand. Imperfection (tilt, torn edges, where the tape lands) is
// seeded from the content so it stays put between redraws.

type Kids = (Node | string | null | undefined | false)[];

const seedOf = (s: string | number | undefined, fallback: string) => (s == null ? hashStr(fallback) : typeof s === 'number' ? s : hashStr(s));

/** A small fixed tilt in degrees, from a seed. */
export function tiltFor(seed: string | number, max = 1.6): number {
  return Math.round(sym(rng(seedOf(seed, ''))) * max * 10) / 10;
}

/**
 * A torn/hand-cut edge as a CSS clip-path polygon.
 * `edges` picks which sides are rough; `depth` is how deep the tears go (px).
 */
export function tornClip(seed: string | number, edges = 'trbl', depth = 3, teeth = 18): string {
  const r = rng(seedOf(seed, 'torn'));
  const pts: string[] = [];
  const d = (rough: boolean) => (rough ? n2(r() * depth) : '0');
  const side = (rough: boolean, f: (t: number, off: string) => string) => {
    for (let i = 0; i < teeth; i++) pts.push(f((i / teeth) * 100, d(rough)));
  };
  side(edges.includes('t'), (t, o) => `${n2(t)}% ${o}px`);
  side(edges.includes('r'), (t, o) => `calc(100% - ${o}px) ${n2(t)}%`);
  side(edges.includes('b'), (t, o) => `${n2(100 - t)}% calc(100% - ${o}px)`);
  side(edges.includes('l'), (t, o) => `${o}px ${n2(100 - t)}%`);
  return `polygon(${pts.join(',')})`;
}

/** Apply a torn edge to an element. */
export function tear(el: HTMLElement, seed: string | number, edges = 'trbl', depth = 3): HTMLElement {
  el.style.clipPath = tornClip(seed, edges, depth);
  return el;
}

/** Hand lettering in a <span> wrapper (keeps text flow and lets CSS size it). */
export function letters(text: string, o: LetterOpts = {}): HTMLSpanElement {
  const s = h('span', { class: 'letters' });
  s.appendChild(lettering(text, o));
  return s;
}

/** A piece of masking tape with a word or two in marker on it. */
export function tape(content: string | Node, o: { tilt?: number; tone?: 'cream' | 'blue' | 'red'; seed?: string } = {}): HTMLSpanElement {
  const seed = o.seed ?? (typeof content === 'string' ? content : 'tape');
  const el = h('span', { class: `tape${o.tone && o.tone !== 'cream' ? ` tape-${o.tone}` : ''}`, style: `--tilt:${o.tilt ?? tiltFor(seed, 2.4)}deg` }, content);
  el.style.clipPath = tornClip(`${seed}|tape`, 'lr', 3, 8);
  return el;
}

/** Two strips of tape holding something to the wall (decoration, added to `el`). */
export function tapeCorners(el: HTMLElement, seed = 'corners', which: ('tl' | 'tr' | 'bl' | 'br')[] = ['tl', 'tr']): HTMLElement {
  const r = rng(hashStr(seed));
  for (const c of which) {
    const t = h('i', { class: `tape-bit ${c}`, style: `--tilt:${n2((c === 'tl' || c === 'br' ? -38 : 38) + sym(r) * 8)}deg;--nudge:${n2(sym(r) * 6)}px` });
    t.style.clipPath = tornClip(`${seed}|${c}`, 'lr', 2.5, 6);
    el.appendChild(t);
  }
  return el;
}

/** A sign: flat cardboard tan with an ink outline and a slight tilt. */
export function sign(children: Kids, o: { seed?: string; tilt?: number; class?: string } = {}): HTMLDivElement {
  const seed = o.seed ?? 'sign';
  return h('div', { class: `cardboard${o.class ? ` ${o.class}` : ''}`, style: `--tilt:${o.tilt ?? tiltFor(seed)}deg` }, ...children);
}

/** A card or sheet: cream, ink outline. */
export function paper(children: Kids, o: { seed?: string; tilt?: number; class?: string; ruled?: boolean } = {}): HTMLDivElement {
  return h('div', { class: `paper${o.ruled ? ' ruled' : ''}${o.class ? ` ${o.class}` : ''}`, style: `--tilt:${o.tilt ?? tiltFor(o.seed ?? 'paper', 1)}deg` }, ...children);
}

/** A chalkboard in a wooden frame. */
export function chalkboard(children: Kids, o: { class?: string } = {}): HTMLDivElement {
  ensureDefs();
  return h('div', { class: `chalkboard${o.class ? ` ${o.class}` : ''}` }, h('div', { class: 'chalkboard-slate' }, ...children), h('i', { class: 'chalk-tray' }));
}

/** A clipboard: the board, a metal clip, a ruled sheet. */
export function clipboard(children: Kids, o: { class?: string; title?: string } = {}): HTMLDivElement {
  return h('div', { class: `clipboard${o.class ? ` ${o.class}` : ''}` },
    h('i', { class: 'clip' }),
    h('div', { class: 'clipboard-sheet' },
      o.title ? h('div', { class: 'sheet-title' }, lettering(o.title, { style: 'comic', size: 22, color: 'var(--sunshine)', seed: 'clip' })) : null,
      ...children));
}

/** A button dressed in the kit: icon + label. `kind` picks the surface. */
export function button(label: string, onClick: (e: Event) => void, o: { icon?: IconName; kind?: 'go' | 'plain' | 'tape' | 'chalk'; size?: 'big' | 'small'; lettered?: boolean; class?: string; seed?: string } = {}): HTMLButtonElement {
  const kind = o.kind ?? 'plain';
  const cls = ['btn', kind === 'go' ? 'go' : kind === 'tape' ? 'btn-tape' : kind === 'chalk' ? 'btn-chalk' : '', o.size ?? '', o.class ?? ''].filter(Boolean).join(' ');
  const seed = o.seed ?? label;
  const b = h('button', { class: cls, type: 'button', style: `--tilt:${tiltFor(seed, 1.2)}deg`, 'aria-label': label, onclick: onClick });
  if (o.icon) b.appendChild(icon(o.icon));
  if (o.lettered) b.appendChild(lettering(label, { size: o.size === 'big' ? 22 : 15, seed, style: kind === 'chalk' ? 'chalk' : 'marker' }));
  else b.appendChild(h('span', null, label));
  return b;
}

/**
 * A pennant for a team: a triangle in the team colour with a stitched edge, a
 * sleeve in the second colour, the name in comic lettering, all outlined in
 * ink. Returned as an inline SVG (scales to its box).
 */
export function pennant(t: { id: string; name: string; street: string; colors: { primary: string; secondary: string; accent: string } }, o: { class?: string } = {}): SVGSVGElement {
  ensureDefs();
  const name = letteringParts(t.name, { style: 'comic', color: t.colors.secondary, ink: '#2b1d14', seed: `pennant-${t.id}`, wobble: 0.8 });
  const street = letteringParts(t.street, { style: 'marker', ink: t.colors.accent, seed: `street-${t.id}`, weight: 1.25 });
  // fit the name into the triangle's fat end
  const fit = (vb: number[], x: number, y: number, w: number, hgt: number) => {
    const s = Math.min(w / vb[2], hgt / vb[3]);
    return `<svg x="${n2(x + (w - vb[2] * s) / 2)}" y="${n2(y + (hgt - vb[3] * s) / 2)}" width="${n2(vb[2] * s)}" height="${n2(vb[3] * s)}" viewBox="${vb.map(n2).join(' ')}" overflow="visible">`;
  };
  const r = rng(hashStr(`pennant-${t.id}`));
  const tip = 60 + sym(r) * 3;
  const stitches = Array.from({ length: 12 }, (_, i) => `M${n2(32 + i * 21)} ${n2(8 + i * 3.9)} l9 ${n2(1.7)}`).join(' ') +
    ' ' + Array.from({ length: 12 }, (_, i) => `M${n2(32 + i * 21)} ${n2(112 - i * 3.9)} l9 -1.7`).join(' ');
  const svg = `<svg class="pennant${o.class ? ` ${o.class}` : ''}" xmlns="http://www.w3.org/2000/svg" viewBox="-4 -6 312 134" role="img" aria-label="${t.street} ${t.name}">
<path d="M26 4 L300 ${n2(tip)} L26 116 Z" fill="#2b1d14" transform="translate(4 5)"/>
<path d="M26 4 L300 ${n2(tip)} L26 116 Z" fill="${t.colors.primary}" stroke="#2b1d14" stroke-width="3.5" stroke-linejoin="round"/>
<path d="${stitches}" stroke="${t.colors.accent}" stroke-width="1.6" stroke-linecap="round" opacity=".7"/>
<rect x="2" y="0" width="26" height="120" rx="4" fill="${t.colors.secondary}" stroke="#2b1d14" stroke-width="3.5"/>
<path d="M15 8 L15 112" stroke="rgba(43,29,20,.35)" stroke-width="2" stroke-dasharray="5 5" stroke-linecap="round"/>
${fit(name.vb, 38, 22, 180, 60)}${name.body}</svg>
${fit(street.vb, 46, 82, 120, 16)}${street.body}</svg>
</svg>`;
  const tpl = document.createElement('template');
  tpl.innerHTML = svg;
  return tpl.content.firstElementChild as SVGSVGElement;
}

/**
 * A trading card. Front: photo, name in marker, persona on a typed label,
 * two or three stats. Back (optional): anything — the full stat sheet and bio.
 * Toggle `.flipped` (or call `flipCard`) to turn it over.
 */
export function tradingCard(o: {
  photo: HTMLElement; name: string; persona?: string; number?: string | number;
  team: { id: string; abbr?: string; colors: { primary: string; secondary: string; accent: string } };
  stats?: [label: string, value: string | number][]; corner?: Node; back?: Node; class?: string; seed?: string;
}): HTMLDivElement {
  const seed = o.seed ?? o.name;
  const front = h('div', { class: 'tcard-face tcard-front' },
    h('div', { class: 'tcard-photo' }, o.photo, o.number != null ? h('span', { class: 'tcard-num' }, lettering(String(o.number), { size: 14, style: 'comic', color: o.team.colors.secondary, seed })) : null, o.corner ?? null),
    h('div', { class: 'tcard-name' }, lettering(o.name, { size: 17, style: 'comic', color: 'var(--poster)', seed, wobble: 0.9 })),
    o.persona ? h('div', { class: 'typed' }, o.persona) : null,
    o.stats?.length ? h('div', { class: 'tcard-stats' }, ...o.stats.map(([l, v]) => h('div', { class: 'tcard-stat' }, h('b', null, String(v)), h('span', null, l)))) : null);
  const card = h('div', {
    class: `tcard${o.class ? ` ${o.class}` : ''}`,
    style: `--team:${o.team.colors.primary};--team2:${o.team.colors.secondary};--team3:${o.team.colors.accent};--tilt:${tiltFor(seed, 1.8)}deg`,
  }, h('div', { class: 'tcard-inner' }, front, o.back ? h('div', { class: 'tcard-face tcard-back' }, o.back) : null));
  return card;
}

export function flipCard(card: HTMLElement, back?: boolean) {
  card.classList.toggle('flipped', back);
}

/**
 * The Channel 4½ (Maple Hollow Public Access) lower-third: a bright caption
 * strip with the station logo and who's talking. Keep the line short.
 */
export function lowerThird(o: { who: string; role?: string; text: string | Node; tone?: 'chet' | 'dottie' | string }): HTMLDivElement {
  return h('div', { class: `lower-third${o.tone ? ` lt-${o.tone}` : ''}` },
    channelBug(),
    h('div', { class: 'lt-body' },
      h('div', { class: 'lt-who' }, h('b', null, o.who), o.role ? h('span', null, o.role) : null),
      h('div', { class: 'lt-text' }, o.text)));
}

/** The station logo: a lopsided blue blob with a hand-lettered "4½". */
export function channelBug(): HTMLSpanElement {
  const el = h('span', { class: 'ch-bug', 'aria-label': 'Channel 4½' });
  el.appendChild(lettering('4½', { style: 'comic', size: 15, color: '#fff6e0', seed: 'ch4' }));
  return el;
}

/** A big moment ("SEE YA!"): now a comic pop-up. Kept for older callers; use comicPop. */
export function bigMoment(text: string, o: { color?: string; ink?: string; sub?: string; ms?: number; seed?: string } = {}): HTMLDivElement {
  return comicPop(text, { textColor: o.color, ink: o.ink, sub: o.sub, ms: o.ms ?? 1150, seed: o.seed });
}

/** The "turn your phone sideways" card (show it with CSS when portrait). */
export function rotateHint(text = 'Turn your phone sideways. The yard is wider than it is tall.'): HTMLDivElement {
  return h('div', { class: 'rotate-card' }, h('span', { class: 'rotate-ico' }, icon('rotate')), h('span', null, text));
}

/**
 * A round team patch with a stitched ring and the team's initial in
 * poster lettering (for scoreboards, lists, the match intro).
 */
export function teamPatch(t: { id: string; name: string; colors: { primary: string; secondary: string; accent: string } }, size = 28): SVGSVGElement {
  ensureDefs();
  const L = letteringParts(t.name[0], { style: 'comic', color: t.colors.secondary, ink: '#2b1d14', seed: `patch-${t.id}`, wobble: 0.6 });
  const s = Math.min(26 / L.vb[2], 26 / L.vb[3]);
  const w = L.vb[2] * s, hh = L.vb[3] * s;
  const svg = `<svg class="patch" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48" width="${size}" height="${size}" role="img" aria-label="${t.name}">
<circle cx="25.5" cy="25.5" r="21" fill="#2b1d14"/>
<circle cx="24" cy="23.6" r="21" fill="${t.colors.primary}" stroke="#2b1d14" stroke-width="3"/>
<circle cx="24" cy="23.6" r="16.4" fill="none" stroke="${t.colors.secondary}" stroke-width="2" stroke-dasharray="3.2 2.6" stroke-linecap="round"/>
<svg x="${n2(24 - w / 2)}" y="${n2(24 - hh / 2)}" width="${n2(w)}" height="${n2(hh)}" viewBox="${L.vb.map(n2).join(' ')}" overflow="visible">${L.body}</svg>
</svg>`;
  const tpl = document.createElement('template');
  tpl.innerHTML = svg;
  return tpl.content.firstElementChild as SVGSVGElement;
}

import { hashStr, n2, rng, sym } from './rand';

// The icon set: drawn in code, one hand. Every icon is a few marker strokes
// in a 24-unit box, wobbled a little (seeded by its name, so it never jitters)
// and stroked with round caps. Strokes use `currentColor`; parts marked
// `accent` use `--ico-accent` (falls back to currentColor), e.g. a ball's red
// stitches. Nothing here is an emoji or a font glyph.

type Part = { d: string; fill?: boolean; accent?: boolean; w?: number };

const I: Record<string, Part[]> = {
  /** a baseball: Play ball, loading */
  ball: [
    { d: 'M12 3 C17 3 21 7 21 12 C21 17 17 21 12 21 C7 21 3 17 3 12 C3 7 7 3 11.4 3.1' },
    { d: 'M7.4 4.4 C9.8 8.4 9.8 15.6 7.4 19.6', accent: true, w: 1.6 },
    { d: 'M16.6 4.4 C14.2 8.4 14.2 15.6 16.6 19.6', accent: true, w: 1.6 },
    { d: 'M7.4 7.6 L9.4 7 M8 11 L10.2 11 M7.6 14.6 L9.6 15.2 M16.6 7.6 L14.6 7 M16 11 L13.8 11 M16.4 14.6 L14.4 15.2', accent: true, w: 1.2 },
  ],
  /** a lightning bolt: a kid's special move */
  bolt: [{ d: 'M13.8 2.4 L5.2 13.6 L11.2 13.4 L9.4 21.8 L19 9.6 L12.8 9.8 Z', fill: true }],
  /** run forward / next */
  forward: [{ d: 'M3 12.4 C8 11.8 14 12.1 19.6 12 M13.8 6.2 L20 12 L14 17.8' }],
  /** go back / previous */
  back: [{ d: 'M21 12.4 C16 11.8 10 12.1 4.4 12 M10.2 6.2 L4 12 L10 17.8' }],
  /** pause */
  pause: [{ d: 'M8.4 5 L8.2 19 M15.8 5 L15.8 19', w: 3.2 }],
  /** play / resume */
  play: [{ d: 'M7 4.6 L19.4 12 L7.2 19.4 Z', fill: true }],
  /** top of the inning */
  up: [{ d: 'M12 6 L19 17 L5 17.2 Z', fill: true }],
  /** bottom of the inning */
  down: [{ d: 'M12 18 L19 7 L5 6.8 Z', fill: true }],
  /** close a panel */
  close: [{ d: 'M5.6 5.4 L18.4 18.6 M18.6 5.6 L5.4 18.2' }],
  /** a check mark (ticked box, done) */
  check: [{ d: 'M4 13 L9.6 18.4 L20.2 5.4' }],
  /** a phone, as it is now (tall) */
  phone: [
    { d: 'M8.6 2.6 L15.4 2.6 C16.6 2.6 17.4 3.4 17.4 4.6 L17.4 19.4 C17.4 20.6 16.6 21.4 15.4 21.4 L8.6 21.4 C7.4 21.4 6.6 20.6 6.6 19.4 L6.6 4.6 C6.6 3.4 7.4 2.6 8.6 2.6' },
    { d: 'M11 18.6 L13 18.6', w: 1.6 },
  ],
  /** turn the phone sideways */
  rotate: [
    { d: 'M4.2 9.6 L4.2 17.6 C4.2 18.6 4.8 19.2 5.8 19.2 L18.2 19.2 C19.2 19.2 19.8 18.6 19.8 17.6 L19.8 11.2 C19.8 10.2 19.2 9.6 18.2 9.6 L5.8 9.6 C4.8 9.6 4.2 10.2 4.2 11.2' },
    { d: 'M17.2 14.4 L17.2 14.6', w: 2 },
    { d: 'M5.6 6.4 C7 3.6 10.6 2.4 13.8 3.4 C15.4 3.9 16.6 4.9 17.4 6.2 M17.8 2.8 L17.6 6.4 L14.2 6.4', accent: true, w: 1.8 },
  ],
  /** replay (the coach, a moment) */
  replay: [{ d: 'M5.4 13 C5.4 16.8 8.4 19.8 12.2 19.8 C16 19.8 19 16.8 19 13 C19 9.2 16 6.2 12.2 6.2 L7.6 6.2 M10.4 3 L7.2 6.2 L10.4 9.4' }],
  /** the coach's whistle (How to Play) */
  whistle: [
    { d: 'M9.6 9.4 C6.6 9.4 4.2 11.8 4.2 14.8 C4.2 17.8 6.6 20.2 9.6 20.2 C12.6 20.2 15 17.8 15 14.8 L15 13 L20.6 13 L20.6 9.4 Z' },
    { d: 'M9.6 14.8 L9.8 14.9', w: 3 },
    { d: 'M5.2 11.6 C3.6 8.6 3.6 5.4 6 3.6', accent: true, w: 1.6 },
  ],
  /** a clipboard (Settings) */
  clipboard: [
    { d: 'M6.2 4.6 L17.8 4.6 C18.6 4.6 19.2 5.2 19.2 6 L19.2 20.4 C19.2 21.2 18.6 21.6 17.8 21.6 L6.2 21.6 C5.4 21.6 4.8 21.2 4.8 20.4 L4.8 6 C4.8 5.2 5.4 4.6 6.2 4.6' },
    { d: 'M9 2.6 L15 2.6 L15.4 6.4 L8.6 6.4 Z', fill: true, accent: true },
    { d: 'M8 10.6 L16 10.4 M8 13.8 L15 13.8 M8 17 L13 17.2', w: 1.5 },
  ],
  /** trading cards (Meet the Kids) */
  cards: [
    { d: 'M3.6 7.4 L11.4 5 L15.4 18 L7.6 20.4 Z' },
    { d: 'M11.6 4.4 L19.6 4.6 L19.4 18.4 L15.6 18.4', accent: true },
    { d: 'M7 10.4 L10.4 9.4 M8 13.8 L11.6 12.8', w: 1.4 },
  ],
  /** a pennant (team pick) */
  pennant: [
    { d: 'M4.2 2.8 L4.2 21.4', w: 2.4 },
    { d: 'M4.6 4.4 L20.6 9.6 L4.6 15 Z', fill: true, accent: true },
  ],
  /** a chalkboard (How to Play) */
  chalkboard: [
    { d: 'M3 4.4 L21 4.4 L21 16.4 L3 16.4 Z' },
    { d: 'M7.6 16.6 L5.6 21.4 M16.4 16.6 L18.4 21.4', w: 1.8 },
    { d: 'M6.4 8.4 C8 7.4 9.6 9.6 11.2 8.6 M6.4 12 L14.6 11.6', accent: true, w: 1.5 },
  ],
  /** sound on */
  sound: [
    { d: 'M3.6 9.2 L7.4 9.2 L12.6 4.6 L12.6 19.4 L7.4 14.8 L3.6 14.8 Z', fill: true },
    { d: 'M15.6 9 C16.8 10.6 16.8 13.4 15.6 15 M18.2 6.4 C20.8 9.4 20.8 14.6 18.2 17.6', w: 1.8 },
  ],
  /** sound off */
  mute: [
    { d: 'M3.6 9.2 L7.4 9.2 L12.6 4.6 L12.6 19.4 L7.4 14.8 L3.6 14.8 Z', fill: true },
    { d: 'M15.6 9.4 L20.6 14.6 M20.6 9.4 L15.6 14.6', w: 1.8 },
  ],
  /** music */
  music: [
    { d: 'M9 17.4 L9 5.4 L19 3.4 L19 15.4', w: 1.9 },
    { d: 'M6.6 15.4 C8.2 15.4 9.2 16.2 9.2 17.4 C9.2 18.8 8 19.8 6.4 19.8 C5 19.8 4.2 19 4.2 18 C4.2 16.6 5.2 15.4 6.6 15.4 Z M16.6 13.4 C18.2 13.4 19.2 14.2 19.2 15.4 C19.2 16.8 18 17.8 16.4 17.8 C15 17.8 14.2 17 14.2 16 C14.2 14.6 15.2 13.4 16.6 13.4 Z', fill: true },
  ],
  /** the announcers' microphone (voice) */
  mic: [
    { d: 'M12 2.8 C13.8 2.8 15 4 15 5.8 L15 11.4 C15 13.2 13.8 14.4 12 14.4 C10.2 14.4 9 13.2 9 11.4 L9 5.8 C9 4 10.2 2.8 11.8 2.8 Z', fill: true },
    { d: 'M5.8 10.6 C5.8 14.6 8.6 17.6 12 17.6 C15.4 17.6 18.2 14.6 18.2 10.6 M12 17.6 L12 21.4 M8.6 21.4 L15.4 21.4', w: 1.8 },
  ],
  /** home: the title screen */
  home: [{ d: 'M3.4 11.6 L12 3.8 L20.6 11.6 M5.8 9.8 L5.8 20.2 L18.2 20.2 L18.2 9.8 M10 20.2 L10 14.6 L14 14.6 L14 20.2' }],
  /** home plate (a throw home, a base target) */
  plate: [{ d: 'M5 5 L19 5 L19 12.4 L12 19.6 L5 12.4 Z' }],
  /** a base bag */
  base: [{ d: 'M12 3.6 L20.4 12 L12 20.4 L3.6 12 Z' }],
  /** a little diamond with the bases (a scoreboard mark) */
  diamond: [{ d: 'M12 4 L20 12 L12 20 L4 12 Z M12 4 L12 4.1', w: 1.6 }],
  /** fullscreen */
  fullscreen: [{ d: 'M4 9 L4 4 L9 4 M15 4 L20 4 L20 9 M20 15 L20 20 L15 20 M9 20 L4 20 L4 15' }],
  /** the share button on an iPhone (Add to Home Screen hint) */
  share: [{ d: 'M8.4 8.4 L5.6 8.4 L5.6 20.6 L18.4 20.6 L18.4 8.4 L15.6 8.4 M12 15 L12 2.8 M8.4 6.2 L12 2.6 L15.6 6.2' }],
  /** add (to the home screen) */
  add: [{ d: 'M5.2 3.8 L18.8 3.8 C19.6 3.8 20.2 4.4 20.2 5.2 L20.2 18.8 C20.2 19.6 19.6 20.2 18.8 20.2 L5.2 20.2 C4.4 20.2 3.8 19.6 3.8 18.8 L3.8 5.2 C3.8 4.4 4.4 3.8 5.2 3.8 M12 8 L12 16 M8 12 L16 12' }],
  /** a trophy (winning) */
  trophy: [
    { d: 'M7 3.6 L17 3.6 L17 9.6 C17 12.6 14.8 14.6 12 14.6 C9.2 14.6 7 12.6 7 9.6 Z' },
    { d: 'M7 5.6 C3.6 5.4 3.6 10 7.4 10.6 M17 5.6 C20.4 5.4 20.4 10 16.6 10.6 M12 14.6 L12 18 M8 20.6 L16 20.6 L15 18 L9 18 Z', w: 1.8 },
  ],
  /** a star (a star of the game, a favourite) */
  star: [{ d: 'M12 2.8 L14.6 9 L21 9.4 L16 13.6 L17.8 20.4 L12 16.6 L6.2 20.4 L8 13.6 L3 9.4 L9.4 9 Z', fill: true }],
  /** a bat */
  bat: [{ d: 'M4.6 20.6 L3.6 19.6 M5.2 19.2 L4.6 18.6 C5.2 17.6 6 17 7 16.4 L17.4 5.2 C18.6 3.8 20.6 3.6 21 4.4 C21.4 5.4 20.4 6.8 18.8 7.6 L8 17.4 C7.4 18 6.6 18.8 5.6 19.4 Z' }],
  /** a ball glove */
  glove: [{ d: 'M6.6 20.4 C4.4 17.6 4 13 4.6 9.4 C4.8 8 6.4 7.8 6.8 9.2 L7.4 11.6 L7.2 5 C7.2 3.4 9.4 3.2 9.6 4.8 L10.2 9.6 L10.4 3.6 C10.6 2 12.8 2 12.8 3.8 L13 9.6 L13.8 4.8 C14.2 3.2 16.2 3.6 16 5.2 L15.6 11 C16.8 9.8 18.6 9.4 19.4 10.4 C20.2 11.4 18.4 13 17.4 14.4 C16 16.6 15.4 18.6 15.2 20.4 Z' }],
  /** a finger tapping (tap hints) */
  tap: [
    { d: 'M9.6 21 L6.4 15.6 C5.8 14.6 6.8 13.4 7.8 14 L9.6 15.4 L9.6 6.4 C9.6 5.2 11.6 5.2 11.6 6.4 L11.6 12 L11.6 10.6 C11.6 9.6 13.6 9.6 13.6 10.6 L13.6 12.2 C13.6 11.2 15.6 11.2 15.6 12.2 L15.6 13 C15.6 12.2 17.6 12.2 17.6 13.2 L17.6 17 C17.6 18.8 16.8 20 16 21' },
    { d: 'M7.6 4.2 C8.6 2.6 12.6 2.4 13.8 4.4', accent: true, w: 1.5 },
  ],
  /** flip a card over */
  flip: [
    { d: 'M8 5.4 L15.6 5.4 L15.6 18.6 L8 18.6 Z' },
    { d: 'M3.6 9.6 C3 6.4 4.6 4 7.6 3.6 M18.4 14.4 C19 17.6 17.4 20 14.4 20.4 M5.8 2 L7.8 3.6 L6.2 5.6 M16.2 22 L14.2 20.4 L15.8 18.4', accent: true, w: 1.6 },
  ],
  /** information / a tip */
  info: [{ d: 'M12 3 C17 3 21 7 21 12 C21 17 17 21 12 21 C7 21 3 17 3 12 C3 7 7 3 11.4 3.1 M12 10.6 L12 16.8 M12 7.2 L12 7.4' }],
  /** a speedometer (the speed readout, graphics) */
  gauge: [{ d: 'M3.6 16.4 C3.6 11.6 7.4 7.6 12 7.6 C16.6 7.6 20.4 11.6 20.4 16.4 M12 16.4 L16.4 10.6 M5.8 18.6 L18.2 18.6' }],
  /** a paint brush (graphics quality) */
  brush: [
    { d: 'M20.4 3.6 L11.4 13.4 M10.4 12.2 L12.6 14.4' },
    { d: 'M9.2 13.6 C6.6 13.4 5 15.2 5 17.4 C5 18.8 4.2 19.8 3.2 20.4 C6.6 21.4 11 20.2 11 16 Z', fill: true, accent: true },
  ],
  /** a hand waving hi (onboarding, friendly messages) */
  wave: [{ d: 'M7.6 20 C5 17.4 4 14.4 4.6 12 L6 9.4 C6.6 8.4 8 8.8 7.8 10 L7.4 12.4 L9.6 4.8 C10 3.6 11.8 3.8 11.6 5.2 L10.6 10 L13 3.6 C13.4 2.4 15.2 2.8 15 4.2 L13.4 10.2 L15.8 5.4 C16.4 4.2 18 4.8 17.6 6.2 L15.4 12.4 C16.2 11.6 17.8 11.6 18.2 12.6 C18.4 13.4 17.2 14.6 16.4 16 C15.2 18.2 13.6 20.2 11.4 21 Z' }],
};

export type IconName = keyof typeof I;
export const ICON_NAMES = Object.keys(I) as IconName[];

function wobble(d: string, r: () => number, amt: number): string {
  return d.replace(/(-?\d*\.?\d+) (-?\d*\.?\d+)/g, (_m, x: string, y: string) => `${n2(Number(x) + sym(r) * amt)} ${n2(Number(y) + sym(r) * amt)}`);
}

const cache = new Map<string, string>();

export interface IconOpts {
  /** CSS px; default 1em via CSS */
  size?: number;
  /** accessible name; without it the icon is decorative */
  title?: string;
  class?: string;
  /** stroke weight multiplier */
  weight?: number;
}

/** An icon as an SVG string. */
export function iconSVG(name: IconName, o: IconOpts = {}): string {
  const key = `${name}|${o.size ?? ''}|${o.title ?? ''}|${o.class ?? ''}|${o.weight ?? 1}`;
  const hit = cache.get(key);
  if (hit) return hit;
  const r = rng(hashStr(`icon:${name}`));
  const k = o.weight ?? 1;
  const parts = I[name].map((p) => {
    const d = wobble(p.d, r, 0.22);
    const col = p.accent ? 'var(--ico-accent, currentColor)' : 'currentColor';
    const w = n2((p.w ?? 2.5) * k * (p.w ? 1.15 : 1));
    return p.fill
      ? `<path d="${d}" fill="${col}" stroke="${col}" stroke-width="${w}" stroke-linejoin="round"/>`
      : `<path d="${d}" fill="none" stroke="${col}" stroke-width="${w}" stroke-linecap="round" stroke-linejoin="round"/>`;
  }).join('');
  const sz = o.size ? ` width="${o.size}" height="${o.size}"` : '';
  const a11y = o.title ? ` role="img" aria-label="${o.title.replace(/"/g, '&quot;')}"` : ' aria-hidden="true"';
  const svg = `<svg class="ico ico-${name}${o.class ? ` ${o.class}` : ''}" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"${sz}${a11y}>${parts}</svg>`;
  cache.set(key, svg);
  return svg;
}

const tpl = typeof document !== 'undefined' ? document.createElement('template') : null;

/** An icon as an inline <svg> element. */
export function icon(name: IconName, o: IconOpts = {}): SVGSVGElement {
  tpl!.innerHTML = iconSVG(name, o);
  return tpl!.content.firstElementChild as SVGSVGElement;
}

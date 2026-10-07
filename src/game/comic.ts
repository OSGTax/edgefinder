import { audio } from '../audio';
import type { SfxName } from '../audio/types';
import { comicPop, type BurstShape } from '../ui/look';

// The comic pop-ups: which word, for which moment, and when. The burst itself
// is drawn by the look kit (`comicPop`). Big plays only; words never repeat
// back to back, and each moment cycles through its own short list.

export type Moment =
  | 'crush' | 'homer' | 'kLooking' | 'kSwinging' | 'snag' | 'splash' | 'fence'
  | 'double' | 'triple' | 'doublePlay' | 'oops' | 'hbp' | 'scores' | 'dog' | 'special';

interface MomentStyle {
  words: string[];
  burst: BurstShape;
  /** outer, inner, letters */
  colors: [string, string, string];
  size: number;
}

/** the Sound helper's stinger for each pop-up (it lands with the burst); anything else gets the cork pop */
const STING: Partial<Record<Moment, SfxName>> = {
  crush: 'stingThwack', homer: 'stingSeeYa', kLooking: 'stingSitDown', kSwinging: 'stingWhiff',
  snag: 'stingSnag', splash: 'stingSploosh', fence: 'stingBonk',
};

const SUN = 'var(--sunshine)', CREAM = '#fff1b8', TOMATO = 'var(--tomato)', POSTER = 'var(--poster)';

const STYLE: Record<Moment, MomentStyle> = {
  crush: { words: ['THWACK!', 'KER-RACK!', 'SMACKED!'], burst: 'star', colors: [SUN, CREAM, TOMATO], size: 1.05 },
  homer: { words: ['SEE YA!', 'OUTTA HERE!', 'BYE-BYE!', 'GOING, GONE!'], burst: 'star', colors: [TOMATO, SUN, POSTER], size: 1.3 },
  kLooking: { words: ['SIT DOWN!', 'FROZEN!', 'CAUGHT LOOKING!'], burst: 'jagged', colors: ['#7cc8f0', '#d8f0ff', POSTER], size: 1 },
  kSwinging: { words: ['WHIFF!', 'WHOOSH!', 'FANNED!'], burst: 'jagged', colors: [TOMATO, SUN, POSTER], size: 1 },
  snag: { words: ['SNAG!', 'GOTCHA!', 'YOINK!'], burst: 'star', colors: ['var(--grass-light)', '#e3f6c8', 'var(--marker-green)'], size: 1 },
  splash: { words: ['SPLOOSH!', 'KER-SPLASH!'], burst: 'cloud', colors: ['#7cc8f0', '#d8f0ff', POSTER], size: 1.15 },
  fence: { words: ['BONK!', 'CLANG!', 'THUNK!'], burst: 'jagged', colors: [SUN, CREAM, TOMATO], size: 0.9 },
  double: { words: ['TWO BAGS!', 'ZIP!'], burst: 'star', colors: ['var(--grass-light)', '#e3f6c8', TOMATO], size: 0.95 },
  triple: { words: ['THREE BAGS!', 'ZOOM!'], burst: 'star', colors: ['var(--grass-light)', '#e3f6c8', TOMATO], size: 1.05 },
  doublePlay: { words: ['TWO FOR ONE!', 'DOUBLE PLAY!'], burst: 'jagged', colors: [TOMATO, SUN, POSTER], size: 1 },
  oops: { words: ['OOPS!', 'WHOOPS!', 'BUTTERFINGERS!'], burst: 'cloud', colors: ['#ffb36b', '#ffe0bd', TOMATO], size: 0.9 },
  hbp: { words: ['OUCH!', 'YEOWCH!'], burst: 'jagged', colors: [TOMATO, SUN, POSTER], size: 0.95 },
  scores: { words: ['SCORES!', 'HOME FREE!'], burst: 'star', colors: [SUN, CREAM, 'var(--marker-green)'], size: 0.9 },
  dog: { words: ['WOOF!', 'ARF!'], burst: 'cloud', colors: [POSTER, '#fff', 'var(--marker)'], size: 0.8 },
  special: { words: ['SPECIAL!'], burst: 'star', colors: ['#c39bff', '#ead9ff', POSTER], size: 1.05 },
};

export class ComicPops {
  private last = '';
  private used = new Map<Moment, number>();
  private lastT = -9;
  private lastSize = 0;

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

  /** Show a pop-up. `now` is game time; a smaller moment won't stomp a bigger, fresher one. */
  show(m: Moment, now: number, text?: string) {
    const st = STYLE[m];
    if (now - this.lastT < 0.6 && st.size < this.lastSize) return;
    this.lastT = now;
    this.lastSize = st.size;
    const word = text ?? this.word(m);
    // phones held sideways are short: keep the burst to about a third of the height
    const width = Math.round(Math.min(300, Math.max(190, window.innerHeight * 0.62)) * st.size);
    // dev: `window.__holdPops = true` keeps the last one up for screenshots in the slow headless browser
    const hold = import.meta.env.DEV && (window as unknown as { __holdPops?: boolean }).__holdPops;
    const el = comicPop(word, { burst: st.burst, colors: st.colors, width, animate: !hold, seed: `${word}|${m}` });
    while (this.host.firstChild) this.host.removeChild(this.host.firstChild);
    this.host.appendChild(el);
    audio.play(STING[m] ?? 'pop');
  }
}

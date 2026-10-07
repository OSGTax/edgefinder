/**
 * #sounds: every sound effect, voice, tune and the backyard ambience on one
 * page, to audition them on any device (works in the built game too: it's a
 * separate little chunk). Tap anything; the first tap turns the audio on.
 *
 * window.__sounds exposes the audio module and a level probe for automated checks.
 */
import { MUSIC_TRACKS, SFX_NAMES, audio } from '../audio';
import { outputProbe } from '../audio/context';
import { KIDS } from '../data/kids';

const LINES: Array<[string, string]> = [
  ['Chet', 'Bases loaded, and Kaboom steps in. I can\'t look. I\'m looking.'],
  ['Dottie', 'Back in my big-league days, I batted ninth. Ninth is the cleanup spot for winners.'],
  ['mrsMendoza', 'KAI! Aim LEFT, sweetie!'],
  ['mrMendoza', 'Burgers are almost ready!'],
];

export function soundBoard(root: HTMLElement): void {
  document.title = 'Sound board';
  const css = document.createElement('style');
  css.textContent = `
    .sb { font: 15px/1.4 system-ui, sans-serif; color: #2b2118; background: #f4ead6; min-height: 100vh;
      padding: 16px max(16px, env(safe-area-inset-right)) 40px max(16px, env(safe-area-inset-left)); box-sizing: border-box; }
    .sb h1 { font-size: 22px; margin: 0 0 4px; } .sb h2 { font-size: 16px; margin: 20px 0 8px; }
    .sb p { margin: 0 0 8px; color: #6b5a48; }
    .sb .row { display: flex; flex-wrap: wrap; gap: 8px; }
    .sb button { min-height: 48px; padding: 0 14px; border: 2px solid #2b2118; border-radius: 10px; background: #fffaf0;
      font: inherit; color: inherit; touch-action: manipulation; }
    .sb button:active { background: #ffe08a; }
    .sb label { display: flex; align-items: center; gap: 10px; margin: 6px 0; }
    .sb .who { display: block; font-size: 12px; color: #6b5a48; }`;
  document.head.appendChild(css);
  const el = (tag: string, props: Record<string, unknown> = {}, ...kids: Array<Node | string>) => {
    const n = document.createElement(tag);
    Object.assign(n, props);
    n.append(...kids);
    return n;
  };
  let intensity = 0.8;
  const btn = (label: string, on: () => void, sub?: string) =>
    el('button', { onclick: () => { audio.unlock(); on(); } }, label, ...(sub ? [el('span', { className: 'who' }, sub)] : []));

  const slider = el('input', { type: 'range', min: '0', max: '1', step: '0.05', value: String(intensity), oninput: (e: Event) => { intensity = Number((e.target as HTMLInputElement).value); } });
  const amb = el('input', { type: 'checkbox', onchange: (e: Event) => { audio.unlock(); audio.setAmbience((e.target as HTMLInputElement).checked); } });

  root.replaceChildren(el('div', { className: 'sb' },
    el('h1', {}, 'Grass Stain League: sound board'),
    el('p', {}, 'Everything here is made in code, live. Tap to hear it.'),
    el('h2', {}, 'Sound effects'),
    el('label', {}, 'How big a moment', slider),
    el('div', { className: 'row' }, ...SFX_NAMES.map((n) => btn(n, () => audio.play(n, { intensity })))),
    el('h2', {}, 'Voices'),
    el('div', { className: 'row' },
      ...KIDS.map((k) => btn(k.nick, () => audio.speak(k.id, k.quips[Math.floor(Math.random() * k.quips.length)]), k.persona)),
      ...LINES.map(([who, text]) => btn(who, () => audio.speak(who, text, { duck: true })))),
    el('h2', {}, 'Music'),
    el('div', { className: 'row' }, ...MUSIC_TRACKS.map((t) => btn(t, () => audio.playMusic(t))), btn('stop', () => audio.stopMusic())),
    el('h2', {}, 'The backyard'),
    el('label', {}, amb, 'Ambience (grill, sprinkler, birds, mower, ice-cream truck... give it a few minutes)'),
  ));

  (window as unknown as { __sounds: unknown }).__sounds = { audio, SFX_NAMES, MUSIC_TRACKS, KIDS, LINES, outputProbe };
  (window as unknown as { __ready: boolean }).__ready = true;
}

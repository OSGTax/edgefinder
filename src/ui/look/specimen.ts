import { TEAMS } from '../../data/teams';
import { h } from '../dom';
import {
  bigMoment, button, chalkboard, clipboard, ICON_NAMES, icon, lettering, lowerThird, paper, pennant, rotateHint, sign, tape, tapeCorners, tradingCard,
} from '.';

// Dev-only: every piece of the kit on one page (#look). Not shipped.

export function specimen(root: HTMLElement) {
  document.body.style.overflow = 'auto';
  document.body.style.touchAction = 'auto';
  root.style.cssText = 'position:static;padding:16px;display:flex;flex-direction:column;gap:22px;align-items:flex-start';
  root.classList.add('lawn');
  const row = (...c: (Node | null)[]) => h('div', { style: 'display:flex;gap:18px;flex-wrap:wrap;align-items:center' }, ...c);
  const abc = 'ABCDEFGHIJKLM\nNOPQRSTUVWXYZ\n0123456789 !?.,\'"-:/&#+½()%';
  const [mud, com] = TEAMS;
  const photo = (c: string) => h('div', { style: `width:100%;height:100%;background:${c}` });
  root.append(
    sign([lettering('GRASS STAIN\nLEAGUE', { style: 'poster', size: 40, colors: ['var(--highlighter)', '#9fd26a'], seed: 'logo' })], { seed: 'spec-sign' }),
    row(
      paper([lettering(abc, { size: 18 })], { seed: 'a' }),
      paper([lettering(abc, { size: 18, style: 'poster', color: 'var(--marker-red)' })], { seed: 'b' }),
    ),
    row(
      chalkboard([lettering(abc, { size: 16, style: 'chalk' })]),
      paper([lettering('SPLASH\nDOUBLE!', { size: 30, style: 'brush', color: 'var(--marker-blue)' }), lettering('Calderón · Peña', { size: 20 })], { seed: 'c' }),
    ),
    paper([row(...ICON_NAMES.map((n) => h('span', { title: n, style: 'font-size:34px;display:inline-flex;flex-direction:column;align-items:center;--ico-accent:var(--marker-red)' }, icon(n), h('small', { style: 'font-size:10px' }, n))))], { seed: 'icons' }),
    row(
      tape('Meet the kids'), tape('Sound', { tone: 'blue' }), tape('Pitching', { tone: 'red' }),
      button('Play ball!', () => {}, { icon: 'ball', kind: 'go', size: 'big' }),
      button('Settings', () => {}, { icon: 'clipboard' }),
      h('button', { class: 'btn small on' }, 'Pro'),
      h('button', { class: 'btn small ghost' }, 'Rookie'),
      h('div', { class: 'choices' }, h('button', { class: 'choice on' }, '3 innings'), h('button', { class: 'choice' }, '6 innings')),
    ),
    row(
      h('div', { style: 'width:280px' }, pennant(mud)),
      h('div', { style: 'width:280px' }, pennant(com)),
    ),
    row(
      tradingCard({ photo: photo(mud.colors.secondary), name: 'Mudpie', persona: 'The Retired Plumber', number: 7, team: mud, stats: [['Power', 10], ['Contact', 6]] }),
      tradingCard({ photo: photo(com.colors.secondary), name: 'Pepper', persona: 'The Weather Lady', number: 12, team: com, stats: [['Pitching', 9], ['Control', 8], ['Speed', 5]] }),
      tapeCorners(paper([h('p', null, 'An index card with tape corners.')], { seed: 'tc' }), 'tc'),
    ),
    row(
      clipboard([h('p', null, 'Sound effects ———'), h('p', null, 'Music ———')], { title: 'SETTINGS' }),
      lowerThird({ who: 'Chet Valentine', role: 'play-by-play', text: 'And that ball is in the POOL, folks. Mr. Mendoza is reaching for the skimmer.', tone: 'chet' }),
      lowerThird({ who: 'Dottie Fairweather', role: 'color', text: 'Back in my tee-ball days we called that a splashdown.', tone: 'dottie' }),
    ),
    row(bigMoment('HOME RUN!', { sub: 'Mudpie, to the moon' }), rotateHint()),
  );
  (window as unknown as { __ready: boolean }).__ready = true;
}

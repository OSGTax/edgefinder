import { audio } from '../audio';
import { kid } from '../data/kids';
import { TEAMS } from '../data/teams';
import { yard } from '../data/yards';
import { SPECIAL_INFO, TRAIT_LABELS, type Kid, type Team, type Traits } from '../data/types';
import { autoLineup } from '../sim/lineup';
import { buildField } from '../sim/field';
import { Match, type MatchConfig } from '../sim/match';
import { PITCHES } from '../sim/pitching';
import type { Difficulty } from '../sim/types';
import { gfxPrefs, qualitySetting, setGfxPrefs, setQuality, softwareTip, TIER_LABEL, type QualityName } from '../gfx/quality';
import { World } from '../game/world';
import { nextPaint, runPaced, type Step } from '../engine/steps';
import { Director } from '../game/director';
import { PortraitStudio } from '../game/portraits';
import { GameScreen, replayCoach } from '../game/screen';
import { jerseyNumber } from '../kid3d/uniform';
import type { Expression } from '../kid3d/face';
import { shell } from '../shell';
import { clear, h } from './dom';
import {
  flipCard, icon, label, lettering, lowerThird, panel, paper, pennant, sign, teamPatch, tradingCard, type IconName,
} from './look';
import { saveSettings, settings } from './settings';

// The app around the game: build the yard and the kids once, run a CPU game
// behind the title screen, and hand the same world to each game the player
// starts. The menus are Saturday-morning cartoon: a lettered sign, pennants,
// a box of trading cards, and clean cartoon panels for the rules and settings.

export function startApp(root: HTMLElement) {
  new App(root);
}

const tap = (sound: 'uiTap' | 'uiSelect' | 'uiBack', fn: () => void) => () => { audio.play(sound); fn(); };

const DIFFICULTY: [Difficulty, string, string][] = [
  ['rookie', 'Rookie', 'Slow pitches and a big sweet spot. Good for learning.'],
  ['pro', 'Pro', 'A fair fight.'],
  ['allstar', 'All-Star', 'These kids are not messing around.'],
];

/** Channel 4½ is on the air behind the title screen. */
const ON_AIR: { who: 'chet' | 'dottie'; text: string }[] = [
  { who: 'chet', text: 'Live from Pool Party Paradise, out back of the Mendozas\'. First pitch as soon as Kai finds his other shoe.' },
  { who: 'dottie', text: 'A reminder from Mrs. Mendoza: no cleats on the patio, and the pool is not a base.' },
  { who: 'chet', text: 'Today\'s game is brought to you by the lemonade stand at the end of the driveway. Twenty-five cents, exact change.' },
  { who: 'dottie', text: 'Weather: hot. Grass: freshly mowed. Knees: about to be green.' },
  { who: 'chet', text: 'Channel 4½ is the only station in Maple Hollow, and we thank our viewer.' },
  { who: 'dottie', text: 'Back in my tee-ball days we played through worse. Much worse. Ask me about it.' },
];

class App {
  private world!: World;
  private studio!: PortraitStudio;
  private canvas = h('canvas', { class: 'scene' });
  private menus = h('div', { class: 'menus' });
  /** the game in progress (kept so it isn't collected mid-play) */
  game: GameScreen | null = null;
  private attract: Attract | null = null;
  private yourTeam: Team = TEAMS[0];
  private innings = 3;
  private boxTeam: Team = TEAMS[0];
  private airTimer = 0;
  private airIdx = Math.floor(Math.random() * ON_AIR.length);
  /** a one-time hint about this device (no graphics card), shown under the title menu */
  private tip: string | null = null;

  constructor(private root: HTMLElement) {
    root.appendChild(this.canvas);
    const loadMsg = h('div', { class: 'loading-msg typed' }, 'Unrolling the tarp…');
    const chalked = h('i', { class: 'chalked' });
    const rolling = h('span', { class: 'rolling' }, icon('ball'));
    const loading = h('div', { class: 'loading lawn' },
      sign([lettering('GRASS STAIN\nLEAGUE', { style: 'comic', size: 40, colors: ['var(--sunshine)', '#9ad36a'], seed: 'logo' })], { seed: 'loading-sign', class: 'loading-sign' }),
      h('div', { class: 'loading-line', 'aria-hidden': 'true' }, icon('plate', { class: 'ico-plate' }), chalked, rolling, icon('base', { class: 'ico-base' })),
      loadMsg,
      h('div', { class: 'loading-sub' }, 'Out back of the Mendozas\' house, eighteen very serious kids are getting ready.'));
    root.appendChild(loading);
    root.appendChild(this.menus);
    window.addEventListener('resize', () => this.resize());
    audio.setSfxVolume(settings.sfx);
    audio.setMusicVolume(settings.music);
    audio.setVoiceVolume(settings.voices);
    const unlock = () => audio.unlock();
    window.addEventListener('pointerdown', unlock);
    window.addEventListener('keydown', unlock);
    shell.onUpdate(() => this.updateReady());
    requestAnimationFrame(() => setTimeout(async () => {
      const t0 = performance.now();
      const show = (s: Step) => {
        loadMsg.textContent = `${s.msg}…`;
        chalked.style.width = `calc((100% - 44px) * ${s.done.toFixed(3)})`;
        rolling.style.left = `calc(10px + (100% - 44px) * ${s.done.toFixed(3)})`;
      };
      if (import.meta.env.DEV && location.hash.includes('loadhold')) { show({ done: 0.45, msg: 'Inflating the pool flamingo' }); (window as unknown as { __menus: boolean }).__menus = true; return; }
      this.world = new World(this.canvas, buildField(yard('poolparty')), [TEAMS[0], TEAMS[1]]);
      await runPaced(this.world.build(), show);
      show({ done: 0.97, msg: 'Chalking the baselines' });
      await nextPaint();
      await this.world.warmUp();
      this.studio = new PortraitStudio(this.world.renderer, this.world.scene.environment);
      if (import.meta.env.DEV) console.debug(`[app] world built in ${Math.round(performance.now() - t0)} ms`);
      this.resize();
      loading.classList.add('done');
      setTimeout(() => loading.remove(), 400);
      // no graphics card: a one-line tip; a lost graphics context: a short note while it recovers
      const tip = softwareTip();
      if (tip) this.tip = tip;
      this.world.onContextChange = (lost) => { if (lost) this.toast('The graphics took a quick nap. Waking them up…', 4000); };
      const hash = new URLSearchParams(location.hash.slice(1));
      if (import.meta.env.DEV && hash.has('play')) this.startGame(hash.get('play') === 'comets' ? TEAMS[1] : TEAMS[0], hash.has('cpu'));
      else if (import.meta.env.DEV && hash.has('menu')) this.devMenu(hash.get('menu')!, hash);
      else this.title();
      (window as unknown as { __menus: boolean }).__menus = true;
    }, 30));
  }

  /** A short note at the bottom of the screen that fades away (tap to dismiss); clear of the game's banners. */
  private toast(text: string, ms: number) {
    const el = h('div', { class: 'toast', role: 'status', onclick: () => el.remove() }, icon('info'), h('span', null, text));
    this.root.appendChild(el);
    setTimeout(() => { el.classList.add('gone'); setTimeout(() => el.remove(), 450); }, ms);
  }

  private resize() {
    this.world?.resize(window.innerWidth, window.innerHeight);
  }

  private portrait(k: Kid, expr: Expression = 'happy', size = 140): HTMLImageElement {
    const team = this.teamOf(k);
    const img = h('img', { class: 'portrait', width: size / 2, height: size / 2, alt: k.nick, draggable: 'false' });
    const a = this.world.actors.get(k.id);
    if (a) this.studio.into(img, a.model, team, expr, size);
    return img;
  }

  private teamOf(k: Kid): Team {
    return TEAMS.find((t) => t.roster.includes(k.id))!;
  }

  private startAttract() {
    if (!this.attract) this.attract = new Attract(this.world, this.studio);
  }

  private stopAttract() {
    this.attract?.stop();
    this.attract = null;
  }

  private show(el: HTMLElement, cls = '') {
    clearInterval(this.airTimer);
    clear(this.menus);
    this.menus.className = `menus${cls ? ` ${cls}` : ''}`;
    this.menus.appendChild(el);
    this.menus.scrollTop = 0;
  }

  /** A screen's header: back button and the title in comic lettering. */
  private head(title: string, back: () => void, ico?: IconName) {
    return h('div', { class: 'screen-head' },
      h('button', { class: 'btn small ghost back', 'aria-label': 'Back', onclick: tap('uiBack', back) }, icon('back')),
      h('h1', { class: 'head-title' }, ico ? h('span', { class: 'head-ico' }, icon(ico)) : null, lettering(title, { style: 'comic', size: 24, color: 'var(--poster)', seed: `head-${title}` })),
      h('div', { class: 'head-spacer' }));
  }

  // ───────────────────────────────────────────────────────────── title

  private title() {
    this.startAttract();
    audio.playMusic('title');
    const item = (label: string, ico: IconName, fn: () => void, cls = '') =>
      h('button', { class: `btn menu-item ${cls}`, onclick: tap('uiTap', fn) }, icon(ico), h('span', null, label));
    const air = h('div', { class: 'on-air' });
    const nextLine = () => {
      const l = ON_AIR[this.airIdx++ % ON_AIR.length];
      clear(air);
      air.appendChild(lowerThird(l.who === 'chet'
        ? { who: 'Chet Valentine', role: 'Channel 4½', text: l.text, tone: 'chet' }
        : { who: 'Dottie Fairweather', role: 'Channel 4½', text: l.text, tone: 'dottie' }));
    };
    nextLine();
    const logo = sign([
      lettering('GRASS STAIN\nLEAGUE', { style: 'comic', size: 44, colors: ['var(--sunshine)', '#9ad36a'], seed: 'logo' }),
      label('Backyard baseball. Very serious kids.', { tilt: 1.5, seed: 'tagline' }),
    ], { seed: 'title-sign', class: 'logo-sign' });
    const notes = h('div', { class: 'title-notes' });
    this.show(h('div', { class: 'screen title' },
      logo,
      h('nav', { class: 'title-menu' },
        item('Play ball!', 'ball', () => this.pickTeam(), 'go big'),
        item('Meet the kids', 'cards', () => this.roster()),
        h('div', { class: 'menu-pair' },
          item('How to play', 'whistle', () => this.howTo()),
          item('Settings', 'clipboard', () => this.settingsScreen())),
        notes),
      air), 'on-title');
    this.airTimer = window.setInterval(nextLine, 9000);
    this.titleNotes(notes);
  }

  /** Little notes under the title menu: install hints, a fresh version. */
  private titleNotes(el: HTMLElement) {
    clear(el);
    if (shell.updateWaiting) {
      el.appendChild(h('button', { class: 'label note-btn', onclick: () => location.reload() }, icon('replay'), 'A fresh version is ready. Tap to reload.'));
      return;
    }
    if (this.tip) {
      el.appendChild(h('div', { class: 'label note' }, icon('info'), h('span', null, this.tip),
        h('button', { class: 'note-x', 'aria-label': 'Hide', onclick: () => { this.tip = null; this.titleNotes(el); } }, icon('close'))));
      return;
    }
    const hint = shell.installHint();
    if (hint === 'ios') {
      el.appendChild(h('div', { class: 'label note' }, icon('share'), h('span', null, 'Play it like an app: tap Share, then “Add to Home Screen”.'),
        h('button', { class: 'note-x', 'aria-label': 'Hide', onclick: () => { shell.dismissInstall(); clear(el); } }, icon('close'))));
    } else if (hint === 'prompt') {
      el.appendChild(h('button', { class: 'label note-btn', onclick: () => shell.install().then(() => this.titleNotes(el)) }, icon('add'), 'Put it on your home screen'));
    }
  }

  private updateReady() {
    const notes = this.menus.querySelector<HTMLElement>('.title-notes');
    if (notes) this.titleNotes(notes);
  }

  // ───────────────────────────────────────────────────────────── team pick

  private pickTeam() {
    const flag = (t: Team) => {
      const you = this.yourTeam.id === t.id;
      return h('button', {
        class: `pennant-pick${you ? ' on' : ''}`, 'aria-pressed': String(you), 'aria-label': `Play as the ${t.street} ${t.name}`,
        onclick: tap('uiSelect', () => { this.yourTeam = t; this.pickTeam(); }),
      },
        h('i', { class: 'pin' }),
        pennant(t),
        you ? h('span', { class: 'sticker', style: '--tilt:-8deg' }, 'Us!') : null,
        h('span', { class: 'pick-note typed' }, t.id === 'comets' ? 'Home team. Kai\'s backyard. You pitch first.' : 'Visitors from Maple Street. You bat first.'),
        h('span', { class: 'pick-kids' }, ...t.roster.map((id) => this.portrait(kid(id), 'happy', 96))));
    };
    const choice = <T,>(v: T, cur: T, label: string, set: (v: T) => void) =>
      h('button', { class: `choice${v === cur ? ' on' : ''}`, 'aria-pressed': String(v === cur), onclick: tap('uiTap', () => { set(v); this.pickTeam(); }) }, label);
    const diff = DIFFICULTY.find(([d]) => d === settings.difficulty) ?? DIFFICULTY[0];
    this.show(h('div', { class: 'screen pick' },
      this.head('Pick your team', () => this.title(), 'pennant'),
      h('div', { class: 'clothesline' }, flag(TEAMS[0]), flag(TEAMS[1])),
      h('div', { class: 'pick-foot' },
        paper([
          h('div', { class: 'opt-row' }, h('span', { class: 'opt-label' }, 'Innings'),
            h('div', { class: 'choices' }, choice(3, this.innings, '3, quick', (v) => { this.innings = v; }), choice(6, this.innings, '6, the full game', (v) => { this.innings = v; }))),
          h('div', { class: 'opt-row' }, h('span', { class: 'opt-label' }, 'How hard'),
            h('div', { class: 'choices' }, ...DIFFICULTY.map(([d, l]) => choice(d, settings.difficulty, l, (v) => { settings.difficulty = v; saveSettings(); })))),
          h('div', { class: 'opt-note' }, diff[2]),
        ], { seed: 'pick-opts', class: 'pick-opts' }),
        h('button', { class: 'btn go big play', onclick: tap('uiSelect', () => this.startGame(this.yourTeam)) },
          icon('ball'), h('span', null, 'Play ball!'))),
    ), 'inner');
  }

  private startGame(you: Team, cpuOnly = false) {
    this.stopAttract();
    clearInterval(this.airTimer);
    this.menus.classList.add('hidden');
    clear(this.menus);
    shell.enterGame();
    const [away, home] = [TEAMS[0], TEAMS[1]];
    const cfg: MatchConfig = {
      away: { team: away, lineup: autoLineup(away.roster.map(kid)), human: !cpuOnly && you.id === away.id },
      home: { team: home, lineup: autoLineup(home.roster.map(kid)), human: !cpuOnly && you.id === home.id },
      yard: yard('poolparty'), innings: this.innings, seed: Math.floor(Math.random() * 1e6), difficulty: settings.difficulty,
      mercy: 10,
    };
    this.game = new GameScreen(this.root, this.world, this.studio, {
      cfg,
      onExit: (_m, again) => {
        this.game = null;
        if (again) this.startGame(you);
        else this.title();
      },
    });
  }

  // ───────────────────────────────────────────────────────────── meet the kids

  /** The two numbers that say the most about a kid: their best traits. */
  private bestTraits(k: Kid): [string, number][] {
    const keys = Object.keys(TRAIT_LABELS) as (keyof Traits)[];
    return keys.map((key) => [TRAIT_LABELS[key], k.traits[key]] as [string, number])
      .sort((a, b) => b[1] - a[1]).slice(0, 2);
  }

  private card(k: Kid, size: 'box' | 'big') {
    const t = this.teamOf(k);
    return tradingCard({
      photo: this.portrait(k, 'happy', size === 'big' ? 360 : 200),
      name: k.nick, persona: k.persona, number: jerseyNumber(k), team: t,
      stats: this.bestTraits(k), corner: h('span', { class: 'tcard-patch' }, teamPatch(t, size === 'big' ? 34 : 24)),
      back: size === 'big' ? this.cardBack(k) : undefined,
      class: size === 'big' ? 'big' : 'boxed', seed: k.id,
    });
  }

  private roster() {
    const t = this.boxTeam;
    const other = TEAMS.find((o) => o.id !== t.id)!;
    const tab = (tm: Team) => h('button', {
      class: `box-tab${tm.id === t.id ? ' on' : ''}`, style: `--team:${tm.colors.primary};--team2:${tm.colors.secondary}`,
      'aria-pressed': String(tm.id === t.id), onclick: tap('uiTap', () => { this.boxTeam = tm; this.roster(); }),
    }, teamPatch(tm, 26), h('span', null, tm.name));
    this.show(h('div', { class: 'screen box-screen' },
      this.head('Meet the kids', () => this.title(), 'cards'),
      h('div', { class: 'shoebox', style: `--team:${t.colors.primary};--team2:${t.colors.secondary}` },
        h('div', { class: 'box-tabs' }, tab(TEAMS[0]), tab(TEAMS[1])),
        label(`${t.street} ${t.name}. Nine kids. Tap one to meet them.`, { class: 'box-label', tilt: -1 }),
        h('div', { class: 'box-cards' }, ...t.roster.map((id) => {
          const k = kid(id);
          const c = this.card(k, 'box');
          return h('button', { class: 'card-btn', 'aria-label': `${k.first} "${k.nick}" ${k.last}, ${k.persona}`, onclick: tap('uiSelect', () => this.kidDetail(k)) }, c);
        })),
        h('div', { class: 'box-more' }, h('button', { class: 'btn small ghost', onclick: tap('uiTap', () => { this.boxTeam = other; this.roster(); }) }, `The ${other.name}`, icon('forward'))))), 'inner');
  }

  private cardBack(k: Kid) {
    const keys = Object.keys(TRAIT_LABELS) as (keyof Traits)[];
    const hand = (x: string) => (x === 'S' ? 'both' : x === 'L' ? 'left' : 'right');
    const sp = SPECIAL_INFO[k.special];
    return h('div', { class: 'back-body' },
      h('div', { class: 'back-head' },
        h('div', null, h('div', { class: 'back-name' }, `${k.first} “${k.nick}” ${k.last}`),
          h('div', { class: 'typed back-meta' }, `Age ${k.age} · bats ${hand(k.bats)} · throws ${hand(k.throws)}`)),
        h('span', { class: 'sticker', style: '--tilt:7deg' }, `#${jerseyNumber(k)}`)),
      h('div', { class: 'back-stats' }, ...keys.map((key) => h('div', { class: 'back-stat' }, h('span', null, TRAIT_LABELS[key]), h('i'), h('b', null, String(k.traits[key]))))),
      h('p', { class: 'back-bio' }, k.bio),
      h('div', { class: 'back-row' }, icon('bolt'), h('b', null, `${sp.label}.`), ` ${sp.blurb}`),
      h('div', { class: 'back-row' }, icon('ball'), h('b', null, 'Throws:'), ` ${k.pitches.map((p) => PITCHES[p].label.toLowerCase()).join(', ')}`),
      h('div', { class: 'back-quips' }, h('b', null, 'Likes to say'), ...k.quips.map((q) => h('span', { class: 'typed' }, `“${q}”`))),
      h('div', { class: 'back-snaps' }, this.portrait(k, 'yell', 140), this.portrait(k, 'smug', 140)));
  }

  private kidDetail(k: Kid) {
    const card = this.card(k, 'big');
    const close = () => { overlay.remove(); };
    const flip = () => { audio.play('uiTap'); flipCard(card); flipBtn.lastElementChild!.textContent = card.classList.contains('flipped') ? 'Front' : 'Flip it over'; };
    const flipBtn = h('button', { class: 'btn', onclick: flip }, icon('flip'), h('span', null, 'Flip it over'));
    card.addEventListener('click', (e) => { if (!(e.target as HTMLElement).closest('.tcard-back')) flip(); });
    const roster = this.teamOf(k).roster;
    const step = (d: number) => () => { audio.play('uiTap'); overlay.remove(); this.kidDetail(kid(roster[(roster.indexOf(k.id) + d + roster.length) % roster.length])); };
    const overlay = h('div', { class: 'overlay kid-detail', onclick: (e: Event) => { if (e.target === overlay) close(); } },
      h('div', { class: 'detail-wrap' },
        h('button', { class: 'btn small ghost step', 'aria-label': 'Previous kid', onclick: step(-1) }, icon('back')),
        card,
        h('div', { class: 'detail-side' },
          flipBtn,
          h('button', { class: 'btn ghost', onclick: tap('uiBack', close) }, icon('cards'), h('span', null, 'Back in the box'))),
        h('button', { class: 'btn small ghost step', 'aria-label': 'Next kid', onclick: step(1) }, icon('forward'))));
    this.root.appendChild(overlay);
  }

  // ───────────────────────────────────────────────────────────── how to play

  private howTo() {
    const sec = (title: string, ico: IconName, items: (string | Node)[][]) => h('section', { class: 'how-sec' },
      h('h3', { class: 'how-head' }, h('span', { class: 'head-ico' }, icon(ico)), lettering(title, { style: 'comic', size: 17, color: 'var(--sunshine)', seed: `how-${title}` })),
      h('ul', null, ...items.map((c) => h('li', null, ...c))));
    const b = (s: string) => h('b', null, s);
    const rules = yard('poolparty').rules;
    this.show(h('div', { class: 'screen how-screen' },
      this.head('How to play', () => this.title(), 'whistle'),
      panel([
        h('div', { class: 'how-cols' },
          sec('BATTING', 'bat', [
            ['Tap ', b('SWING'), ', or anywhere on the right side, just as the ball gets to the plate. Early pulls it, late pushes it the other way.'],
            ['On Rookie the circle aims itself. Drag on the left side to steer it.'],
            [b('Power'), ' hits it farther with a smaller sweet spot. ', b('Bunt'), ' squares around.'],
            ['Fill the hype meter and your kid\'s special move lights up.'],
          ]),
          sec('PITCHING', 'ball', [
            ['Pick a pitch on the left. Drag anywhere to move the mitt.'],
            ['Tap ', b('THROW'), ' to start the meter, then tap again when the needle is in the green.'],
          ]),
          sec('FIELDING', 'glove', [
            ['Your kids chase the ball by themselves. Tap a base to throw there, or wait and your kid picks.'],
          ]),
          sec('RUNNING', 'forward', [
            ['Runners go on their own. ', b('GO!'), ' sends everybody, ', b('BACK'), ' sends them back.'],
          ]),
          sec('THE YARD', 'home', rules.map((r) => [r])),
          sec('ON A COMPUTER', 'info', [
            ['Mouse aims. Click or ', b('Space'), ' swings. ', b('P'), ' power, ', b('B'), ' bunt, ', b('S'), ' special.'],
            [b('1 2 3'), ' pick a pitch; ', b('Space'), ' or two clicks throw. ', b('1 2 3 H'), ' throw to a base. ', b('R'), ' run, ', b('F'), ' back.'],
          ])),
        h('div', { class: 'how-foot' },
          h('button', { class: 'btn', onclick: (e: Event) => {
            replayCoach(); audio.play('uiSelect');
            const btn = e.currentTarget as HTMLButtonElement;
            btn.disabled = true; btn.lastElementChild!.textContent = 'Coach is back for your next game';
          } }, icon('whistle'), h('span', null, 'Replay the coach')))], { tone: 'sky', class: 'how-board' }),
    ), 'inner');
  }

  // ───────────────────────────────────────────────────────────── settings

  private settingsScreen() {
    const row = (ico: IconName, label: string, control: Node, note?: string) =>
      h('div', { class: `set-row${(control as Element).classList.contains('check') ? '' : ' wide'}` }, h('div', { class: 'set-label' }, icon(ico), h('span', null, label), note ? h('small', null, note) : null), control);
    const slider = (v: number, set: (x: number) => void, name: string) =>
      h('input', { class: 'cap-slider', type: 'range', min: '0', max: '1', step: '0.05', value: String(v), 'aria-label': name, oninput: (e: Event) => { set(Number((e.target as HTMLInputElement).value)); saveSettings(); } });
    const check = (v: boolean, set: (x: boolean) => void, name: string) => {
      const box = h('button', { class: `check${v ? ' on' : ''}`, role: 'switch', 'aria-checked': String(v), 'aria-label': name }, icon('check'));
      box.onclick = () => { const on = !box.classList.contains('on'); box.classList.toggle('on', on); box.setAttribute('aria-checked', String(on)); set(on); saveSettings(); audio.play('uiTap'); };
      return box;
    };
    const choices = <T,>(cur: T, opts: [T, string][], set: (v: T) => void) => h('div', { class: 'choices' },
      ...opts.map(([v, l]) => h('button', { class: `choice${v === cur ? ' on' : ''}`, 'aria-pressed': String(v === cur), onclick: tap('uiTap', () => set(v)) }, l)));
    const q = qualitySetting();
    this.show(h('div', { class: 'screen set-screen' },
      this.head('Settings', () => this.title(), 'clipboard'),
      panel([
        row('sound', 'Sound effects', slider(settings.sfx, (x) => { settings.sfx = x; audio.setSfxVolume(x); }, 'Sound effects volume')),
        row('music', 'Music', slider(settings.music, (x) => { settings.music = x; audio.setMusicVolume(x); }, 'Music volume')),
        row('mic', 'Voices', slider(settings.voices, (x) => { settings.voices = x; audio.setVoiceVolume(x); }, 'Voices volume'), 'The kids and the announcers'),
        row('plate', 'Always show the strike zone', check(settings.showZone, (x) => { settings.showZone = x; }, 'Always show the strike zone')),
        row('tap', 'Aim help', choices<'auto' | 'on' | 'off'>(settings.aimAssist, [['auto', 'By difficulty'], ['on', 'On'], ['off', 'Off']], (v) => { settings.aimAssist = v; saveSettings(); this.settingsScreen(); })),
        row('tap', 'Buzz on big plays', check(settings.haptics, (x) => { settings.haptics = x; }, 'Buzz on big plays'), 'Phones that can vibrate'),
        row('fullscreen', 'Full screen on phones', check(settings.fullscreen, (x) => { settings.fullscreen = x; }, 'Full screen on phones')),
        row('brush', 'Graphics', choices<QualityName | 'auto'>(q, [['auto', 'Auto'], ['low', 'Fast'], ['medium', 'Balanced'], ['high', 'Beautiful']], (v) => { setQuality(v); location.reload(); }),
          `Changing this reloads the game. Right now: ${TIER_LABEL[this.world.tier]}${q === 'auto' ? ', picked for this device' : ''}.`),
        row('gauge', 'Battery saver', check(gfxPrefs().cap30, (x) => setGfxPrefs({ cap30: x }), 'Battery saver'), '30 frames a second, cooler phone'),
        row('info', 'Speed readout', check(gfxPrefs().readout, (x) => setGfxPrefs({ readout: x }), 'Speed readout'), 'Frames a second and the graphics chip, in a corner'),
      ], { title: 'STUFF YOU CAN CHANGE', class: 'set-board' }),
    ), 'inner');
  }

  /** Dev only: open a menu straight from the URL (#menu=pick|kids|kid&kid=bo|how|settings). */
  private devMenu(which: string, hash: URLSearchParams) {
    this.startAttract();
    if (which === 'pick') this.pickTeam();
    else if (which === 'kids') { if (hash.get('team') === 'comets') this.boxTeam = TEAMS[1]; this.roster(); }
    else if (which === 'kid') { this.roster(); this.kidDetail(kid(hash.get('kid') || 'bo')); if (hash.has('back')) setTimeout(() => this.root.querySelector('.tcard.big')?.classList.add('flipped'), 50); }
    else if (which === 'how') this.howTo();
    else if (which === 'settings') this.settingsScreen();
    else this.title();
  }
}

/** A CPU-vs-CPU game playing behind the title screen. */
class Attract {
  private match: Match;
  private director: Director;
  private raf = 0;
  private last = performance.now();
  private stopped = false;

  constructor(private world: World, private studio: PortraitStudio) {
    this.match = this.newMatch();
    this.director = new Director(world.camera);
    this.director.adopt(world.camera);
    this.director.forced = 'title';
    this.raf = requestAnimationFrame(this.frame);
  }

  private newMatch() {
    const [a, b] = Math.random() > 0.5 ? [TEAMS[0], TEAMS[1]] : [TEAMS[1], TEAMS[0]];
    return new Match({
      away: { team: a, lineup: autoLineup(a.roster.map(kid)), human: false },
      home: { team: b, lineup: autoLineup(b.roster.map(kid)), human: false },
      yard: yard('poolparty'), innings: 6, seed: Math.floor(Math.random() * 1e6), difficulty: 'pro',
    });
  }

  private frame = (now: number) => {
    if (this.stopped) return;
    const dt = Math.min(0.05, (now - this.last) / 1000);
    this.last = now;
    this.match.update(dt);
    this.match.events.length = 0;
    if (this.match.phase === 'over' && this.match.phaseT > 6) this.match = this.newMatch();
    this.world.sync(this.match, dt, { aim: null, aimColor: '#fff', aimRadius: 1, pitchAim: null, showZone: false, bases: null });
    this.director.update(this.match, dt, null);
    this.studio.update();
    this.world.render();
    this.world.adapt(dt);
    this.raf = requestAnimationFrame(this.frame);
  };

  stop() {
    this.stopped = true;
    cancelAnimationFrame(this.raf);
  }
}


import { audio } from '../audio';
import { kid } from '../data/kids';
import { TEAMS, teamName } from '../data/teams';
import { yard } from '../data/yards';
import { SPECIAL_INFO, TRAIT_LABELS, type Kid, type Team, type Traits } from '../data/types';
import { autoLineup } from '../sim/lineup';
import { buildField } from '../sim/field';
import { Match, type MatchConfig } from '../sim/match';
import { PITCHES } from '../sim/pitching';
import type { Difficulty } from '../sim/types';
import { qualitySetting, setQuality, type QualityName } from '../gfx/quality';
import { World } from '../game/world';
import { nextPaint, runPaced, type Step } from '../engine/steps';
import { Director } from '../game/director';
import { PortraitStudio } from '../game/portraits';
import { GameScreen, teamBadge } from '../game/screen';
import type { Expression } from '../kid3d/face';
import { clear, h } from './dom';
import { saveSettings, settings } from './settings';

// The demo app: build the yard and the kids once, run a CPU game behind the
// title screen, and hand the same world to each game the player starts.

export function startApp(root: HTMLElement) {
  new App(root);
}

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

  constructor(private root: HTMLElement) {
    root.appendChild(this.canvas);
    const loadMsg = h('div', null, 'Unrolling the tarp…');
    const loadBar = h('i');
    const loading = h('div', { class: 'loading' }, h('div', { class: 'loading-ball' }), loadMsg,
      h('div', { class: 'loading-bar' }, loadBar),
      h('div', { class: 'loading-sub' }, 'Building the Mendozas\' backyard and eighteen very serious kids'));
    root.appendChild(loading);
    root.appendChild(this.menus);
    window.addEventListener('resize', () => this.resize());
    audio.setSfxVolume(settings.sfx);
    audio.setMusicVolume(settings.music);
    const unlock = () => audio.unlock();
    window.addEventListener('pointerdown', unlock);
    window.addEventListener('keydown', unlock);
    requestAnimationFrame(() => setTimeout(async () => {
      const t0 = performance.now();
      const show = (s: Step) => { loadMsg.textContent = `${s.msg}…`; loadBar.style.width = `${Math.round(s.done * 100)}%`; };
      this.world = new World(this.canvas, buildField(yard('poolparty')), [TEAMS[0], TEAMS[1]]);
      await runPaced(this.world.build(), show);
      show({ done: 0.97, msg: 'Chalking the baselines' });
      await nextPaint();
      await this.world.warmUp();
      this.studio = new PortraitStudio(this.world.renderer, this.world.scene.environment);
      if (import.meta.env.DEV) console.debug(`[app] world built in ${Math.round(performance.now() - t0)} ms`);
      this.resize();
      loading.remove();
      const hash = new URLSearchParams(location.hash.slice(1));
      if (import.meta.env.DEV && hash.has('play')) this.startGame(hash.get('play') === 'comets' ? TEAMS[1] : TEAMS[0], hash.has('cpu'));
      else this.title();
    }, 30));
  }

  private resize() {
    this.world?.resize(window.innerWidth, window.innerHeight);
  }

  private portrait(k: Kid, expr: Expression = 'happy', size = 140): HTMLImageElement {
    const team = TEAMS.find((t) => t.roster.includes(k.id))!;
    const img = h('img', { class: 'portrait', width: size / 2, height: size / 2, alt: k.nick });
    const a = this.world.actors.get(k.id);
    if (a) this.studio.into(img, a.model, team, expr, size);
    return img;
  }

  private startAttract() {
    if (!this.attract) this.attract = new Attract(this.world, this.studio);
  }

  private stopAttract() {
    this.attract?.stop();
    this.attract = null;
  }

  private show(el: HTMLElement) {
    clear(this.menus);
    this.menus.classList.remove('hidden');
    this.menus.appendChild(el);
    this.menus.scrollTop = 0;
  }

  // ───────────────────────────────────────────────────────────── screens

  private title() {
    this.startAttract();
    audio.playMusic('title');
    const btn = (label: string, on: () => void, cls = 'btn big') => h('button', { class: cls, onclick: () => { audio.play('uiTap'); on(); } }, label);
    this.show(h('div', { class: 'screen title' },
      h('div', { class: 'logo-big' },
        h('div', { class: 'logo-small' }, 'the'),
        h('div', { class: 'logo-word' }, 'GRASS'),
        h('div', { class: 'logo-word w2' }, 'STAIN'),
        h('div', { class: 'logo-word w3' }, 'LEAGUE')),
      h('div', { class: 'tagline' }, 'Backyard baseball, played by very serious kids.'),
      h('div', { class: 'menu-col' },
        btn('⚾ Play Ball!', () => this.pickTeam()),
        btn('Meet the Kids', () => this.roster(), 'btn big ghost'),
        h('div', { class: 'row' },
          btn('How to Play', () => this.howTo(), 'btn ghost'),
          btn('Settings', () => this.settingsScreen(), 'btn ghost'))),
      h('div', { class: 'footer' }, 'Demo build · two teams, one backyard · every model, texture, sound and song is generated in code')));
  }

  private head(title: string, back: () => void) {
    return h('div', { class: 'screen-head' },
      h('button', { class: 'btn icon ghost', onclick: () => { audio.play('uiBack'); back(); } }, '◀'),
      h('h1', null, title),
      h('div', { class: 'spacer' }));
  }

  private pickTeam() {
    const card = (t: Team) => {
      const other = TEAMS.find((o) => o.id !== t.id)!;
      const el = h('button', {
        class: `team-pick${this.yourTeam.id === t.id ? ' selected' : ''}`, style: `--team:${t.colors.primary};--team2:${t.colors.secondary}`,
        onclick: () => { audio.play('uiSelect'); this.yourTeam = t; this.pickTeam(); },
      },
        h('div', { class: 'tp-head' }, teamBadge(t, 54), h('div', null, h('div', { class: 'tc-street' }, t.street), h('div', { class: 'tc-name' }, t.name))),
        h('div', { class: 'tp-kids' }, ...t.roster.map((id) => this.portrait(kid(id), 'happy', 112))),
        h('div', { class: 'tc-sub' }, t.id === 'comets' ? `Home team — it's Kai Mendoza's backyard. You pitch first.` : `The visitors from Maple Street. You bat first against the ${other.name}.`));
      return el;
    };
    const seg = <T extends string | number>(label: string, opts: [T, string][], cur: T, set: (v: T) => void) =>
      h('div', { class: 'seg' }, h('span', { class: 'seg-label' }, label),
        ...opts.map(([v, l]) => h('button', { class: `btn small${cur === v ? ' on' : ''}`, onclick: () => { audio.play('uiTap'); set(v); this.pickTeam(); } }, l)));
    this.show(h('div', { class: 'screen' },
      this.head('Pick your team', () => this.title()),
      h('div', { class: 'team-picks' }, card(TEAMS[0]), h('div', { class: 'vs' }, 'vs'), card(TEAMS[1])),
      h('div', { class: 'options' },
        seg('Innings', [[3, '3 (quick)'], [6, '6 (full game)']], this.innings, (v) => { this.innings = v; }),
        seg('Difficulty', [['rookie', 'Rookie'], ['pro', 'Pro'], ['allstar', 'All-Star']] as [Difficulty, string][], settings.difficulty, (v) => { settings.difficulty = v; saveSettings(); })),
      h('div', { class: 'cta' }, h('button', { class: 'btn big', onclick: () => { audio.play('uiSelect'); this.startGame(this.yourTeam); } }, `⚾ Play as the ${this.yourTeam.name}!`))));
  }

  private startGame(you: Team, cpuOnly = false) {
    this.stopAttract();
    this.menus.classList.add('hidden');
    clear(this.menus);
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

  private traitBars(t: Traits) {
    return h('div', { class: 'traits' }, ...(Object.keys(TRAIT_LABELS) as (keyof Traits)[]).map((k) =>
      h('div', { class: 'bar' }, h('span', null, TRAIT_LABELS[k]), h('i', null, h('b', { style: `width:${t[k] * 10}%` })), h('em', null, String(t[k])))));
  }

  private roster() {
    const teamBlock = (t: Team) => h('div', { class: 'team-block', style: `--team:${t.colors.primary};--team2:${t.colors.secondary}` },
      h('div', { class: 'ds-head' }, teamBadge(t, 36), h('span', null, teamName(t))),
      h('div', { class: 'kid-grid' }, ...t.roster.map((id) => {
        const k = kid(id);
        return h('button', { class: 'kid-card', onclick: () => { audio.play('uiSelect'); this.kidModal(k); } },
          this.portrait(k, 'happy', 152),
          h('div', { class: 'kc-body' },
            h('div', { class: 'kc-name' }, `${k.first} "${k.nick}" ${k.last}`),
            h('div', { class: 'kc-persona' }, k.persona),
            h('div', { class: 'kc-special' }, `⚡ ${SPECIAL_INFO[k.special].label}`)),
          this.traitBars(k.traits));
      })));
    this.show(h('div', { class: 'screen' },
      this.head('Meet the Kids', () => this.title()),
      h('p', { class: 'lede' }, 'Eighteen kids, two teams, and not one of them acts their age. Everyone pitches. Nobody sits out. Tap a kid to hear their whole deal.'),
      teamBlock(TEAMS[0]), teamBlock(TEAMS[1])));
  }

  private kidModal(k: Kid) {
    const team = TEAMS.find((t) => t.roster.includes(k.id))!;
    const close = () => overlay.remove();
    const overlay = h('div', { class: 'overlay', onclick: (e: Event) => { if (e.target === overlay) close(); } },
      h('div', { class: 'panel kid-modal' },
        h('div', { class: 'km-faces' }, this.portrait(k, 'happy', 240), this.portrait(k, 'yell', 160), this.portrait(k, 'smug', 160)),
        h('h2', null, `${k.first} "${k.nick}" ${k.last}`),
        h('div', { class: 'km-team' }, teamBadge(team, 22), ` ${teamName(team)} · age ${k.age} · bats ${k.bats === 'S' ? 'both' : k.bats === 'L' ? 'left' : 'right'}, throws ${k.throws === 'L' ? 'left' : 'right'}`),
        h('div', { class: 'km-persona' }, k.persona),
        h('p', null, k.bio),
        this.traitBars(k.traits),
        h('div', { class: 'km-special' }, h('b', null, `⚡ ${SPECIAL_INFO[k.special].label}: `), SPECIAL_INFO[k.special].blurb),
        h('div', { class: 'km-pitches' }, h('b', null, 'Pitches: '), k.pitches.map((p) => PITCHES[p].label).join(', ')),
        h('div', { class: 'km-quips' }, ...k.quips.map((q) => h('span', null, `"${q}"`))),
        h('button', { class: 'btn', onclick: close }, 'Back')));
    this.root.appendChild(overlay);
  }

  private howTo() {
    this.show(h('div', { class: 'screen narrow' },
      this.head('How to Play', () => this.title()),
      h('div', { class: 'how' },
        h('h3', null, 'Batting'),
        h('ul', null,
          h('li', null, 'Move the yellow circle over where the pitch will cross the plate (mouse, drag, or arrow keys).'),
          h('li', null, 'Click, tap SWING, or press Space when the ball arrives. Early swings pull the ball, late ones go the other way.'),
          h('li', null, 'POWER (P) hits harder with a smaller sweet spot. BUNT (B) squares around.'),
          h('li', null, 'When the hype meter fills, unleash your kid\'s special (S).'),
          h('li', null, 'Runners run on their own; RUN! / BACK! (R / F) takes charge.')),
        h('h3', null, 'Pitching'),
        h('ul', null,
          h('li', null, 'Pick a pitch (1-3), then click or tap in the zone to throw it there.'),
          h('li', null, 'Better pitchers hit their spots and last longer. Tired arms get swapped out automatically.')),
        h('h3', null, 'Fielding'),
        h('ul', null,
          h('li', null, 'Your kids chase the ball themselves. Tap a base (or press 1, 2, 3, H) to choose where they throw.')),
        h('h3', null, 'Pool Party Paradise'),
        h('ul', null, ...yard('poolparty').rules.map((r) => h('li', null, r))))));
  }

  private settingsScreen() {
    const slider = (label: string, v: number, set: (x: number) => void) => h('label', { class: 'slider' }, label,
      h('input', { type: 'range', min: '0', max: '1', step: '0.05', value: String(v), oninput: (e: Event) => { set(Number((e.target as HTMLInputElement).value)); saveSettings(); } }));
    const toggle = (label: string, v: boolean, set: (x: boolean) => void) => h('label', { class: 'toggle' },
      h('input', { type: 'checkbox', checked: v, onchange: (e: Event) => { set((e.target as HTMLInputElement).checked); saveSettings(); } }), ` ${label}`);
    const q = qualitySetting();
    const qBtn = (v: QualityName | 'auto', l: string) => h('button', { class: `btn small${q === v ? ' on' : ''}`, onclick: () => { setQuality(v); location.reload(); } }, l);
    const aim = (v: 'auto' | 'on' | 'off', l: string) => h('button', { class: `btn small${settings.aimAssist === v ? ' on' : ''}`, onclick: () => { settings.aimAssist = v; saveSettings(); this.settingsScreen(); } }, l);
    this.show(h('div', { class: 'screen narrow' },
      this.head('Settings', () => this.title()),
      h('div', { class: 'options' },
        slider('Sound effects', settings.sfx, (x) => { settings.sfx = x; audio.setSfxVolume(x); }),
        slider('Music', settings.music, (x) => { settings.music = x; audio.setMusicVolume(x); }),
        toggle('Announcer voice (uses your device\'s speech)', settings.voice, (x) => { settings.voice = x; }),
        toggle('Always show the strike zone', settings.showZone, (x) => { settings.showZone = x; }),
        h('div', { class: 'seg' }, h('span', { class: 'seg-label' }, 'Aim assist'), aim('auto', 'By difficulty'), aim('on', 'On'), aim('off', 'Off')),
        h('div', { class: 'seg' }, h('span', { class: 'seg-label' }, 'Graphics'), qBtn('auto', 'Auto'), qBtn('low', 'Fast'), qBtn('medium', 'Balanced'), qBtn('high', 'Beautiful')),
        h('div', { class: 'tc-sub' }, `Changing graphics reloads the page. Right now: ${this.world.q.name}.`))));
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


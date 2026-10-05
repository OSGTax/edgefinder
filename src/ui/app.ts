import { Rng } from '../engine/rng';
import { load, remove, save } from '../engine/storage';
import { audio } from '../audio';
import { KID_BY_ID, kid } from '../data/kids';
import { ANNOUNCERS, TEAMS, team, teamName } from '../data/teams';
import { YARDS, yard } from '../data/yards';
import { SPECIAL_INFO, type Kid, type Team } from '../data/types';
import { autoLineup } from '../sim/lineup';
import type { Match, MatchConfig } from '../sim/match';
import type { Difficulty } from '../sim/types';
import { PITCHES } from '../sim/pitching';
import { buildField } from '../sim/field';
import { cpuPick, draftDone, pick, startDraft, type Draft } from '../league/draft';
import {
  createSeason, currentDay, inPlayoffs, leaders, matchConfigFor, nextUserGame, recordResult, simThrough, standings,
  type ScheduledGame, type SeasonState,
} from '../league/season';
import { logoCanvas } from '../art/logo';
import { SceneRenderer, type KidSprite } from '../render/scene';
import { OVERVIEW_CAM } from '../render/cameras';
import { clear, h } from './dom';
import { GameScreen } from './game';
import { portraitCanvas } from './portraits';
import { saveSettings, settings } from './settings';

export const GAME_TITLE = 'Grass Stain League';
const VERSION = '0.1.0';

type Screen = () => HTMLElement;

/** The whole front end: an animated backyard behind a stack of menus. */
export class App {
  private root: HTMLElement;
  private bg: SceneRenderer;
  private bgKids: KidSprite[] = [];
  private bgCanvas: HTMLCanvasElement;
  private layer: HTMLElement;
  private t = 0;
  private game: GameScreen | null = null;
  private season: SeasonState | null = load<SeasonState | null>('season', null);

  constructor(root: HTMLElement) {
    this.root = root;
    this.bgCanvas = h('canvas', { class: 'bg' });
    this.layer = h('div', { class: 'menus' });
    root.append(this.bgCanvas, this.layer);
    this.bg = new SceneRenderer(this.bgCanvas);
    this.setBackdrop(YARDS[Math.floor(Math.random() * YARDS.length)].id);
    window.addEventListener('resize', () => this.resizeBg());
    this.resizeBg();
    const unlock = () => { audio.unlock(); audio.playMusic(this.game ? 'game' : 'title'); };
    window.addEventListener('pointerdown', unlock, { once: true });
    window.addEventListener('keydown', unlock, { once: true });
    audio.setSfxVolume(settings.sfx);
    audio.setMusicVolume(settings.music);
    this.loop();
    this.show(() => this.title());
  }

  // ──────────────────────────────────────────────────── backdrop

  private setBackdrop(yardId: string, teamId?: string) {
    const y = yard(yardId);
    this.bg.field = buildField(y);
    const t = teamId ? team(teamId) : TEAMS.find((tt) => tt.yardId === y.id) ?? TEAMS[0];
    this.bgKids = t.roster.map((id, i) => ({
      kid: kid(id), team: t, x: 0, y: 0, facing: 0, pose: { anim: 'run' as const, t: i * 0.7 },
    }));
  }

  private resizeBg() {
    this.bg.resize(window.innerWidth, window.innerHeight);
  }

  private loop = () => {
    requestAnimationFrame(this.loop);
    if (this.game) return;
    this.t += 1 / 60;
    const a = Math.sin(this.t * 0.05) * 0.9;
    this.bg.cam.pose = { ...OVERVIEW_CAM, pos: { x: Math.sin(a) * 170, y: 50 + Math.cos(a) * 170, z: 60 }, target: { x: 0, y: 55, z: 0 } };
    const f = this.bg.field!;
    this.bgKids.forEach((s, i) => {
      // kids chasing each other around the bases
      const u = ((this.t * 0.06 + i / 9) % 1) * 4;
      const b = Math.floor(u), frac = u - b;
      const A = f.bases[b % 4], B = f.bases[(b + 1) % 4];
      s.x = A.x + (B.x - A.x) * frac;
      s.y = A.y + (B.y - A.y) * frac;
      s.facing = Math.atan2(B.x - A.x, B.y - A.y);
      s.pose.t = this.t + i;
    });
    this.bg.draw({ kids: this.bgKids, ball: null, t: this.t }, 1 / 60);
  };

  // ──────────────────────────────────────────────────── navigation

  private show(screen: Screen) {
    clear(this.layer);
    this.layer.appendChild(screen());
    this.layer.scrollTop = 0;
  }

  private back(to: Screen) {
    audio.play('uiBack');
    this.show(to);
  }

  private go(to: Screen) {
    audio.play('uiTap');
    this.show(to);
  }

  private startGame(cfg: MatchConfig, onDone: (m: Match | null) => void) {
    audio.play('uiSelect');
    this.layer.classList.add('hidden');
    this.bgCanvas.classList.add('hidden');
    this.game = new GameScreen(this.root, {
      cfg,
      onExit: (m) => {
        this.game = null;
        this.layer.classList.remove('hidden');
        this.bgCanvas.classList.remove('hidden');
        audio.playMusic('title');
        onDone(m);
      },
    });
  }

  // ──────────────────────────────────────────────────── screens

  private title(): HTMLElement {
    audio.playMusic('title');
    return h('div', { class: 'screen title' },
      h('div', { class: 'logo-big' },
        h('div', { class: 'logo-small' }, 'the'),
        h('div', { class: 'logo-word w1' }, 'GRASS'),
        h('div', { class: 'logo-word w2' }, 'STAIN'),
        h('div', { class: 'logo-word w3' }, 'LEAGUE')),
      h('div', { class: 'tagline' }, 'Backyard baseball, played by tiny professionals.'),
      h('div', { class: 'menu-col' },
        h('button', { class: 'btn big', onclick: () => this.go(() => this.quickSetup()) }, '⚾ Play Ball!'),
        h('button', { class: 'btn big ghost', onclick: () => this.go(() => this.draftSetup()) }, '🙋 Pick-Up Game'),
        h('button', { class: 'btn big ghost', onclick: () => this.go(() => (this.season ? this.seasonHub() : this.seasonSetup())) }, `🏆 ${this.season ? 'Continue Season' : 'Season'}`),
        h('div', { class: 'row' },
          h('button', { class: 'btn small ghost', onclick: () => this.go(() => this.rosters()) }, 'The Kids'),
          h('button', { class: 'btn small ghost', onclick: () => this.go(() => this.howTo()) }, 'How to Play'),
          h('button', { class: 'btn small ghost', onclick: () => this.go(() => this.settingsScreen()) }, 'Settings'))),
      h('div', { class: 'footer' }, `v${VERSION} · every kid, yard, sound and song is made from code`));
  }

  private header(title: string, back: Screen, extra?: HTMLElement): HTMLElement {
    return h('div', { class: 'screen-head' },
      h('button', { class: 'btn small ghost', onclick: () => this.back(back) }, '◀ Back'),
      h('h1', null, title),
      extra ?? h('span', { class: 'spacer' }));
  }

  private teamCard(t: Team, selected: boolean, onPick: () => void, note?: string) {
    const star = kid(t.roster[0]);
    return h('button', {
      class: `team-card${selected ? ' selected' : ''}`,
      style: `--team:${t.colors.primary};--team2:${t.colors.secondary}`,
      onclick: onPick,
    },
    logoCanvas(t, 56),
    h('div', { class: 'tc-txt' },
      h('div', { class: 'tc-street' }, t.street),
      h('div', { class: 'tc-name' }, t.name),
      h('div', { class: 'tc-sub' }, note ?? `${yard(t.yardId).name} · ace: ${star.nick}`)));
  }

  private segmented<T extends string | number>(label: string, options: [T, string][], value: T, onChange: (v: T) => void) {
    const wrap = h('div', { class: 'seg' }, h('span', { class: 'seg-label' }, label));
    for (const [v, text] of options) {
      wrap.appendChild(h('button', {
        class: `btn small${v === value ? ' on' : ' ghost'}`,
        onclick: () => { audio.play('uiTap'); onChange(v); },
      }, text));
    }
    return wrap;
  }

  // ── quick game

  private qs = { mine: 'mudcats', theirs: 'comets', home: true, innings: 3, difficulty: settings.difficulty as Difficulty };

  private quickSetup(): HTMLElement {
    const q = this.qs;
    const rerender = () => this.show(() => this.quickSetup());
    const grid = (sel: string, onPick: (id: string) => void, exclude?: string) =>
      h('div', { class: 'team-grid' }, ...TEAMS.filter((t) => t.id !== exclude).map((t) => this.teamCard(t, t.id === sel, () => { audio.play('uiSelect'); onPick(t.id); this.setBackdrop(t.yardId, t.id); rerender(); })));
    if (q.theirs === q.mine) q.theirs = TEAMS.find((t) => t.id !== q.mine)!.id;
    const homeTeam = team(q.home ? q.mine : q.theirs);
    return h('div', { class: 'screen' },
      this.header('Play Ball!', () => this.title()),
      h('h3', null, 'Your team'),
      grid(q.mine, (id) => { q.mine = id; }),
      h('h3', null, 'Opponent'),
      grid(q.theirs, (id) => { q.theirs = id; }, q.mine),
      h('div', { class: 'options' },
        this.segmented('Where', [[1, `Home (${yard(team(q.mine).yardId).name})`], [0, `Away (${yard(team(q.theirs).yardId).name})`]], q.home ? 1 : 0, (v) => { q.home = v === 1; rerender(); }),
        this.segmented('Innings', [[3, '3'], [6, '6'], [9, '9']], q.innings, (v) => { q.innings = v; rerender(); }),
        this.diffSeg(q.difficulty, (d) => { q.difficulty = d; rerender(); })),
      h('div', { class: 'cta' },
        h('button', {
          class: 'btn big', onclick: () => {
            const mine = team(q.mine), theirs = team(q.theirs);
            const [away, home] = q.home ? [theirs, mine] : [mine, theirs];
            this.startGame({
              away: { team: away, lineup: autoLineup(away.roster.map(kid)), human: away === mine },
              home: { team: home, lineup: autoLineup(home.roster.map(kid)), human: home === mine },
              yard: yard(homeTeam.yardId), innings: q.innings, seed: Math.floor(Math.random() * 1e9), difficulty: q.difficulty, mercy: 10,
            }, () => this.show(() => this.quickSetup()));
          },
        }, `Play at ${yard(homeTeam.yardId).name} ▶`)));
  }

  private diffSeg(value: Difficulty, onChange: (d: Difficulty) => void) {
    return this.segmented<Difficulty>('Difficulty', [['rookie', 'Rookie'], ['pro', 'Pro'], ['allstar', 'All-Star']], value, onChange);
  }

  // ── pick-up game

  private draft: Draft | null = null;
  private draftOpts = { jersey: 'owls', rival: 'rockets', yardId: 'grandmabea', innings: 3, difficulty: settings.difficulty as Difficulty };
  private draftRng = new Rng(Date.now() & 0xffff);

  private draftSetup(): HTMLElement {
    const o = this.draftOpts;
    const rerender = () => this.show(() => this.draftSetup());
    if (o.rival === o.jersey) o.rival = TEAMS.find((t) => t.id !== o.jersey)!.id;
    return h('div', { class: 'screen' },
      this.header('Pick-Up Game', () => this.title()),
      h('p', { class: 'lede' }, 'Twenty kids show up at the park. You and the other captain take turns picking teams. Choose wisely.'),
      h('h3', null, 'Your jerseys'),
      h('div', { class: 'team-grid' }, ...TEAMS.map((t) => this.teamCard(t, t.id === o.jersey, () => { audio.play('uiSelect'); o.jersey = t.id; rerender(); }, `${t.street} colors`))),
      h('h3', null, 'Whose yard?'),
      h('div', { class: 'yard-grid' }, ...YARDS.map((y) => h('button', {
        class: `yard-card${y.id === o.yardId ? ' selected' : ''}`,
        onclick: () => { audio.play('uiSelect'); o.yardId = y.id; this.setBackdrop(y.id); rerender(); },
      }, h('div', { class: 'yc-name' }, y.name), h('div', { class: 'yc-sub' }, y.blurb)))),
      h('div', { class: 'options' },
        this.segmented('Innings', [[3, '3'], [6, '6'], [9, '9']], o.innings, (v) => { o.innings = v; rerender(); }),
        this.diffSeg(o.difficulty, (d) => { o.difficulty = d; rerender(); })),
      h('div', { class: 'cta' }, h('button', {
        class: 'btn big', onclick: () => {
          this.draft = startDraft(Math.floor(Math.random() * 1e9));
          this.go(() => this.draftBoard());
        },
      }, 'Head to the park ▶')));
  }

  private draftBoard(): HTMLElement {
    const d = this.draft!;
    const o = this.draftOpts;
    const mine = team(o.jersey), rival = team(o.rival);
    const rerender = () => this.show(() => this.draftBoard());
    if (draftDone(d)) return this.draftReady();
    if (!d.myTurn) {
      setTimeout(() => {
        if (this.draft !== d || d.myTurn || draftDone(d)) return;
        const id = cpuPick(d, kid, this.draftRng);
        pick(d, id);
        audio.play('uiSelect');
        this.toast(`The other captain picks ${kid(id).nick}!`);
        rerender();
      }, 900);
    }
    const roster = (ids: string[], t: Team, label: string) => h('div', { class: 'draft-side', style: `--team:${t.colors.primary};--team2:${t.colors.secondary}` },
      h('div', { class: 'ds-head' }, logoCanvas(t, 28), h('span', null, `${label} (${ids.length}/9)`)),
      h('div', { class: 'ds-list' }, ...ids.map((id) => h('div', { class: 'ds-kid' }, portraitCanvas(kid(id), t, 34, 34), h('span', null, kid(id).nick)))));
    return h('div', { class: 'screen' },
      this.header('Pick Your Team', () => this.draftSetup(),
        h('div', { class: `turn ${d.myTurn ? 'mine' : 'theirs'}` }, d.myTurn ? 'Your pick!' : 'They\'re picking…')),
      h('div', { class: 'draft' },
        roster(d.mine, mine, 'You'),
        h('div', { class: 'kid-grid' }, ...d.pool.map((id) => this.kidCard(kid(id), null, d.myTurn ? () => { pick(d, id); audio.play('uiSelect'); rerender(); } : undefined))),
        roster(d.theirs, rival, 'Them')));
  }

  private draftReady(): HTMLElement {
    const d = this.draft!;
    const o = this.draftOpts;
    const mineT = { ...team(o.jersey), name: 'Picks', street: 'Your', roster: d.mine };
    const theirT = { ...team(o.rival), name: 'Rejects', street: 'The', roster: d.theirs };
    return h('div', { class: 'screen' },
      this.header('Teams Are Set!', () => this.draftSetup()),
      h('div', { class: 'draft' },
        this.lineupCard(mineT, d.mine),
        h('div', { class: 'vs' }, 'VS'),
        this.lineupCard(theirT, d.theirs)),
      h('div', { class: 'cta' }, h('button', {
        class: 'btn big', onclick: () => {
          this.startGame({
            away: { team: theirT, lineup: autoLineup(d.theirs.map(kid)), human: false },
            home: { team: mineT, lineup: autoLineup(d.mine.map(kid)), human: true },
            yard: yard(o.yardId), innings: o.innings, seed: Math.floor(Math.random() * 1e9), difficulty: o.difficulty, mercy: 10,
          }, () => this.show(() => this.title()));
        },
      }, `Play at ${yard(o.yardId).name} ▶`)));
  }

  private lineupCard(t: Team, ids: string[]) {
    const lu = autoLineup(ids.map(kid));
    const posOf = (id: string) => ['P', 'C', '1B', '2B', '3B', 'SS', 'LF', 'CF', 'RF'][lu.defense.indexOf(id)];
    return h('div', { class: 'lineup', style: `--team:${t.colors.primary};--team2:${t.colors.secondary}` },
      h('div', { class: 'ds-head' }, logoCanvas(t, 32), h('span', null, `${t.street} ${t.name}`)),
      ...lu.order.map((id, i) => h('div', { class: 'lu-row' }, h('b', null, `${i + 1}.`), portraitCanvas(kid(id), t, 30, 30), h('span', null, kid(id).nick), h('i', null, posOf(id)))));
  }

  private toast(text: string) {
    const el = h('div', { class: 'toast' }, text);
    this.root.appendChild(el);
    setTimeout(() => el.remove(), 1800);
  }

  // ── kid cards

  private kidCard(k: Kid, t: Team | null, onPick?: () => void) {
    const s = k.stats;
    t = t ?? TEAMS.find((tt) => tt.roster.includes(k.id)) ?? null;
    const bar = (label: string, v: number) => h('div', { class: 'bar' }, h('span', null, label), h('i', null, h('b', { style: `width:${v * 10}%` })));
    return h('div', { class: 'kid-card' },
      h('button', { class: 'kc-face', onclick: () => this.kidModal(k, t) }, portraitCanvas(k, t, 76, 76, '#fff3d6')),
      h('div', { class: 'kc-body' },
        h('div', { class: 'kc-name' }, k.nick),
        h('div', { class: 'kc-persona' }, k.persona),
        bar('Bat', s.contact), bar('Pow', s.power), bar('Spd', s.speed), bar('Arm', s.arm), bar('Glv', s.fielding), bar('Pit', s.pitching),
        h('div', { class: 'kc-special' }, `⚡ ${SPECIAL_INFO[k.special].label}`)),
      onPick ? h('button', { class: 'btn small pick', onclick: onPick }, 'Pick!') : null);
  }

  private kidModal(k: Kid, t: Team | null) {
    audio.play('uiTap');
    const home = TEAMS.find((tt) => tt.roster.includes(k.id)) ?? null;
    const sl = this.season?.stats[k.id];
    const statLine = sl ? `${sl.bat.h}-for-${sl.bat.ab}, ${sl.bat.hr} HR, ${sl.bat.rbi} RBI${sl.pitch.outs ? ` · ${Math.floor(sl.pitch.outs / 3)}.${sl.pitch.outs % 3} IP, ${sl.pitch.so} K` : ''}` : null;
    const close = () => modal.remove();
    const modal = h('div', { class: 'overlay', onclick: (e: Event) => { if (e.target === modal) close(); } },
      h('div', { class: 'panel kid-modal' },
        portraitCanvas(k, t ?? home, 160, 160, '#fff3d6'),
        h('h2', null, `${k.first} "${k.nick}" ${k.last}`),
        h('div', { class: 'km-persona' }, `${k.persona} · Age ${k.age} · Bats ${k.bats} · Throws ${k.throws}`),
        home ? h('div', { class: 'km-team' }, logoCanvas(home, 22), ` ${teamName(home)}`) : null,
        h('p', null, k.bio),
        h('div', { class: 'km-quips' }, ...k.quips.map((q) => h('span', null, `"${q}"`))),
        h('div', { class: 'km-special' }, h('b', null, `⚡ ${SPECIAL_INFO[k.special].label}: `), SPECIAL_INFO[k.special].blurb),
        h('div', { class: 'km-pitches' }, `Pitches: ${k.pitches.map((p) => PITCHES[p].label).join(', ')}`),
        statLine ? h('div', { class: 'km-season' }, `This season: ${statLine}`) : null,
        h('button', { class: 'btn', onclick: close }, 'Close')));
    this.root.appendChild(modal);
  }

  private rosters(): HTMLElement {
    return h('div', { class: 'screen' },
      this.header('The Kids', () => this.title()),
      h('p', { class: 'lede' }, `Seventy-two kids, eight backyards, zero adults in charge. In the booth: ${ANNOUNCERS.play.name} (${ANNOUNCERS.play.blurb}) and ${ANNOUNCERS.color.name} (${ANNOUNCERS.color.blurb})`),
      ...TEAMS.map((t) => h('div', { class: 'team-block', style: `--team:${t.colors.primary};--team2:${t.colors.secondary}` },
        h('div', { class: 'ds-head' }, logoCanvas(t, 40), h('span', null, `${teamName(t)} — ${t.division} Division · ${yard(t.yardId).name}`)),
        h('div', { class: 'kid-grid' }, ...t.roster.map((id) => this.kidCard(kid(id), t))))));
  }

  // ── season

  private seasonOpts = { team: 'mudcats', rounds: 1, innings: 3, difficulty: settings.difficulty as Difficulty };

  private seasonSetup(): HTMLElement {
    const o = this.seasonOpts;
    const rerender = () => this.show(() => this.seasonSetup());
    return h('div', { class: 'screen' },
      this.header('New Season', () => this.title()),
      h('p', { class: 'lede' }, 'Eight teams, two divisions. The top two in each division make the playoffs. The winner takes home the Lemonade Cup.'),
      h('h3', null, 'Pick your team'),
      h('div', { class: 'team-grid' }, ...TEAMS.map((t) => this.teamCard(t, t.id === o.team, () => { audio.play('uiSelect'); o.team = t.id; this.setBackdrop(t.yardId, t.id); rerender(); }, `${t.division} Division`))),
      h('div', { class: 'options' },
        this.segmented('Length', [[1, '7 games'], [2, '14 games'], [3, '21 games']], o.rounds, (v) => { o.rounds = v; rerender(); }),
        this.segmented('Innings', [[3, '3'], [6, '6']], o.innings, (v) => { o.innings = v; rerender(); }),
        this.diffSeg(o.difficulty, (d) => { o.difficulty = d; rerender(); })),
      h('div', { class: 'cta' }, h('button', {
        class: 'btn big', onclick: () => {
          this.season = createSeason(o.team, { innings: o.innings, difficulty: o.difficulty, rounds: o.rounds });
          this.saveSeason();
          this.go(() => this.seasonHub());
        },
      }, 'Start the season ▶')));
  }

  private saveSeason() {
    if (this.season) save('season', this.season);
  }

  private hubTab: 'standings' | 'leaders' | 'results' | 'roster' = 'standings';

  private seasonHub(): HTMLElement {
    const s = this.season!;
    const me = team(s.userTeam);
    const rerender = () => this.show(() => this.seasonHub());
    const st = standings(s);
    const myRow = [...st['Front Porch'], ...st['Back Fence']].find((r) => r.team.id === me.id)!;
    const rank = st[me.division].indexOf(myRow) + 1;
    const next = nextUserGame(s);
    const playoffs = inPlayoffs(s);

    let nextCard: HTMLElement;
    if (s.champion) {
      const champ = team(s.champion);
      nextCard = h('div', { class: 'next-card champ' },
        h('div', { class: 'cup' }, '🏆'),
        h('div', null,
          h('div', { class: 'card-label' }, 'LEMONADE CUP CHAMPIONS'),
          h('div', { class: 'nc-title' }, teamName(champ)),
          h('div', { class: 'nc-sub' }, champ.id === me.id ? 'That\'s YOU! Juice boxes for everyone!' : 'Better luck next summer.')),
        h('button', { class: 'btn', onclick: () => { remove('season'); this.season = null; this.go(() => this.seasonSetup()); } }, 'New season'));
    } else if (next) {
      const opp = team(next.away === me.id ? next.home : next.away);
      const where = yard(team(next.home).yardId);
      const label = next.kind === 'final' ? 'THE LEMONADE CUP FINAL' : next.kind === 'semi' ? 'PLAYOFF SEMIFINAL' : `GAME ${s.games.filter((g) => g.kind === 'regular' && (g.away === me.id || g.home === me.id) && g.result).length + 1}`;
      nextCard = h('div', { class: 'next-card' },
        logoCanvas(opp, 64),
        h('div', null,
          h('div', { class: 'card-label' }, label),
          h('div', { class: 'nc-title' }, `${next.home === me.id ? 'vs' : '@'} ${teamName(opp)}`),
          h('div', { class: 'nc-sub' }, `at ${where.name}`)),
        h('div', { class: 'nc-btns' },
          h('button', { class: 'btn', onclick: () => this.playSeasonGame(next) }, 'Play ▶'),
          h('button', { class: 'btn small ghost', onclick: () => { this.simSeasonGame(next); rerender(); } }, 'Sim it')));
    } else {
      nextCard = h('div', { class: 'next-card' }, h('div', null,
        h('div', { class: 'card-label' }, playoffs ? 'PLAYOFFS' : 'SEASON OVER'),
        h('div', { class: 'nc-title' }, playoffs ? 'You\'re out — but the playoffs go on!' : 'Waiting on the other games'),
        h('button', { class: 'btn', onclick: () => { simThrough(s, currentDay(s), true); this.saveSeason(); rerender(); } }, 'Sim next day')));
    }

    const tabs = h('div', { class: 'tabs' }, ...(['standings', 'leaders', 'results', 'roster'] as const).map((tb) =>
      h('button', { class: `btn small${this.hubTab === tb ? ' on' : ' ghost'}`, onclick: () => { this.hubTab = tb; audio.play('uiTap'); rerender(); } }, tb[0].toUpperCase() + tb.slice(1))));

    let body: HTMLElement;
    if (this.hubTab === 'standings') {
      body = h('div', { class: 'standings' }, ...(['Front Porch', 'Back Fence'] as const).map((div) =>
        h('table', { class: 'tbl' },
          h('tr', null, h('th', { class: 'l' }, `${div} Division`), h('th', null, 'W'), h('th', null, 'L'), h('th', null, 'T'), h('th', null, 'GB'), h('th', null, 'RS'), h('th', null, 'RA'), h('th', null, 'Strk')),
          ...st[div].map((r, i) => h('tr', { class: r.team.id === me.id ? 'me' : '' },
            h('td', { class: 'l' }, logoCanvas(r.team, 18), ` ${r.team.street} ${r.team.name}${i < 2 && !playoffs ? '' : ''}`),
            h('td', null, r.w), h('td', null, r.l), h('td', null, r.t), h('td', null, r.gb === 0 ? '—' : r.gb.toFixed(1)), h('td', null, r.rs), h('td', null, r.ra), h('td', null, r.streak))))));
    } else if (this.hubTab === 'leaders') {
      const board = (title: string, stat: Parameters<typeof leaders>[1]) => h('div', { class: 'leader' },
        h('h4', null, title),
        ...leaders(s, stat).map((l, i) => {
          const k = KID_BY_ID[l.id];
          const t = TEAMS.find((tt) => tt.roster.includes(l.id)) ?? null;
          return h('div', { class: 'lr', onclick: () => this.kidModal(k, t) }, h('b', null, `${i + 1}`), portraitCanvas(k, t, 28, 28), h('span', null, k.nick), h('i', null, l.label));
        }));
      body = h('div', { class: 'leaders' }, board('Batting Avg', 'avg'), board('Home Runs', 'hr'), board('RBI', 'rbi'), board('Strikeouts', 'so'), board('ERA', 'era'), board('Runs', 'r'));
    } else if (this.hubTab === 'results') {
      const played = s.games.filter((g) => g.result).slice(-24).reverse();
      body = h('div', { class: 'results' }, played.length ? null : h('p', null, 'No games yet. Play ball!'),
        ...played.map((g) => {
          const [a, hh] = g.result!.score;
          const star = g.result!.star ? KID_BY_ID[g.result!.star] : null;
          return h('div', { class: `res${g.away === me.id || g.home === me.id ? ' me' : ''}` },
            h('span', { class: 'res-kind' }, g.kind === 'regular' ? `Day ${g.day + 1}` : g.kind === 'semi' ? 'Semi' : 'FINAL'),
            h('span', null, `${team(g.away).abbr} ${a}  @  ${team(g.home).abbr} ${hh}`),
            star ? h('span', { class: 'res-star' }, `⭐ ${star.nick}`) : null);
        }));
    } else {
      body = h('div', { class: 'kid-grid' }, ...me.roster.map((id) => {
        const k = kid(id);
        const card = this.kidCard(k, me);
        const sl = s.stats[id];
        if (sl) card.appendChild(h('div', { class: 'kc-season' }, `${sl.bat.h}/${sl.bat.ab} · ${sl.bat.hr} HR · ${sl.bat.rbi} RBI${sl.pitch.so ? ` · ${sl.pitch.so} K` : ''}`));
        return card;
      }));
    }

    return h('div', { class: 'screen' },
      this.header(`${me.street} ${me.name}`, () => this.title(), logoCanvas(me, 40)),
      h('div', { class: 'record' }, `${myRow.w}-${myRow.l}${myRow.t ? `-${myRow.t}` : ''} · ${ordinalRank(rank)} in the ${me.division} Division`),
      nextCard,
      tabs,
      body,
      !s.champion ? h('div', { class: 'cta' },
        h('button', { class: 'btn small ghost', onclick: () => { this.simRest(); rerender(); } }, 'Sim the rest of the season'),
        confirmButton('Abandon season', 'Tap again to throw it away', () => { remove('season'); this.season = null; this.go(() => this.seasonSetup()); })) : null);
  }

  private playSeasonGame(g: ScheduledGame) {
    const s = this.season!;
    // the rest of the league plays the same day
    simThrough(s, g.day - 1, false);
    this.startGame(matchConfigFor(s, g, s.userTeam), (m) => {
      if (m && m.phase === 'over') {
        recordResult(s, g, m);
        simThrough(s, g.day, false);
        this.saveSeason();
      }
      this.show(() => this.seasonHub());
    });
  }

  private simSeasonGame(g: ScheduledGame) {
    const s = this.season!;
    simThrough(s, g.day, true);
    this.saveSeason();
    audio.play('uiSelect');
  }

  private simRest() {
    const s = this.season!;
    let guard = 0;
    while (!s.champion && guard++ < 80) {
      const d = currentDay(s);
      if (d < 0) break;
      simThrough(s, d, true);
    }
    this.saveSeason();
  }

  // ── settings & help

  private settingsScreen(): HTMLElement {
    const slider = (label: string, value: number, on: (v: number) => void) => h('label', { class: 'slider' }, h('span', null, label),
      h('input', { type: 'range', min: 0, max: 1, step: 0.05, value, oninput: (e: Event) => on(Number((e.target as HTMLInputElement).value)) }));
    const check = (label: string, value: boolean, on: (v: boolean) => void) => h('label', { class: 'toggle' },
      h('input', { type: 'checkbox', checked: value, onchange: (e: Event) => on((e.target as HTMLInputElement).checked) }), ` ${label}`);
    const rerender = () => this.show(() => this.settingsScreen());
    return h('div', { class: 'screen narrow' },
      this.header('Settings', () => { saveSettings(); return this.title(); }),
      h('div', { class: 'panel flat' },
        slider('Sound effects', settings.sfx, (v) => { settings.sfx = v; audio.setSfxVolume(v); saveSettings(); }),
        slider('Music', settings.music, (v) => { settings.music = v; audio.setMusicVolume(v); saveSettings(); }),
        check('Announcer voice (uses your device\'s speech)', settings.voice, (v) => { settings.voice = v; saveSettings(); }),
        check('Show the strike zone', settings.showZone, (v) => { settings.showZone = v; saveSettings(); }),
        this.segmented<'auto' | 'on' | 'off'>('Aim assist', [['auto', 'By difficulty'], ['on', 'Always'], ['off', 'Off']], settings.aimAssist, (v) => { settings.aimAssist = v; saveSettings(); rerender(); }),
        this.segmented<number>('Throw timer', [[1, 'Quick'], [1.6, 'Normal'], [3, 'Relaxed']], settings.autoThrow, (v) => { settings.autoThrow = v; saveSettings(); rerender(); }),
        this.diffSeg(settings.difficulty, (d) => { settings.difficulty = d; this.qs.difficulty = d; this.draftOpts.difficulty = d; this.seasonOpts.difficulty = d; saveSettings(); rerender(); }),
        confirmButton('Erase saved data', 'Tap again to erase everything', () => { remove('season'); remove('settings'); this.season = null; location.reload(); })));
  }

  private howTo(): HTMLElement {
    const sec = (title: string, ...lines: string[]) => h('div', { class: 'how' }, h('h3', null, title), h('ul', null, ...lines.map((l) => h('li', null, l))));
    return h('div', { class: 'screen narrow' },
      this.header('How to Play', () => this.title()),
      sec('Batting', 'The yellow circle is your bat\'s sweet spot. Line it up with the pitch.', 'Tap SWING (or click / press Space) right as the ball reaches the plate. Early swings pull the ball, late swings go the other way.', 'Under the ball = fly ball. Over it = grounder. POWER hits harder but the sweet spot shrinks. BUNT just taps it.', 'On Rookie the circle helps aim itself — you just worry about timing.'),
      sec('Pitching', 'Pick a pitch, then tap where you want it (inside the box is a strike).', 'Mix it up! Curves and changeups fool batters into swinging early.', 'Wild kids miss their spot. Tired kids miss it more.'),
      sec('Fielding', 'Your kids chase the ball on their own.', 'When one of them has it, tap a base to throw there — or wait and they\'ll decide.'),
      sec('Running', 'Runners run on their own. Tap RUN! to send everybody, or BACK! to send them home.'),
      sec('Specials & Hype', 'Big plays fill your team\'s Hype meter (the bar next to your score).', 'When it\'s full, the batter or pitcher can unleash their special: Moonshots, Brain Freezes, Wobblers and more.', 'Some kids have perks instead: Rocket Arms, Flypaper Gloves, Spring Sneakers, The Zoomies.'),
      sec('Backyard rules', 'Every yard has its own ground rules. Into the pool is a Splash Double. Over the barn is a Barn Burner. Do not step on Grandma Bea\'s tomatoes.'));
  }
}

/** A danger button that asks for a second tap instead of a pop-up dialog. */
function confirmButton(label: string, confirmLabel: string, onConfirm: () => void) {
  let armed = false;
  let timer = 0;
  const btn = h('button', {
    class: 'btn small ghost danger',
    onclick: () => {
      if (armed) { window.clearTimeout(timer); onConfirm(); return; }
      armed = true;
      btn.textContent = confirmLabel;
      btn.classList.add('armed');
      timer = window.setTimeout(() => { armed = false; btn.textContent = label; btn.classList.remove('armed'); }, 3000);
    },
  }, label);
  return btn;
}

function ordinalRank(n: number) {
  return ['1st', '2nd', '3rd', '4th'][n - 1] ?? `${n}th`;
}

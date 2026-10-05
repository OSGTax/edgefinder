import { Rng } from '../engine/rng';
import { kid } from '../data/kids';
import { team, TEAMS } from '../data/teams';
import { yard } from '../data/yards';
import type { Team } from '../data/types';
import { autoLineup } from '../sim/lineup';
import { simulateMatch, type Match, type MatchConfig } from '../sim/match';
import { addBat, addPitch, emptyBat, emptyPitch, type BatLine, type PitchLine } from '../sim/stats';
import type { Difficulty } from '../sim/types';

// A season in the neighborhood league: a round-robin schedule, standings,
// season stats, then a four-team playoff for the Lemonade Cup.

export interface GameResult { score: [number, number]; winner: 0 | 1 | -1; star?: string }

export interface ScheduledGame {
  id: string;
  day: number;
  away: string;
  home: string;
  kind: 'regular' | 'semi' | 'final';
  result?: GameResult;
}

export interface SeasonLine { g: number; bat: BatLine; pitch: PitchLine; e: number }

export interface SeasonState {
  version: 1;
  seed: number;
  userTeam: string;
  innings: number;
  difficulty: Difficulty;
  rounds: number;
  games: ScheduledGame[];
  stats: Record<string, SeasonLine>;
  champion: string | null;
}

export interface SeasonOptions { innings: number; difficulty: Difficulty; rounds: number; seed?: number }

/** Circle-method round robin: every team plays every other team once per round. */
function roundRobin(ids: string[], rng: Rng): [string, string][][] {
  const t = [...ids];
  rng.shuffle(t);
  const n = t.length;
  const days: [string, string][][] = [];
  for (let r = 0; r < n - 1; r++) {
    const day: [string, string][] = [];
    for (let i = 0; i < n / 2; i++) {
      const a = t[i], b = t[n - 1 - i];
      day.push(r % 2 === 0 ? [a, b] : [b, a]);
    }
    days.push(day);
    t.splice(1, 0, t.pop()!);
  }
  return days;
}

export function createSeason(userTeam: string, o: SeasonOptions): SeasonState {
  const seed = o.seed ?? Math.floor(Math.random() * 1e9);
  const rng = new Rng(seed);
  const ids = TEAMS.map((t) => t.id);
  const games: ScheduledGame[] = [];
  let day = 0;
  const pairings = roundRobin(ids, rng);
  for (let round = 0; round < o.rounds; round++) {
    for (const matchups of pairings) {
      for (const [a, h] of matchups) {
        const [away, home] = round % 2 === 0 ? [a, h] : [h, a];
        games.push({ id: `g${games.length}`, day, away, home, kind: 'regular' });
      }
      day++;
    }
  }
  return { version: 1, seed, userTeam, innings: o.innings, difficulty: o.difficulty, rounds: o.rounds, games, stats: {}, champion: null };
}

export const regularDays = (s: SeasonState) => Math.max(...s.games.filter((g) => g.kind === 'regular').map((g) => g.day)) + 1;

export function nextUserGame(s: SeasonState): ScheduledGame | null {
  return s.games.find((g) => !g.result && (g.away === s.userTeam || g.home === s.userTeam)) ?? null;
}

export function pendingOnDay(s: SeasonState, day: number) {
  return s.games.filter((g) => g.day === day && !g.result);
}

export function currentDay(s: SeasonState): number {
  const open = s.games.filter((g) => !g.result);
  return open.length ? Math.min(...open.map((g) => g.day)) : -1;
}

export function matchConfigFor(s: SeasonState, g: ScheduledGame, humanTeam: string | null): MatchConfig {
  const away = team(g.away), home = team(g.home);
  return {
    away: { team: away, lineup: autoLineup(away.roster.map(kid)), human: away.id === humanTeam },
    home: { team: home, lineup: autoLineup(home.roster.map(kid)), human: home.id === humanTeam },
    yard: yard(home.yardId),
    innings: s.innings,
    seed: (s.seed ^ (Number(g.id.slice(1)) * 2654435761)) >>> 0,
    difficulty: s.difficulty,
    mercy: 10,
  };
}

export function starOf(m: Match): string | undefined {
  let star: string | undefined, best = -1;
  for (const [id, l] of Object.entries(m.box)) {
    const tb = l.bat.h + l.bat.d + l.bat.t * 2 + l.bat.hr * 3;
    const score = tb * 2 + l.bat.rbi * 1.5 + l.bat.r + l.pitch.so * 0.8 + (l.side === m.winner ? 1 : 0);
    if (score > best) { best = score; star = id; }
  }
  return star;
}

export function recordResult(s: SeasonState, g: ScheduledGame, m: Match) {
  g.result = { score: [m.score[0], m.score[1]], winner: m.winner ?? -1, star: starOf(m) };
  for (const [id, line] of Object.entries(m.box)) {
    const sl = (s.stats[id] ??= { g: 0, bat: emptyBat(), pitch: emptyPitch(), e: 0 });
    sl.g++;
    addBat(sl.bat, line.bat);
    addPitch(sl.pitch, line.pitch);
    sl.e += line.e;
  }
  maybeAdvancePlayoffs(s);
}

export function simGame(s: SeasonState, g: ScheduledGame): Match {
  const m = simulateMatch(matchConfigFor(s, g, null));
  recordResult(s, g, m);
  return m;
}

/** Sim every unplayed game up to (and including) `day`, except the user's own unless asked. */
export function simThrough(s: SeasonState, day: number, includeUser = false) {
  for (const g of s.games) {
    if (g.result || g.day > day) continue;
    const mine = g.away === s.userTeam || g.home === s.userTeam;
    if (mine && !includeUser) continue;
    simGame(s, g);
  }
}

// ───────────────────────────────────────────────────────── standings

export interface Standing {
  team: Team; w: number; l: number; t: number; rs: number; ra: number; pct: number; gb: number; streak: string;
}

export function standings(s: SeasonState): Record<Team['division'], Standing[]> {
  const rows: Record<string, Standing> = {};
  for (const t of TEAMS) rows[t.id] = { team: t, w: 0, l: 0, t: 0, rs: 0, ra: 0, pct: 0, gb: 0, streak: '' };
  const last: Record<string, string[]> = {};
  for (const g of s.games) {
    if (g.kind !== 'regular' || !g.result) continue;
    const [a, h] = g.result.score;
    const A = rows[g.away], H = rows[g.home];
    A.rs += a; A.ra += h; H.rs += h; H.ra += a;
    if (g.result.winner === 0) { A.w++; H.l++; (last[g.away] ??= []).push('W'); (last[g.home] ??= []).push('L'); }
    else if (g.result.winner === 1) { H.w++; A.l++; (last[g.home] ??= []).push('W'); (last[g.away] ??= []).push('L'); }
    else { A.t++; H.t++; (last[g.away] ??= []).push('T'); (last[g.home] ??= []).push('T'); }
  }
  for (const r of Object.values(rows)) {
    const gp = r.w + r.l + r.t;
    r.pct = gp ? (r.w + r.t * 0.5) / gp : 0;
    const seq = last[r.team.id] ?? [];
    if (seq.length) {
      const k = seq[seq.length - 1];
      let n = 0;
      for (let i = seq.length - 1; i >= 0 && seq[i] === k; i--) n++;
      r.streak = `${k}${n}`;
    }
  }
  const out = { 'Front Porch': [] as Standing[], 'Back Fence': [] as Standing[] };
  for (const r of Object.values(rows)) out[r.team.division].push(r);
  for (const div of Object.values(out)) {
    div.sort((a, b) => b.pct - a.pct || (b.rs - b.ra) - (a.rs - a.ra) || b.rs - a.rs);
    const lead = div[0];
    for (const r of div) r.gb = ((lead.w - r.w) + (r.l - lead.l)) / 2;
  }
  return out;
}

// ─────────────────────────────────────────────────────────── playoffs

function maybeAdvancePlayoffs(s: SeasonState) {
  const reg = s.games.filter((g) => g.kind === 'regular');
  if (reg.some((g) => !g.result)) return;
  const semis = s.games.filter((g) => g.kind === 'semi');
  const day = regularDays(s);
  if (!semis.length) {
    const st = standings(s);
    const fp = st['Front Porch'], bf = st['Back Fence'];
    // division winners host the other division's runner-up
    s.games.push({ id: `g${s.games.length}`, day, away: bf[1].team.id, home: fp[0].team.id, kind: 'semi' });
    s.games.push({ id: `g${s.games.length}`, day, away: fp[1].team.id, home: bf[0].team.id, kind: 'semi' });
    return;
  }
  if (semis.some((g) => !g.result)) return;
  const fin = s.games.find((g) => g.kind === 'final');
  const winnerOf = (g: ScheduledGame) => (g.result!.winner === 0 ? g.away : g.home);
  if (!fin) {
    // ties in the playoffs go to the home team (it's their yard, their rules)
    const [w1, w2] = semis.map(winnerOf);
    const st = standings(s);
    const pct = (id: string) => [...st['Front Porch'], ...st['Back Fence']].find((r) => r.team.id === id)!.pct;
    const [home, away] = pct(w1) >= pct(w2) ? [w1, w2] : [w2, w1];
    s.games.push({ id: `g${s.games.length}`, day: day + 1, away, home, kind: 'final' });
    return;
  }
  if (fin.result && !s.champion) s.champion = winnerOf(fin);
}

export function inPlayoffs(s: SeasonState) {
  return s.games.some((g) => g.kind !== 'regular');
}

// ─────────────────────────────────────────────────────────── leaders

export interface Leader { id: string; value: number; label: string }

export function leaders(s: SeasonState, stat: 'avg' | 'hr' | 'rbi' | 'h' | 'so' | 'era' | 'r', n = 5): Leader[] {
  const regGames = (id: string) => s.stats[id]?.g ?? 0;
  const maxG = Math.max(1, ...Object.values(s.stats).map((x) => x.g));
  const rows: Leader[] = [];
  for (const [id, l] of Object.entries(s.stats)) {
    const b = l.bat, p = l.pitch;
    switch (stat) {
      case 'avg':
        if (b.ab >= Math.max(3, maxG * 1.5)) rows.push({ id, value: b.h / b.ab, label: (b.h / b.ab).toFixed(3).replace(/^0/, '') });
        break;
      case 'hr': if (b.hr) rows.push({ id, value: b.hr, label: String(b.hr) }); break;
      case 'rbi': if (b.rbi) rows.push({ id, value: b.rbi, label: String(b.rbi) }); break;
      case 'h': if (b.h) rows.push({ id, value: b.h, label: String(b.h) }); break;
      case 'r': if (b.r) rows.push({ id, value: b.r, label: String(b.r) }); break;
      case 'so': if (p.so) rows.push({ id, value: p.so, label: String(p.so) }); break;
      case 'era':
        if (p.outs >= Math.max(6, regGames(id) * 3)) {
          const era = (p.r * s.innings * 3) / p.outs;
          rows.push({ id, value: -era, label: era.toFixed(2) });
        }
        break;
    }
  }
  rows.sort((a, b) => b.value - a.value);
  return rows.slice(0, n);
}

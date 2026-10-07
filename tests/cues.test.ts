import { describe, expect, it } from 'vitest';
import { GameSound, type CueSink } from '../src/audio/cues';
import { setErrorHandler } from '../src/audio/context';
import { kid } from '../src/data/kids';
import { team } from '../src/data/teams';
import { yard } from '../src/data/yards';
import { autoLineup } from '../src/sim/lineup';
import { Match } from '../src/sim/match';
import type { MusicTrack, PlayOpts, SfxName } from '../src/audio/types';

/** Plays a CPU game and records everything GameSound would have played. */
function listen(seed: number, innings = 3) {
  const a = team('mudcats'), h = team('comets');
  const m = new Match({
    away: { team: a, lineup: autoLineup(a.roster.map(kid)), human: false },
    home: { team: h, lineup: autoLineup(h.roster.map(kid)), human: true },
    yard: yard(h.yardId), innings, seed, difficulty: 'pro', fast: true,
  });
  // the "human" side is played by the CPU too: only the cheering perspective matters here
  m.cfg.home.human = false;
  const played: Array<{ name: SfxName; opts?: PlayOpts }> = [];
  const said: Array<{ who: string; text: string }> = [];
  const music: MusicTrack[] = [];
  const sink: CueSink = {
    play: (name, opts) => { played.push({ name, opts }); },
    speak: (who, text) => { said.push({ who, text }); return 0.8; },
    music: (t) => { music.push(t); },
  };
  const gs = new GameSound(sink);
  const quips: string[] = [];
  let steps = 0;
  const pump = () => {
    for (const e of m.events) if (e.type === 'quip') quips.push(`${e.kid}|${e.text}`);
    gs.events(m.events, m, 1);
    m.events.length = 0;
  };
  pump();
  while (m.phase !== 'over' && steps++ < 400000) {
    m.update(1 / 30);
    gs.tick(1 / 30);
    pump();
  }
  return { m, played, said, music, quips };
}

describe('game sound cues', () => {
  it('sizes every crowd reaction to the moment and says every speech bubble out loud', () => {
    for (const seed of [3, 5, 8]) {
      const { played, said, quips } = listen(seed);
      const crowd = played.filter((p) => ['cheer', 'bigCheer', 'aww', 'ooh'].includes(p.name));
      expect(crowd.length).toBeGreaterThan(5);
      for (const c of crowd) {
        const i = c.opts?.intensity ?? 0.7;
        expect(i).toBeGreaterThanOrEqual(0);
        expect(i).toBeLessThanOrEqual(1);
      }
      // a crowd that only knows one volume sounds canned: reactions must vary
      const levels = new Set(crowd.map((c) => Math.round((c.opts?.intensity ?? 0.7) * 10)));
      expect(levels.size).toBeGreaterThan(2);
      // every bubble is voiced, by the kid who says it
      for (const q of quips) expect(said.some((s) => `${s.who}|${s.text}` === q)).toBe(true);
    }
  });

  it('plays the right bounce for each surface, and a kid voices some barks', () => {
    const names = new Set<string>();
    let barks = 0;
    for (const seed of [1, 2, 3, 4, 5, 6]) {
      const { played, said, quips } = listen(seed);
      for (const p of played) names.add(p.name);
      barks += said.length - quips.length;
    }
    expect(names.has('bounce')).toBe(true);
    expect(names.has('bounceDirt')).toBe(true);
    expect(barks).toBeGreaterThan(10);
  });

  it('plays the toy-keyboard jingle between innings, never after the final out', () => {
    const { m, music } = listen(7, 3);
    expect(m.phase).toBe('over');
    expect(music.every((t) => t === 'inning')).toBe(true);
    // 3 innings: 6 half-innings, the last one ends the game (and the 3rd top may too)
    expect(music.length).toBeGreaterThanOrEqual(4);
    expect(music.length).toBeLessThanOrEqual(5);
  });

  it('never throws, whatever the sink does', () => {
    const sink: CueSink = { play: () => { throw new Error('x'); }, speak: () => { throw new Error('y'); }, music: () => undefined };
    const gs = new GameSound(sink);
    setErrorHandler(() => undefined); // reported, not thrown: keep the test output quiet
    const a = team('mudcats'), h = team('comets');
    const m = new Match({ away: { team: a, lineup: autoLineup(a.roster.map(kid)), human: false }, home: { team: h, lineup: autoLineup(h.roster.map(kid)), human: false }, yard: yard(h.yardId), innings: 1, seed: 1, difficulty: 'pro', fast: true });
    let steps = 0;
    while (m.phase !== 'over' && steps++ < 100000) {
      m.update(1 / 30);
      expect(() => gs.events(m.events, m, 0)).not.toThrow();
      expect(() => gs.caption('Chet', 'Here we go.')).not.toThrow();
      m.events.length = 0;
    }
    setErrorHandler(null);
  });
});

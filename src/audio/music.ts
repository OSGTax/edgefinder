/**
 * Step-sequencer music player. A 25 ms timer schedules notes a little ahead
 * on the AudioContext clock, so timing stays tight even when the main thread
 * is busy. In a background tab (timers throttled to ~1 Hz) it schedules
 * further ahead, and after a long stall it skips forward instead of firing a
 * burst of stale notes.
 */
import { getBus, onBus, pageHidden, reportError, type Bus } from './context';
import { Patch, rand } from './dsp';
import { INSTRUMENTS } from './instruments';
import type { Song } from './notation';
import { SONGS } from './tracks';
import type { MusicTrack } from './types';

const TICK_MS = 25;
const LOOKAHEAD = 0.1;
const HIDDEN_LOOKAHEAD = 1.5;

class Player {
  private readonly out: GainNode;
  private step = 0;
  private next = 0;
  private timer: ReturnType<typeof setInterval> | undefined;
  private stopped = false;

  constructor(
    private readonly bus: Bus,
    private readonly song: Song,
    private readonly onEnd: () => void,
  ) {
    this.out = bus.ctx.createGain();
    this.out.gain.value = 0;
    this.out.connect(bus.music);
  }

  start(fadeIn: number): void {
    const now = this.bus.ctx.currentTime;
    this.out.gain.setValueAtTime(0, now);
    this.out.gain.linearRampToValueAtTime(this.song.level, now + fadeIn);
    this.next = now + 0.05;
    this.timer = setInterval(() => this.tick(), TICK_MS);
    this.tick();
  }

  stop(fade: number): void {
    if (this.stopped) return;
    this.stopped = true;
    clearInterval(this.timer);
    const g = this.out.gain;
    const now = this.bus.ctx.currentTime;
    const v = g.value;
    g.cancelScheduledValues(now);
    g.setValueAtTime(v, now);
    g.linearRampToValueAtTime(0, now + fade);
    // notes already queued ahead keep their own cleanup; just cut this bus loose after them
    setTimeout(() => this.release(), (fade + HIDDEN_LOOKAHEAD + 0.25) * 1000);
  }

  private release(): void {
    this.out.disconnect();
    this.onEnd();
  }

  private finish(): void {
    if (this.stopped) return;
    this.stopped = true;
    clearInterval(this.timer);
    const tail = this.next - this.bus.ctx.currentTime + 2;
    setTimeout(() => this.release(), tail * 1000);
  }

  private tick(): void {
    if (this.stopped) return;
    try {
      const dt = this.song.stepDur;
      const now = this.bus.ctx.currentTime;
      if (this.next < now - 0.05) {
        const missed = Math.ceil((now - this.next) / dt);
        this.step += missed;
        this.next += missed * dt;
      }
      const horizon = now + (pageHidden() ? HIDDEN_LOOKAHEAD : LOOKAHEAD);
      while (this.next < horizon) {
        if (!this.song.loop && this.step >= this.song.length) {
          this.finish();
          return;
        }
        this.schedule(this.step, this.next);
        this.step++;
        this.next += dt;
      }
    } catch (e) {
      reportError(e);
      this.stop(0.05);
    }
  }

  private schedule(step: number, t: number): void {
    const { song } = this;
    const at = step % 4 === 2 ? t + song.swing * song.stepDur : t; // swung off-beat 8ths
    for (const part of song.parts) {
      const evs = part.steps[step % part.len];
      if (!evs) continue;
      for (const ev of evs) {
        const p = new Patch(this.bus);
        try {
          const vel = ev.vel * part.gain * rand(0.92, 1.05); // a touch of human
          INSTRUMENTS[part.inst](p, this.out, at, ev.midis, ev.len * song.stepDur * 0.92, vel);
        } finally {
          p.seal();
        }
      }
    }
  }
}

let current: { name: MusicTrack; player: Player } | null = null;
/** The looping track asked for, remembered so it can start once audio unlocks. */
let wanted: MusicTrack | null = null;

export function playMusic(name: MusicTrack): void {
  const song = (SONGS as Partial<Record<string, Song>>)[name];
  if (!song) return;
  wanted = song.loop ? name : null;
  const bus = getBus();
  if (!bus || current?.name === name) return;

  const prev = current;
  prev?.player.stop(song.loop ? 0.8 : 0.25);
  const player: Player = new Player(bus, song, () => {
    if (current?.player === player) current = null;
  });
  current = { name, player };
  player.start(song.loop ? (prev ? 0.8 : 0.3) : 0.01);
}

export function stopMusic(): void {
  wanted = null;
  current?.player.stop(0.5);
  current = null;
}

onBus(() => {
  if (wanted) playMusic(wanted);
});

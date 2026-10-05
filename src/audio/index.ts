/**
 * Grass Stain League audio: every sound and tune is synthesized with Web Audio.
 *
 *   import { audio } from './audio';
 *   canvas.addEventListener('pointerdown', () => audio.unlock());   // every gesture is fine
 *   audio.playMusic('title');            // may be called before unlock; starts once unlocked
 *   audio.play('batCrack', { intensity: 0.9, pan: -0.2 });
 *
 * Every method is a silent no-op without an AudioContext and never throws.
 */
import { setAmbience } from './ambience';
import {
  isReady,
  safely,
  setErrorHandler,
  setMusicVolume,
  setMuted,
  setSfxVolume,
  unlock,
} from './context';
import { playMusic, stopMusic } from './music';
import { SFX, playSfx } from './sfx';
import { SONGS } from './tracks';
import type { MusicTrack, PlayOpts, SfxName } from './types';

export type { MusicTrack, PlayOpts, SfxName } from './types';

export interface GameAudio {
  /** Call from a user gesture; creates/resumes the AudioContext. Safe to call repeatedly. */
  unlock(): void;
  play(name: SfxName, opts?: PlayOpts): void;
  /** Loops (except 'victory', which plays once), crossfading from whatever was playing. */
  playMusic(track: MusicTrack): void;
  stopMusic(): void;
  /** Gentle backyard ambience: wind, bird chirps, a distant lawnmower now and then. */
  setAmbience(on: boolean): void;
  setSfxVolume(v: number): void;
  setMusicVolume(v: number): void;
  setMuted(m: boolean): void;
  /** True once the AudioContext exists and is running. */
  readonly ready: boolean;
}

export const SFX_NAMES: SfxName[] = Object.keys(SFX) as SfxName[];
export const MUSIC_TRACKS: MusicTrack[] = Object.keys(SONGS) as MusicTrack[];

export const audio: GameAudio = {
  unlock: () => safely(unlock),
  play: (name, opts) => safely(() => playSfx(name, opts)),
  playMusic: (track) => safely(() => playMusic(track)),
  stopMusic: () => safely(stopMusic),
  setAmbience: (on) => safely(() => setAmbience(on)),
  setSfxVolume: (v) => safely(() => setSfxVolume(v)),
  setMusicVolume: (v) => safely(() => setMusicVolume(v)),
  setMuted: (m) => safely(() => setMuted(m)),
  get ready() {
    try {
      return isReady();
    } catch {
      return false;
    }
  },
};

/** Route internal audio errors somewhere visible (they are swallowed otherwise). */
export const setAudioErrorHandler = setErrorHandler;

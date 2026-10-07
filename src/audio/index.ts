/**
 * Grass Stain League audio: every sound, voice and tune is synthesized with Web Audio.
 *
 *   import { audio } from './audio';
 *   audio.playMusic('title');            // may be called before unlock; starts once unlocked
 *   audio.play('batCrack', { intensity: 0.9, pan: -0.2 });
 *   audio.speak('kai', 'SUNDAY! SUNDAY! SUNDAY!');   // a kid's gibberish voice
 *
 * The first touch, click or key anywhere on the page unlocks audio by itself
 * (calling audio.unlock() from a gesture is still fine). Every method is a
 * silent no-op without an AudioContext and never throws.
 */
import { setAmbience } from './ambience';
import {
  isReady,
  safely,
  setErrorHandler,
  setMusicVolume,
  setMuted,
  setSfxVolume,
  setVoiceVolume,
  unlock,
} from './context';
import { playMusic, stopMusic } from './music';
import { SFX, playSfx } from './sfx';
import { SONGS } from './tracks';
import type { MusicTrack, PlayOpts, SfxName } from './types';
import { speak, type SpeakOpts } from './voice';

export type { MusicTrack, PlayOpts, SfxName } from './types';
export type { SpeakOpts } from './voice';
export { GameSound } from './cues';

export interface GameAudio {
  /** Creates/resumes the AudioContext; call from a user gesture. Safe to call repeatedly. */
  unlock(): void;
  play(name: SfxName, opts?: PlayOpts): void;
  /**
   * Say a line in someone's gibberish voice: a kid id, 'Chet', 'Dottie',
   * 'mrsMendoza' or 'mrMendoza'. Returns how long it takes (0 if silent).
   */
  speak(who: string, text: string, opts?: SpeakOpts): number;
  /** Loops, crossfading from whatever was playing. 'inning' plays once and hands back; 'victory'/'defeat' play once. */
  playMusic(track: MusicTrack): void;
  stopMusic(): void;
  /** The backyard: breeze, birds, the grill, a sprinkler, a mower, the ice-cream truck... */
  setAmbience(on: boolean): void;
  setSfxVolume(v: number): void;
  setMusicVolume(v: number): void;
  /** Kids' and announcers' voices; 0 turns them off. */
  setVoiceVolume(v: number): void;
  setMuted(m: boolean): void;
  /** True once the AudioContext exists and is running. */
  readonly ready: boolean;
}

export const SFX_NAMES: SfxName[] = Object.keys(SFX) as SfxName[];
export const MUSIC_TRACKS: MusicTrack[] = Object.keys(SONGS) as MusicTrack[];

export const audio: GameAudio = {
  unlock: () => safely(unlock),
  play: (name, opts) => safely(() => playSfx(name, opts)),
  speak: (who, text, opts) => {
    let d = 0;
    safely(() => {
      d = speak(String(who), String(text ?? ''), opts);
    });
    return d;
  },
  playMusic: (track) => safely(() => playMusic(track)),
  stopMusic: () => safely(stopMusic),
  setAmbience: (on) => safely(() => setAmbience(on)),
  setSfxVolume: (v) => safely(() => setSfxVolume(v)),
  setMusicVolume: (v) => safely(() => setMusicVolume(v)),
  setVoiceVolume: (v) => safely(() => setVoiceVolume(v)),
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

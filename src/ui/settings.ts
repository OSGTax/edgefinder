import { load, save } from '../engine/storage';
import type { Difficulty } from '../sim/types';

export interface Settings {
  sfx: number;
  music: number;
  /** kids' and announcers' voices, 0..1 (0 = off) */
  voices: number;
  difficulty: Difficulty;
  /** how long you get to pick a throw before the CPU does it */
  autoThrow: number;
  aimAssist: 'auto' | 'on' | 'off';
  showZone: boolean;
  /** phone buzzes on contact, catches and outs (where the browser supports it) */
  haptics: boolean;
  /** the first-game coach has shown all its tips (see `replayCoach` in game/screen.ts) */
  coachDone: boolean;
  /** which coach tips have been shown */
  coachSeen: string[];
  /** phones: go full screen (and lock landscape where allowed) on the first tap */
  fullscreen: boolean;
}

const DEFAULTS: Settings = {
  sfx: 0.8,
  music: 0.5,
  voices: 0.8,
  difficulty: 'rookie',
  autoThrow: 1.6,
  aimAssist: 'auto',
  showZone: true,
  haptics: true,
  coachDone: false,
  coachSeen: [],
  fullscreen: true,
};

export const settings: Settings = { ...DEFAULTS, ...load<Partial<Settings>>('settings', {}) };

export function saveSettings() {
  save('settings', settings);
}

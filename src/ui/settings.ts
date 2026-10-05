import { load, save } from '../engine/storage';
import type { Difficulty } from '../sim/types';

export interface Settings {
  sfx: number;
  music: number;
  voice: boolean;
  difficulty: Difficulty;
  /** how long you get to pick a throw before the CPU does it */
  autoThrow: number;
  aimAssist: 'auto' | 'on' | 'off';
  showZone: boolean;
}

const DEFAULTS: Settings = {
  sfx: 0.8,
  music: 0.5,
  voice: false,
  difficulty: 'rookie',
  autoThrow: 1.6,
  aimAssist: 'auto',
  showZone: true,
};

export const settings: Settings = { ...DEFAULTS, ...load<Partial<Settings>>('settings', {}) };

export function saveSettings() {
  save('settings', settings);
}

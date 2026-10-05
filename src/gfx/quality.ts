import { load, save } from '../engine/storage';

export type QualityName = 'low' | 'medium' | 'high';

export interface Quality {
  name: QualityName;
  pixelRatio: number;
  antialias: boolean;
  shadowMap: number;
  /** real 3D grass blades on the lawn */
  grassBlades: number;
  /** leaf cards per tree */
  leaves: number;
  texSize: number;
  splatSize: number;
}

const TIERS: Record<QualityName, Omit<Quality, 'name'>> = {
  low: { pixelRatio: 1, antialias: false, shadowMap: 1024, grassBlades: 0, leaves: 700, texSize: 256, splatSize: 1024 },
  medium: { pixelRatio: 1.5, antialias: true, shadowMap: 2048, grassBlades: 26000, leaves: 1400, texSize: 512, splatSize: 2048 },
  high: { pixelRatio: 2, antialias: true, shadowMap: 4096, grassBlades: 70000, leaves: 2600, texSize: 1024, splatSize: 2048 },
};

function autoTier(): QualityName {
  if (typeof window === 'undefined') return 'medium';
  const coarse = window.matchMedia?.('(pointer: coarse)').matches;
  const small = Math.min(window.screen?.width ?? 1920, window.screen?.height ?? 1080) < 700;
  const cores = navigator.hardwareConcurrency ?? 4;
  if (coarse && small) return cores >= 6 ? 'medium' : 'low';
  return 'high';
}

export function getQuality(): Quality {
  const forced = typeof location !== 'undefined' ? new URLSearchParams(location.hash.slice(1)).get('q') : null;
  const chosen = (forced && forced in TIERS ? forced : load<QualityName | 'auto'>('quality', 'auto')) as QualityName | 'auto';
  const name = chosen === 'auto' ? autoTier() : chosen;
  const t = TIERS[name];
  const dpr = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1;
  return { name, ...t, pixelRatio: Math.min(dpr, t.pixelRatio) };
}

export function setQuality(q: QualityName | 'auto') {
  save('quality', q);
}

export function qualitySetting(): QualityName | 'auto' {
  return load<QualityName | 'auto'>('quality', 'auto');
}

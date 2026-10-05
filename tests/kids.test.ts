import { describe, expect, it } from 'vitest';
import { KIDS } from '../src/data/kids';
import { TRAIT_LABELS, type Traits } from '../src/data/types';

describe('kid traits', () => {
  const keys = Object.keys(TRAIT_LABELS) as (keyof Traits)[];
  it('every kid has all seven traits, each 1–10', () => {
    expect(keys).toHaveLength(7);
    for (const k of KIDS) for (const t of keys) {
      expect(Number.isInteger(k.traits[t])).toBe(true);
      expect(k.traits[t]).toBeGreaterThanOrEqual(1);
      expect(k.traits[t]).toBeLessThanOrEqual(10);
    }
  });
  it('no two kids share the same seven numbers', () => {
    const seen = new Set(KIDS.map((k) => keys.map((t) => k.traits[t]).join(',')));
    expect(seen.size).toBe(KIDS.length);
  });
});

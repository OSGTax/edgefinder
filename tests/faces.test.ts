import { describe, expect, it } from 'vitest';
import { KIDS } from '../src/data/kids';
import { MR_MENDOZA } from '../src/world/grownups';
import { FACE_RECIPES } from '../src/kid3d/face-recipes';

describe('face recipes', () => {
  const everyone = [...KIDS, MR_MENDOZA];

  it('gives every kid (and Mr. Mendoza) a hand-picked face', () => {
    for (const k of everyone) expect(FACE_RECIPES[k.id], k.id).toBeDefined();
  });

  it('never repeats the same eyes + brows + mouth', () => {
    const seen = new Map<string, string>();
    for (const k of everyone) {
      const r = FACE_RECIPES[k.id];
      const key = `${r.eye}|${r.brow}|${r.mouth}`;
      expect(seen.get(key), `${k.id} looks like ${seen.get(key)}`).toBeUndefined();
      seen.set(key, k.id);
    }
  });
});

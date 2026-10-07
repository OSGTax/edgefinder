import { describe, expect, it } from 'vitest';
import { TierGovernor, type GovernorAction } from '../src/gfx/governor';
import { autoTier, isSoftwareRenderer, pixelRatioFor, shortGpuName, TIERS } from '../src/gfx/quality';

const run = (g: TierGovernor, ms: number, seconds: number) => {
  const acts: GovernorAction[] = [];
  for (let t = 0; t < seconds * 1000; t += ms) {
    const a = g.frame(ms);
    if (a) acts.push(a);
  }
  return acts;
};

describe('graphics tier governor', () => {
  it('steps a slow computer down a whole tier, then shrinks resolution only at Fast', () => {
    const g = new TierGovernor({ tier: 1, min: 0, max: 2, tiers: true });
    expect(run(g, 40, 7)).toEqual(['down']);
    expect(g.tier).toBe(0);
    expect(g.tooSlow).toBe(1);
    const more = run(g, 40, 30);
    expect(more.every((a) => a === 'shrink')).toBe(true);
    expect(g.scale).toBeLessThan(1);
  });

  it('climbs on a fast machine and never retries a tier that was too slow', () => {
    const g = new TierGovernor({ tier: 1, min: 0, max: 2, tiers: true });
    expect(run(g, 16.7, 14)).toEqual(['up']);
    expect(g.tier).toBe(2);
    expect(run(g, 30, 7)).toEqual(['down']);
    expect(g.tooSlow).toBe(2);
    expect(run(g, 16.7, 60)).toEqual([]);
    expect(g.tier).toBe(1);
  });

  it('ignores lone hitches and hidden-tab gaps', () => {
    const g = new TierGovernor({ tier: 1, min: 0, max: 1, tiers: true });
    const acts: GovernorAction[] = [];
    for (let i = 0; i < 1500; i++) {
      const a = g.frame(i % 40 === 0 ? 120 : i % 500 === 0 ? 5000 : 16.7);
      if (a) acts.push(a);
    }
    expect(acts).toEqual([]);
  });

  it('only touches resolution when the player picked a tier', () => {
    const g = new TierGovernor({ tier: 2, min: 0, max: 2, tiers: false });
    const acts = run(g, 45, 20);
    expect(acts.length).toBeGreaterThan(0);
    expect(acts.every((a) => a === 'shrink')).toBe(true);
    expect(g.tier).toBe(2);
  });

  it('judges the battery saver against 30 fps', () => {
    const g = new TierGovernor({ tier: 1, min: 0, max: 2, tiers: true });
    g.setTarget(1000 / 30);
    expect(run(g, 33.4, 20)).toEqual([]);
  });
});

describe('device detection', () => {
  it('spots software renderers', () => {
    for (const s of [
      'ANGLE (Google, Vulkan 1.3.0 (SwiftShader Device (Subzero) (0x0000C0DE)), SwiftShader driver)',
      'llvmpipe (LLVM 15.0.7, 256 bits)',
      'ANGLE (Microsoft, Microsoft Basic Render Driver Direct3D11 vs_5_0 ps_5_0, D3D11)',
    ]) expect(isSoftwareRenderer(s)).toBe(true);
    for (const s of ['ANGLE (NVIDIA, NVIDIA GeForce RTX 3060 (0x00002504) Direct3D11 vs_5_0 ps_5_0, D3D11)', 'Apple GPU', 'Mali-G78', 'Adreno (TM) 650'])
      expect(isSoftwareRenderer(s)).toBe(false);
  });

  it('shortens GPU names for the readout', () => {
    expect(shortGpuName('ANGLE (NVIDIA, NVIDIA GeForce RTX 3060 (0x00002504) Direct3D11 vs_5_0 ps_5_0, D3D11)')).toBe('NVIDIA GeForce RTX 3060');
    expect(shortGpuName('Apple GPU')).toBe('Apple GPU');
  });

  it('starts computers on Balanced and software renderers on Fast', () => {
    expect(autoTier({ gpu: 'x', software: false, phone: false, cores: 16 })).toBe('medium');
    expect(autoTier({ gpu: 'x', software: true, phone: false, cores: 16 })).toBe('low');
    expect(autoTier({ gpu: 'x', software: false, phone: true, cores: 4 })).toBe('low');
  });

  it('caps pixels on big screens', () => {
    const pr = pixelRatioFor(TIERS.medium, 3840, 2160);
    expect(3840 * 2160 * pr * pr).toBeLessThanOrEqual(TIERS.medium.maxPixels * 1.01);
  });
});

describe('resolution hysteresis', () => {
  it('does not bounce between two resolutions', () => {
    const g = new TierGovernor({ tier: 0, min: 0, max: 0, tiers: true });
    const acts: GovernorAction[] = [];
    // slow at full size, fine one step down
    for (let i = 0; i < 6000; i++) {
      const a = g.frame(g.scale > 0.9 ? 30 : 16.7);
      if (a) acts.push(a);
    }
    expect(acts).toEqual(['shrink', 'grow', 'shrink']);
  });
});

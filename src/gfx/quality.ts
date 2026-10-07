import { load, save } from '../engine/storage';

// Graphics tiers and how a device picks one.
//
// Every device starts on Balanced (older phones and software renderers on
// Fast), then `TierGovernor` (gfx/governor.ts) watches real frame times and
// steps the whole tier down or up. What it learns is remembered per graphics
// chip, so the next visit builds the yard at the right tier straight away.

export type QualityName = 'low' | 'medium' | 'high';
export const TIER_ORDER: QualityName[] = ['low', 'medium', 'high'];
export const TIER_LABEL: Record<QualityName, string> = { low: 'Fast', medium: 'Balanced', high: 'Beautiful' };

export interface Quality {
  name: QualityName;
  pixelRatio: number;
  /** most pixels drawn per frame (keeps 4K monitors and 3× phones in budget) */
  maxPixels: number;
  antialias: boolean;
  shadowMap: number;
  /** real 3D grass blades on the lawn */
  grassBlades: number;
  /** leaf cards per tree */
  leaves: number;
  texSize: number;
  splatSize: number;
  /** soft (PCFSoft) shadow filtering; off = plain PCF, cheaper per pixel */
  softShadows: boolean;
  /** ground mesh subdivisions per side (the far hills need some) */
  groundSeg: number;
  /** how many of the kids nearest the camera cast real shadows (the rest get a soft contact shadow) */
  kidShadows: number;
  /** kids farther than this from the camera (ft) use the lite model */
  liteDist: number;
}

export type TierSpec = Omit<Quality, 'name'>;

export const TIERS: Record<QualityName, TierSpec> = {
  low: {
    pixelRatio: 1, maxPixels: 1.0e6, antialias: false, shadowMap: 1024, grassBlades: 0, leaves: 700, texSize: 256, splatSize: 1024,
    softShadows: false, groundSeg: 96, kidShadows: 4, liteDist: 40,
  },
  medium: {
    pixelRatio: 1.5, maxPixels: 2.1e6, antialias: true, shadowMap: 2048, grassBlades: 26000, leaves: 1400, texSize: 512, splatSize: 2048,
    softShadows: true, groundSeg: 160, kidShadows: 10, liteDist: 70,
  },
  high: {
    pixelRatio: 2, maxPixels: 4.2e6, antialias: true, shadowMap: 4096, grassBlades: 70000, leaves: 2600, texSize: 1024, splatSize: 2048,
    softShadows: true, groundSeg: 160, kidShadows: 19, liteDist: 110,
  },
};

// ───────────────────────────────────────────────────────── the device

export interface DeviceInfo {
  /** the graphics chip the browser reports ('' if it won't say) */
  gpu: string;
  /** drawing without a GPU (SwiftShader, llvmpipe, Basic Render Driver...) */
  software: boolean;
  /** a phone-sized touch screen */
  phone: boolean;
  cores: number;
}

const SOFTWARE_RE = /swiftshader|llvmpipe|softpipe|lavapipe|basic render|microsoft basic|software|mesa offscreen|gdi generic/i;

/** True for renderer strings that mean "no graphics card is helping". */
export function isSoftwareRenderer(name: string): boolean {
  return SOFTWARE_RE.test(name);
}

/** Tidy a renderer string for display: "ANGLE (NVIDIA, NVIDIA GeForce RTX 3060 (0x...) Direct3D11 ...)" → "NVIDIA GeForce RTX 3060". */
export function shortGpuName(name: string): string {
  let s = name.trim();
  const angle = /^ANGLE \((.*)\)$/.exec(s);
  if (angle) {
    const parts = angle[1].split(', ');
    s = parts.length >= 2 ? parts[1] : parts[0];
  }
  s = s.replace(/\s*\(0x[0-9a-f]+\)/gi, '').replace(/\s*(Direct3D|OpenGL|Vulkan|Metal)\S*.*$/i, '').replace(/\s+/g, ' ').trim();
  return s || name;
}

/** The renderer string from a live WebGL context. */
export function gpuName(gl: WebGLRenderingContext | WebGL2RenderingContext): string {
  try {
    const dbg = gl.getExtension('WEBGL_debug_renderer_info');
    const raw = dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
    return typeof raw === 'string' ? raw : '';
  } catch {
    return '';
  }
}

let device: DeviceInfo | null = null;

/** Probe the device once (a throwaway WebGL context, released straight away). */
export function deviceInfo(): DeviceInfo {
  if (device) return device;
  const info: DeviceInfo = { gpu: '', software: false, phone: false, cores: 4 };
  if (typeof window === 'undefined' || typeof document === 'undefined') return (device = info);
  const coarse = window.matchMedia?.('(pointer: coarse)').matches ?? false;
  const small = Math.min(window.screen?.width ?? 1920, window.screen?.height ?? 1080) < 700;
  info.phone = coarse && small;
  info.cores = navigator.hardwareConcurrency ?? 4;
  try {
    const c = document.createElement('canvas');
    c.width = c.height = 1;
    // a browser that would only give us a slow (software) context refuses this one
    const fast = c.getContext('webgl2', { failIfMajorPerformanceCaveat: true }) as WebGL2RenderingContext | null;
    const gl = fast ?? (c.getContext('webgl2') as WebGL2RenderingContext | null);
    if (gl) {
      info.gpu = gpuName(gl);
      info.software = !fast || isSoftwareRenderer(info.gpu);
      gl.getExtension('WEBGL_lose_context')?.loseContext();
    }
  } catch {
    /* no WebGL probe: assume a normal GPU and let the governor sort it out */
  }
  return (device = info);
}

// ───────────────────────────────────────────────────────── remembered tier

interface Learned {
  gpu: string;
  tier: QualityName;
  /** lowest tier that was too slow here (never climb back to it automatically) */
  tooSlow: QualityName | null;
}

const LEARNED_KEY = 'gfx:learned';

function learned(d: DeviceInfo): Learned | null {
  const l = load<Learned | null>(LEARNED_KEY, null);
  if (!l || l.gpu !== d.gpu || !(l.tier in TIERS)) return null;
  return l;
}

/** Remember what the governor found for this graphics chip. */
export function rememberTier(tier: QualityName, tooSlow: QualityName | null) {
  save(LEARNED_KEY, { gpu: deviceInfo().gpu, tier, tooSlow } satisfies Learned);
}

/** The highest tier the governor may climb to automatically on this device. */
export function autoCeiling(d = deviceInfo()): QualityName {
  if (d.software) return 'low';
  const slow = learned(d)?.tooSlow;
  const top: QualityName = d.phone ? 'medium' : 'high'; // phones: battery and heat matter more than the last bit of shine
  if (!slow) return top;
  const i = Math.min(TIER_ORDER.indexOf(top), TIER_ORDER.indexOf(slow) - 1);
  return TIER_ORDER[Math.max(0, i)];
}

/** The tier to start on when the setting is Auto. */
export function autoTier(d = deviceInfo()): QualityName {
  if (d.software) return 'low';
  const l = learned(d);
  if (l) return l.tier;
  if (d.phone && d.cores < 6) return 'low';
  return 'medium';
}

/** Tier from the URL (`q=low|medium|high`), for screenshots and testing. */
export function forcedTier(): QualityName | null {
  if (typeof location === 'undefined') return null;
  const f = new URLSearchParams(location.hash.slice(1)).get('q');
  return f && f in TIERS ? (f as QualityName) : null;
}

export function tierSpec(name: QualityName): Quality {
  return { name, ...TIERS[name] };
}

export function getQuality(): Quality {
  const chosen = forcedTier() ?? qualitySetting();
  return forDevice(tierSpec(chosen === 'auto' ? autoTier() : chosen));
}

/**
 * Phone budgets: plain PCF shadows (soft filtering is the priciest per-pixel
 * cost on mobile GPUs), fewer grass blades on a smaller screen, and a shadow
 * map no bigger than 2048 (64 MB of iOS Safari's tight GPU memory at 4096).
 */
export function forDevice(q: Quality, d = deviceInfo()): Quality {
  if (!d.phone) return q;
  return { ...q, softShadows: false, grassBlades: Math.round(q.grassBlades * 0.6), shadowMap: Math.min(2048, q.shadowMap), texSize: Math.min(512, q.texSize) };
}

/** Device pixel ratio to render at for a tier and canvas size (CSS px). */
export function pixelRatioFor(q: TierSpec, w: number, h: number, scale = 1): number {
  const dpr = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1;
  const byBudget = Math.sqrt(q.maxPixels / Math.max(1, w * h));
  return Math.max(0.5, Math.min(dpr, q.pixelRatio, byBudget) * scale);
}

export function setQuality(q: QualityName | 'auto') {
  save('quality', q);
}

export function qualitySetting(): QualityName | 'auto' {
  const v = load<QualityName | 'auto'>('quality', 'auto');
  return v === 'auto' || v in TIERS ? v : 'auto';
}

// ───────────────────────────────────────────────────────── player prefs

export interface GfxPrefs {
  /** fps + graphics chip in a corner */
  readout: boolean;
  /** battery saver: draw at most 30 frames a second */
  cap30: boolean;
}

export function gfxPrefs(): GfxPrefs {
  return { readout: false, cap30: false, ...load<Partial<GfxPrefs>>('gfx:prefs', {}) };
}

export function setGfxPrefs(p: Partial<GfxPrefs>) {
  save('gfx:prefs', { ...gfxPrefs(), ...p });
  for (const f of prefListeners) f();
}

const prefListeners: (() => void)[] = [];
export function onGfxPrefs(f: () => void) { prefListeners.push(f); }

/** One line for players whose browser draws without the graphics card; null when all is well. */
export function softwareTip(): string | null {
  if (!deviceInfo().software) return null;
  return 'Your browser is drawing without your graphics card, so the game is on Fast. Turn on "Use graphics acceleration" in your browser\'s settings and restart it for a much smoother game.';
}

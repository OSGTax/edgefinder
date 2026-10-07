import { n2, rng, sym } from './rand';

// A few drawings made in code at startup and handed to CSS as custom
// properties on :root: the marker loop that circles a chosen option
// (`--scribble`) and a wavy underline. The cartoon look is otherwise flat
// colour; without these, everything still works.

/** A loose hand-drawn marker loop (for circling the chosen option), as an SVG data URL. */
export function scribbleLoop(seed: number, color = '#e0452c'): string {
  const r = rng(seed);
  const pts: string[] = [];
  const turns = 1.25;
  const steps = 26;
  for (let i = 0; i <= steps; i++) {
    const t = (i / steps) * turns * Math.PI * 2 - 2.2;
    const rx = 47 + sym(r) * 1.6 + i * 0.08, ry = 40 + sym(r) * 1.6 - i * 0.05;
    pts.push(`${n2(50 + Math.cos(t) * rx)} ${n2(50 + Math.sin(t) * ry)}`);
  }
  const d = `M${pts[0]} ` + pts.slice(1).map((p) => `L${p}`).join(' ');
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" preserveAspectRatio="none"><path d="${d}" fill="none" stroke="${color}" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round" vector-effect="non-scaling-stroke"/></svg>`;
  return `url("data:image/svg+xml,${encodeURIComponent(svg)}")`;
}

/** A wavy marker underline, as an SVG data URL. */
function underline(seed: number, color: string): string {
  const r = rng(seed);
  let d = 'M2 6';
  for (let x = 12; x <= 198; x += 14) d += ` Q${x - 7} ${n2(6 + sym(r) * 3.4)} ${x} ${n2(6 + sym(r) * 1.2)}`;
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 12" preserveAspectRatio="none"><path d="${d}" fill="none" stroke="${color}" stroke-width="3" stroke-linecap="round" vector-effect="non-scaling-stroke"/></svg>`;
  return `url("data:image/svg+xml,${encodeURIComponent(svg)}")`;
}

let installed = false;

/** Paint the drawings and publish them as CSS custom properties. Safe to call twice. */
export function installTextures() {
  if (installed || typeof document === 'undefined') return;
  installed = true;
  const vars: Record<string, string> = {
    '--scribble': scribbleLoop(7),
    '--scribble-blue': scribbleLoop(8, '#2f5ea8'),
    '--underline': underline(3, '#e0452c'),
  };
  const root = document.documentElement.style;
  for (const [k, v] of Object.entries(vars)) root.setProperty(k, v);
}

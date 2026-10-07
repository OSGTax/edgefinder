import { n2, rng, sym } from './rand';

// Small tiling textures painted in code at startup and handed to CSS as custom
// properties on :root (`--tex-paper`, `--tex-cardboard`, ...). They are painted
// neutral (light greys with transparent gaps) and tinted by the surface's
// background colour with `background-blend-mode: multiply`, so one tile serves
// every colour. Without them (no canvas) the surfaces fall back to flat colour.

function tile(size: number, paint: (g: CanvasRenderingContext2D, r: () => number) => void, seed: number): string {
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const g = c.getContext('2d');
  if (!g) return 'none';
  paint(g, rng(seed));
  return `url(${c.toDataURL('image/png')})`;
}

/** Speckle: n dots of random grey and size. */
function speckle(g: CanvasRenderingContext2D, r: () => number, size: number, n: number, lo: number, hi: number, maxR: number, alpha: number) {
  for (let i = 0; i < n; i++) {
    const v = Math.round(lo + r() * (hi - lo));
    g.fillStyle = `rgba(${v},${v},${v},${alpha * (0.4 + r() * 0.6)})`;
    const x = r() * size, y = r() * size, rr = 0.3 + r() * maxR;
    g.beginPath(); g.arc(x, y, rr, 0, Math.PI * 2); g.fill();
  }
}

/** Short fibres, wrapped around the tile edges so it tiles. */
function fibres(g: CanvasRenderingContext2D, r: () => number, size: number, n: number, len: number, shade: number, alpha: number, angle = -1) {
  g.lineCap = 'round';
  for (let i = 0; i < n; i++) {
    const a = angle < 0 ? r() * Math.PI : angle + sym(r) * 0.25;
    const l = len * (0.4 + r());
    const x = r() * size, y = r() * size;
    const v = Math.round(shade + sym(r) * 30);
    g.strokeStyle = `rgba(${v},${v},${v},${alpha * (0.3 + r() * 0.7)})`;
    g.lineWidth = 0.5 + r() * 0.8;
    for (const ox of [-size, 0, size]) for (const oy of [-size, 0, size]) {
      g.beginPath();
      g.moveTo(x + ox, y + oy);
      g.quadraticCurveTo(x + ox + Math.cos(a) * l * 0.5 + sym(r) * 2, y + oy + Math.sin(a) * l * 0.5 + sym(r) * 2, x + ox + Math.cos(a) * l, y + oy + Math.sin(a) * l);
      g.stroke();
    }
  }
}

/** A loose hand-drawn marker loop (for circling the chosen option), as an SVG data URL. */
export function scribbleLoop(seed: number, color = '#c8372d'): string {
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

/** Paint the textures and publish them as CSS custom properties. Safe to call twice. */
export function installTextures() {
  if (installed || typeof document === 'undefined') return;
  installed = true;
  const S = 160;
  const vars: Record<string, string> = {
    // poster board / index card / jersey paper: fine grain and a few fibres
    '--tex-paper': tile(S, (g, r) => {
      g.fillStyle = '#fff'; g.fillRect(0, 0, S, S);
      speckle(g, r, S, 900, 200, 245, 0.9, 0.35);
      fibres(g, r, S, 50, 9, 205, 0.35);
    }, 11),
    // corrugated cardboard: soft vertical flutes, fibres, the odd dark fleck
    '--tex-cardboard': tile(S, (g, r) => {
      g.fillStyle = '#fff'; g.fillRect(0, 0, S, S);
      for (let x = 0; x < S; x += 10) {
        const gr = g.createLinearGradient(x, 0, x + 10, 0);
        gr.addColorStop(0, 'rgba(150,150,150,0.10)'); gr.addColorStop(0.5, 'rgba(255,255,255,0)'); gr.addColorStop(1, 'rgba(150,150,150,0.10)');
        g.fillStyle = gr; g.fillRect(x, 0, 10, S);
      }
      fibres(g, r, S, 160, 14, 170, 0.32);
      speckle(g, r, S, 400, 120, 210, 1.2, 0.3);
      speckle(g, r, S, 14, 60, 110, 1.4, 0.5);
    }, 22),
    // felt: dense, soft, fuzzy
    '--tex-felt': tile(S, (g, r) => {
      g.fillStyle = '#fff'; g.fillRect(0, 0, S, S);
      fibres(g, r, S, 1400, 5, 190, 0.28);
      speckle(g, r, S, 600, 160, 240, 1.3, 0.25);
    }, 33),
    // chalkboard: white smudges and old half-erased strokes on transparent
    '--tex-chalkdust': tile(256, (g, r) => {
      for (let i = 0; i < 14; i++) {
        const x = r() * 256, y = r() * 256, rad = 20 + r() * 60;
        const gr = g.createRadialGradient(x, y, 0, x, y, rad);
        gr.addColorStop(0, `rgba(255,255,255,${0.035 + r() * 0.04})`); gr.addColorStop(1, 'rgba(255,255,255,0)');
        g.fillStyle = gr; g.fillRect(x - rad, y - rad, rad * 2, rad * 2);
      }
      g.lineCap = 'round';
      for (let i = 0; i < 9; i++) {
        g.strokeStyle = `rgba(255,255,255,${0.03 + r() * 0.04})`;
        g.lineWidth = 6 + r() * 14;
        g.beginPath();
        const x = r() * 256, y = r() * 256;
        g.moveTo(x, y);
        g.bezierCurveTo(x + sym(r) * 80, y + sym(r) * 30, x + sym(r) * 80, y + sym(r) * 30, x + sym(r) * 120, y + sym(r) * 20);
        g.stroke();
      }
      speckle(g, r, 256, 500, 230, 255, 0.8, 0.12);
    }, 44),
    // lawn: mowing stripes are done in CSS; this adds clippings and clover
    '--tex-lawn': tile(S, (g, r) => {
      g.fillStyle = '#fff'; g.fillRect(0, 0, S, S);
      fibres(g, r, S, 700, 6, 180, 0.4, Math.PI * 0.45);
      speckle(g, r, S, 300, 150, 230, 1.5, 0.3);
    }, 55),
    '--scribble': scribbleLoop(7),
    '--scribble-blue': scribbleLoop(8, '#2f5ea8'),
    '--underline': underline(3, '#c8372d'),
  };
  const root = document.documentElement.style;
  for (const [k, v] of Object.entries(vars)) root.setProperty(k, v);
}

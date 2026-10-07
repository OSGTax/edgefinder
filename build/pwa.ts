/// <reference types="node" />
import { deflateSync } from 'node:zlib';
import { createHash } from 'node:crypto';
import type { Plugin } from 'vite';

// The phone-app shell, generated at build time: the web app manifest, the
// home-screen icons (painted pixel by pixel in code and encoded as PNG right
// here, no image files in the repo), an SVG favicon and the service worker
// that makes the game work offline.
//
// Service worker rules (never strand a player on an old build):
// - the page itself (index.html) is network-first, so a new deploy is picked up
//   on the next launch; the cached copy is only used offline;
// - hashed files under assets/ are cache-first (their names change when they do);
// - each build gets its own cache, precached on install; the old caches are
//   deleted when the new worker takes over.
// All paths are relative: the site lives under a subpath on GitHub Pages.

const NAME = 'Grass Stain League';
const SHORT = 'Grass Stain';
const THEME = '#3f6f31';

// ── a tiny painter: signed distances, 4×4 supersampling ──────────────────────

type RGBA = [number, number, number, number];
const hex = (s: string, a = 1): RGBA => [parseInt(s.slice(1, 3), 16), parseInt(s.slice(3, 5), 16), parseInt(s.slice(5, 7), 16), a];

function hash2(x: number, y: number) {
  const s = Math.sin(x * 127.1 + y * 311.7) * 43758.5453;
  return s - Math.floor(s);
}
function noise(x: number, y: number) {
  const ix = Math.floor(x), iy = Math.floor(y), fx = x - ix, fy = y - iy;
  const u = fx * fx * (3 - 2 * fx), v = fy * fy * (3 - 2 * fy);
  const a = hash2(ix, iy), b = hash2(ix + 1, iy), c = hash2(ix, iy + 1), d = hash2(ix + 1, iy + 1);
  return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
}

/** Colour of the icon at (x, y) in a unit square (0..1). `safe` shrinks the ball for maskable icons. */
function iconColor(x: number, y: number, safe: boolean): RGBA {
  // lawn with mowing stripes
  const stripe = Math.floor((x * 0.9 + y * 0.35) * 5) % 2;
  const grain = noise(x * 60, y * 60) * 0.08 - 0.04;
  let col: RGBA = stripe ? hex('#6fa548') : hex('#5f9640');
  col = [col[0] * (1 + grain), col[1] * (1 + grain), col[2] * (1 + grain), 1];
  const R = safe ? 0.29 : 0.355;
  const cx = 0.5, cy = 0.5;
  const mix = (c: RGBA, a: number) => { col = [col[0] + (c[0] - col[0]) * a, col[1] + (c[1] - col[1]) * a, col[2] + (c[2] - col[2]) * a, 1]; };
  // shadow on the grass
  const ds = Math.hypot((x - cx - R * 0.12) / 1.05, y - cy - R * 0.16) - R;
  if (ds < 0) mix(hex('#24401c'), 0.45);
  const d = Math.hypot(x - cx, y - cy) - R;
  const ink = R * 0.075;
  if (d < ink) {
    if (d > 0) { mix(hex('#26211c'), 1); return col; }
    // the ball: off-white leather, a touch of shading
    const sh = Math.hypot(x - cx + R * 0.35, y - cy + R * 0.4) / (R * 2);
    col = hex('#f6f0df');
    mix(hex('#cfc4a8'), Math.min(1, Math.max(0, sh - 0.35)) * 0.8);
    // the grass stain: a smudgy green-brown blob on the lower left
    const sx = x - (cx - R * 0.32), sy = y - (cy + R * 0.38);
    const blob = Math.hypot(sx * 1.2, sy * 1.7) - R * (0.34 + 0.12 * noise(x * 18, y * 18));
    if (blob < 0) mix(hex('#6f8a3a'), Math.min(0.75, -blob / (R * 0.12)) * (0.65 + 0.35 * noise(x * 40, y * 40)));
    // two seams: arcs of circles centred off to each side, with stitches
    for (const side of [-1, 1]) {
      const scx = cx + side * R * 1.32, scy = cy;
      const sr = R * 0.95;
      const dd = Math.hypot(x - scx, y - scy) - sr;
      const ang = Math.atan2(y - scy, x - scx);
      if (Math.abs(dd) < R * 0.035) mix(hex('#c8372d'), 1);
      // stitches: short ticks across the seam
      const t = (ang * 7) / Math.PI;
      const tick = Math.abs(t - Math.round(t));
      if (Math.abs(dd) < R * 0.12 && tick < 0.09) mix(hex('#c8372d'), 0.95);
    }
    return col;
  }
  return col;
}

function paint(size: number, safe: boolean): Uint8Array {
  const px = new Uint8Array(size * size * 4);
  const S = 4;
  for (let j = 0; j < size; j++) {
    for (let i = 0; i < size; i++) {
      let r = 0, g = 0, b = 0;
      for (let sj = 0; sj < S; sj++) for (let si = 0; si < S; si++) {
        const c = iconColor((i + (si + 0.5) / S) / size, (j + (sj + 0.5) / S) / size, safe);
        r += c[0]; g += c[1]; b += c[2];
      }
      const o = (j * size + i) * 4;
      px[o] = Math.min(255, Math.round(r / (S * S)));
      px[o + 1] = Math.min(255, Math.round(g / (S * S)));
      px[o + 2] = Math.min(255, Math.round(b / (S * S)));
      px[o + 3] = 255;
    }
  }
  return px;
}

// ── PNG encoding ─────────────────────────────────────────────────────────────

const CRC = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();
function crc32(buf: Uint8Array) {
  let c = 0xffffffff;
  for (let i = 0; i < buf.length; i++) c = CRC[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}
function chunk(type: string, data: Uint8Array) {
  const out = Buffer.alloc(12 + data.length);
  out.writeUInt32BE(data.length, 0);
  out.write(type, 4, 'ascii');
  Buffer.from(data).copy(out, 8);
  out.writeUInt32BE(crc32(out.subarray(4, 8 + data.length)), 8 + data.length);
  return out;
}
export function png(size: number, rgba: Uint8Array): Buffer {
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(size, 0); ihdr.writeUInt32BE(size, 4);
  ihdr[8] = 8; ihdr[9] = 6; ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0;
  const raw = Buffer.alloc(size * (size * 4 + 1));
  for (let y = 0; y < size; y++) {
    raw[y * (size * 4 + 1)] = 0;
    Buffer.from(rgba.buffer, rgba.byteOffset + y * size * 4, size * 4).copy(raw, y * (size * 4 + 1) + 1);
  }
  return Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk('IHDR', ihdr), chunk('IDAT', deflateSync(raw, { level: 9 })), chunk('IEND', new Uint8Array(0))]);
}

// ── the files ────────────────────────────────────────────────────────────────

const FAVICON = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><rect width="64" height="64" rx="14" fill="${THEME}"/><circle cx="32" cy="32" r="22" fill="#f6f0df" stroke="#26211c" stroke-width="3.5"/><path d="M21 15c6 10 6 24 0 34M43 15c-6 10-6 24 0 34" fill="none" stroke="#c8372d" stroke-width="2.6" stroke-linecap="round" stroke-dasharray="1.5 3.5"/><ellipse cx="25" cy="41" rx="7" ry="5" fill="#6f8a3a" opacity=".6"/></svg>`;

function manifest() {
  return JSON.stringify({
    name: NAME,
    short_name: SHORT,
    description: 'Backyard baseball, played by very serious kids.',
    id: './',
    start_url: './',
    scope: './',
    display: 'fullscreen',
    display_override: ['fullscreen', 'standalone'],
    orientation: 'landscape',
    background_color: THEME,
    theme_color: THEME,
    categories: ['games', 'sports'],
    icons: [
      { src: 'icon-192.png', sizes: '192x192', type: 'image/png', purpose: 'any' },
      { src: 'icon-512.png', sizes: '512x512', type: 'image/png', purpose: 'any' },
      { src: 'icon-maskable-512.png', sizes: '512x512', type: 'image/png', purpose: 'maskable' },
      { src: 'favicon.svg', sizes: 'any', type: 'image/svg+xml' },
    ],
  }, null, 2);
}

function serviceWorker(version: string, files: string[]) {
  return `// Grass Stain League offline worker (generated at build time, ${version}).
const CACHE = 'gsl-${version}';
const PRECACHE = ${JSON.stringify(['./', ...files.map((f) => `./${f}`)])};

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(PRECACHE)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (e) => {
  e.waitUntil(caches.keys()
    .then((keys) => Promise.all(keys.filter((k) => k.startsWith('gsl-') && k !== CACHE).map((k) => caches.delete(k))))
    .then(() => self.clients.claim()));
});

self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  if (url.origin !== location.origin) return;
  const scope = new URL(self.registration.scope);
  const path = url.pathname.slice(scope.pathname.length);
  // the page: network first, so a new build is used as soon as it's online
  if (req.mode === 'navigate' || path === '' || path === 'index.html') {
    // no-cache: always ask the server (the HTTP cache may hold an old page for minutes)
    e.respondWith(fetch(req.url, { cache: 'no-cache', credentials: 'same-origin' }).then((res) => {
      if (res.ok) { const copy = res.clone(); caches.open(CACHE).then((c) => c.put('./', copy)); }
      return res;
    }).catch(() => caches.match('./', { ignoreSearch: true }).then((r) => r || caches.match('./index.html'))));
    return;
  }
  // hashed build files never change: cache first
  if (path.startsWith('assets/')) {
    e.respondWith(caches.match(req).then((hit) => hit || fetch(req).then((res) => {
      if (res.ok) { const copy = res.clone(); caches.open(CACHE).then((c) => c.put(req, copy)); }
      return res;
    })));
    return;
  }
  // everything else (icons, manifest): cached copy now, refreshed in the background
  e.respondWith(caches.match(req).then((hit) => {
    const net = fetch(req).then((res) => {
      if (res.ok) { const copy = res.clone(); caches.open(CACHE).then((c) => c.put(req, copy)); }
      return res;
    });
    return hit ? (net.catch(() => hit), hit) : net;
  }));
});
`;
}

const HEAD_TAGS = [
  { tag: 'link', attrs: { rel: 'manifest', href: './manifest.webmanifest' } },
  { tag: 'link', attrs: { rel: 'icon', href: './favicon.svg', type: 'image/svg+xml' } },
  { tag: 'link', attrs: { rel: 'apple-touch-icon', href: './apple-touch-icon.png' } },
];

export function pwa(): Plugin {
  const icons = new Map<string, Buffer>();
  const iconFiles = () => {
    if (!icons.size) {
      icons.set('icon-192.png', png(192, paint(192, false)));
      icons.set('icon-512.png', png(512, paint(512, false)));
      icons.set('icon-maskable-512.png', png(512, paint(512, true)));
      icons.set('apple-touch-icon.png', png(180, paint(180, true)));
    }
    return icons;
  };
  return {
    name: 'gsl-pwa',
    transformIndexHtml: { order: 'post', handler: () => HEAD_TAGS.map((t) => ({ ...t, injectTo: 'head' as const })) },
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const name = (req.url ?? '').split('?')[0].split('/').pop() ?? '';
        if (name === 'manifest.webmanifest') { res.setHeader('Content-Type', 'application/manifest+json'); res.end(manifest()); return; }
        if (name === 'favicon.svg') { res.setHeader('Content-Type', 'image/svg+xml'); res.end(FAVICON); return; }
        const icon = iconFiles().get(name);
        if (icon) { res.setHeader('Content-Type', 'image/png'); res.end(icon); return; }
        next();
      });
    },
    generateBundle(_opts, bundle) {
      const emit = (fileName: string, source: string | Uint8Array) => this.emitFile({ type: 'asset', fileName, source });
      emit('manifest.webmanifest', manifest());
      emit('favicon.svg', FAVICON);
      for (const [name, buf] of iconFiles()) emit(name, buf);
      const files = Object.keys(bundle).filter((f) => !f.endsWith('.map')).sort();
      const all = [...new Set([...files, 'manifest.webmanifest', 'favicon.svg', ...iconFiles().keys()])].filter((f) => f !== 'sw.js');
      const version = createHash('sha256').update(files.join('|')).digest('hex').slice(0, 10);
      emit('sw.js', serviceWorker(version, all));
    },
  };
}

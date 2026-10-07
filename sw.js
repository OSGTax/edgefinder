// Grass Stain League offline worker (generated at build time, 5dedbc43da).
const CACHE = 'gsl-5dedbc43da';
const PRECACHE = ["./","./apple-touch-icon.png","./assets/app-Bsi4MBPj.js","./assets/index-C2UZDfjB.css","./assets/index-DoCA-R4e.js","./assets/index-DujeXJwF.js","./assets/soundboard--0Jo4KqY.js","./assets/three-B2isxR2I.js","./favicon.svg","./icon-192.png","./icon-512.png","./icon-maskable-512.png","./manifest.webmanifest"];

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

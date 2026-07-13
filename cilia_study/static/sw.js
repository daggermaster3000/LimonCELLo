const CACHE = 'cilia-consensus-v4';
const SHELL = ['/', '/index.html', '/style.css', '/app.js',
  '/manifest.webmanifest', '/icon-192.png', '/icon-512.png'];

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(SHELL)).then(() => self.skipWaiting()));
});
self.addEventListener('activate', (e) => {
  e.waitUntil(caches.keys().then(ks =>
    Promise.all(ks.filter(k => k !== CACHE).map(k => caches.delete(k)))).then(() => self.clients.claim()));
});
self.addEventListener('fetch', (e) => {
  const url = new URL(e.request.url);
  // always hit the network for API + ROI images (fresh votes)
  if (url.pathname.startsWith('/api/')) {
    e.respondWith(fetch(e.request).catch(() =>
      new Response('{"offline":true}', { headers: { 'Content-Type': 'application/json' } })));
    return;
  }
  // cache-first for the static shell
  e.respondWith(caches.match(e.request).then(r => r || fetch(e.request)));
});
